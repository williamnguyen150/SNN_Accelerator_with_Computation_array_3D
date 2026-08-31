#include "SNN_Accelerator.h"
#include "Sconv_layer_param.h"

class SNN_Hardware_Scheduler {
private:
    SpikeRF spike_rf[Num_Cores];
    MultiBankMemory core_mems[Num_Cores];
    ComputationArray3D core_array3d[Num_Cores];
    NonZeroDataFetcher core_fetchers[Num_Cores];
    AsyncCoreMemRouter mem_router;
    V_reg main_V_reg;
    StatisticalData metrics;

public:
    void SConv_Scheduler (
        const vector<char>& input_spikes, const vector<int8_t>& full_weight, 
        const vector<int>& bias, vector<int>& output_v_mem, vector<char>& output_spikes,
        Sconv_param p, bool ena_nz, bool ena_lb) 
    {
        // Tính toán sức chứa ic, oc_groups của SRAM và số oc mỗi lần tile
        for (int i = 0; i < Num_Cores; i++) core_mems[i].set_num_ic_oc_of_bank(p.C_in);
        int num_oc_tile = core_mems[0].get_num_oc_group_per_bank() * A_H;

        // =================================================================
        // 0. Tiling từng cụm oc_groups để load weight và tính toán Sconv
        // =================================================================
        for (int oc_tile_start = 0; oc_tile_start < p.C_out; oc_tile_start += num_oc_tile) {
            // Số oc_group cần nạp trong vòng lặp tính toán hiện tại
            int num_oc_groups = min(num_oc_tile / A_H, (p.C_out - oc_tile_start + A_H - 1) / A_H);

            // =================================================================
            // 1. Load weight từ DRAM lên SRAM từng core
            // =================================================================
            #pragma omp parallel for num_threads(Num_Cores)
            for (int core_id = 0; core_id < Num_Cores; core_id++) {
                int kh = core_id / p.K2; int kw = core_id % p.K2;

                // Tách full_weight theo từng pixel trong kernel về 2 chiều [C_out][C_in] và nạp vào SRAM
                vector<int8_t> weight_h_w(p.C_in * p.C_out, 0);
                for (int oc = 0; oc < p.C_out; oc++) {
                    for (int ic = 0; ic < p.C_in; ic++) {
                        weight_h_w[oc*p.C_in + ic] = full_weight[oc*p.C_in*p.K1*p.K2 + ic*p.K1*p.K2 + kh*p.K2 + kw];
                    }
                }
                core_mems[core_id].load_weight_DRAM_to_SRAM(oc_tile_start, p.C_in, p.C_out, weight_h_w, false);
            }

            // =================================================================
            // 2. Tính Sconv với từng pixel output trên cụm oc_groups đã nạp
            // =================================================================
            bool pp_spike = 0;
            bool pp_v_reg = 0;
            for (int h_out = 0; h_out < p.H_out; h_out++) {
                for (int w_out = 0; w_out < p.W_out; w_out++) {

                    // Xét từng oc_group đã nạp, tính theo fan-in: gom hết ic ứng với mỗi oc_group
                    for (int local_oc_group = 0; local_oc_group < num_oc_groups; local_oc_group++) {

                        // Số oc thật sự còn lại cần tính toán:
                        // Bằng A_H nếu C_out chia hết cho A_H hoặc chưa phải oc_group cuối cùng
                        // Nhỏ hơn A_H nếu C_out không chia hết cho A_H và ở oc_group cuối cùng
                        int num_oc_actual = min (A_H, p.C_out - oc_tile_start - local_oc_group * A_H);

                        for (int core_id = 0; core_id < Num_Cores; core_id++) core_array3d[core_id].reset_v_reg(pp_v_reg);

                        // =================================================================
                        // 2.1. Weight Accumulation (WA) bất đồng bộ 9 core với toàn bộ C_in
                        // =================================================================

                        // Cần tiling Spike_Mem nếu C_in > max_ic_per_spike_rf   
                        // global_ic_start = 0, 512, 1024, ... 
                        // Ở model SpikingFormer 768, C_in luôn <= 512 nên thực tế vòng lặp tiling này chỉ chạy 1 lần
                        for (int ic_tile_start = 0; ic_tile_start < p.C_in; ic_tile_start += max_ic_per_spike_rf) {
                            for (int core_id = 0; core_id < Num_Cores; core_id++) {

                                int kh = core_id / p.K2; int kw = core_id % p.K2;
                                int h_in = h_out * p.stride - p.padding + kh;
                                int w_in = w_out * p.stride - p.padding + kw;

                                // Nạp spike vào SpikeRF tại input position hợp lệ qua router
                                // Router sẽ so sánh input position, chốt mượn/load ngoài/giữ nguyên và set position mới
                                // Không được gọi hàm set position mới (h_in, w_in) ở đây
                                if (h_in >= 0 && h_in < p.H && w_in >= 0 && w_in < p.W) {
                                    mem_router.get_spikes_with_router(
                                        core_id, ic_tile_start, h_in, w_in, pp_spike,
                                        input_spikes, spike_rf, p.T, p.C_in, p.H, p.W);
                                } else {
                                    // Phải reset spike_rf và position (h_in, w_in), tránh store data cũ
                                    spike_rf[core_id].set_input_position(h_in, w_in, ic_tile_start);
                                    spike_rf[core_id].reset_spike_rf(pp_spike);
                                }
                                
                                // Chia SpikeRF ra từng cụm 64 ic, đẩy lên fetcher và tính WA
                                int num_ic_in_rf = min(max_ic_per_spike_rf, p.C_in - ic_tile_start);
                                // local_ic_start = 0, 64, 128, ..., 448 -> đánh dấu vị trí bắt đầu của từng cụm 64 ic
                                for (int local_ic_start = 0; local_ic_start < num_ic_in_rf; local_ic_start += 64) {

                                    // Số input channel thật sự được đưa vào fetcher
                                    int num_ic_to_scan = min(64, num_ic_in_rf - local_ic_start);
                                    
                                    // global_ic_start trong hàm này là ic thật sự trong IFM
                                    // không phải local_ic_start: giá trị offset trong từng lần tile
                                    // global_ic_start = 0, 64, 128, ..., 512, ...
                                    // nên phải = global_ic_start + local_ic_start
                                    unsigned long long global_bitmap = core_fetchers[core_id].generate_64_global_bitmap(
                                        spike_rf[core_id].get_local_spike(pp_spike),
                                        p.T, p.C_in, ic_tile_start + local_ic_start);
                                        
                                    // Không được check global_bitmap != 0 thì mới gọi hàm
                                    // Như vậy sẽ vô tình check độ thưa spike, sai với Mode 1: No NZ No LB
                                    // Với Mode 2, 3: cứ gọi hàm này kể cả khi chuỗi global_bitmap = 0
                                    // -> Đúng bản chất phần cứng: tách 8 chuỗi con, sinh địa chỉ và reset bit 1 ngoài cùng
                                    // -> Không có module thực hiện check trước 64 global bit này khác 0 hay không
                                    core_fetchers[core_id].fetch_data_and_compute(
                                        global_bitmap, local_oc_group, num_oc_actual,
                                        ic_tile_start + local_ic_start,
                                        p.C_in, core_mems[core_id], spike_rf[core_id].get_local_spike(pp_spike),
                                        NULL, p.T, core_array3d[core_id], pp_v_reg, ena_nz, ena_lb, false);
                                }
                            }
                            // Từng core đã tính xong WA cho một cụm ic_tile (nếu phải tile C_in) với 1 local_oc_group đang xét
                        }
                        // Từng core đã tính xong WA cho toàn bộ C_in với 1 local_oc_group đang xét

                        // =================================================================
                        // 2.2. Gom WA của các cores và tính MU ở V_reg chung
                        // =================================================================

                        // Không đếm cycle_count trong phép MU do đã bị giấu trong WA ở vị trí tiếp theo
                        main_V_reg.reset(pp_v_reg); // reset main_V_reg trước khi gom P_sum và MU

                        for (int oh = 0; oh < A_H; oh++) {
                            int actual_oc = oc_tile_start + local_oc_group * A_H + oh;   // oc thật sự trong OFM
                            if (actual_oc >= p.C_out) continue;
                            int v_mem = 0;
                            
                            for (int t = 0; t < p.T; t++) {
                                // Gom kết quả P_sum từ 9 Cores
                                for (int core_id = 0; core_id < 9; core_id++)
                                    main_V_reg.RF[pp_v_reg][t][oh] += core_array3d[core_id].get_v_reg(t, oh, pp_v_reg);

                                // Cập nhật membrane potential, nhân leaky = 0.5 tương đương >> 1
                                v_mem = (v_mem >> 1) + main_V_reg.RF[pp_v_reg][t][oh] + bias[actual_oc];
                                
                                // Layput output: [C_out][H_out][W_out][T]
                                int out_idx = actual_oc * p.H_out * p.W_out * p.T + h_out * p.W_out * p.T + w_out * p.T + t;
                                output_v_mem[out_idx] = v_mem;

                                // Phát xung, lưu vào output_spikes và soft reset
                                if (v_mem >= p.V_th) {
                                    output_spikes[out_idx] = 1;
                                    v_mem -= p.V_th; 
                                } else output_spikes[out_idx] = 0;
                            }
                        } // Xong MU

                        pp_v_reg = !pp_v_reg; // Đảo ping-pong V_reg ở các cores để tính WA cho local_oc_group tiếp theo
                    }
                    // Xong các local_oc_group ở cụm oc tile hiện tại

                    // Đảo ping-pong SpikeRF ở các cores để tính Sconv cho vị trí (h,w) tiếp theo trong OFM
                    // Memory router sẽ phát huy tác dụng khi sang vị trí tiếp theo, cửa sổ conv đè lên nhau
                    pp_spike = !pp_spike;
                }
            }
            // Xong toàn bộ các vị trí (h,w) trong output ở 1 cụm oc_groups tiling
        }
        // Xong toàn bộ OFM (tiling hết các output channels)
    }

    void count_metrics(){
        for (int i = 0; i < Num_Cores; i++) {
            metrics.SOP_count += core_array3d[i].get_SOP_count();
            metrics.spike_load_DRAM_to_SRAM_count += spike_rf[i].get_spike_load_DRAM_to_SRAM_count();
            metrics.weight_load_DRAM_to_SRAM_count += core_mems[i].get_weight_load_DRAM_to_SRAM_count();
            metrics.weight_load_SRAM_to_array_count += core_fetchers[i].get_weight_load_SRAM_to_array_count();
            // cycle_count là số cycle lớn nhất trong các cores chạy song song bất đồng bộ
            metrics.cycle_count = max(metrics.cycle_count, core_fetchers[i].get_cycle_count());
            metrics.average_cycle_all_core += core_fetchers[i].get_cycle_count();
        }
        metrics.average_cycle_all_core /= Num_Cores;
        metrics.count_average_num_WeightVector_load_per_cycle();
        metrics.router_shared_count = mem_router.get_router_shared_count();
    }

    void print_metrics(){
        cout << "Total SOP              : " << metrics.SOP_count << "\n";
        cout << "Total cycle (max of cores)     : " << metrics.cycle_count << "\n";
        cout << "Total cycle (avg of cores)     : " << metrics.average_cycle_all_core << "\n";
        cout << "Total load weight DRAM to SRAM : " << metrics.weight_load_DRAM_to_SRAM_count << "\n";
        cout << "Total load weight SRAM to array: " << metrics.weight_load_SRAM_to_array_count << "\n";
        cout << "Total load spike DRAM to SRAM w/o router: " << metrics.spike_load_DRAM_to_SRAM_count + metrics.router_shared_count << "\n";
        cout << "Total load spike DRAM to SRAM w router  : " << metrics.spike_load_DRAM_to_SRAM_count << "\n";
        cout << "Average number of WeightVector load to array per cycle: " << metrics.average_num_WeightVector_load_per_cycle << "\n";
    }

    StatisticalData get_metrics(){return metrics;}

    void reset_all_data(){
        for (int i = 0; i < Num_Cores; i++){
            spike_rf[i].reset_all_data();
            core_mems[i].reset_all_data();
            core_array3d[i].reset_all_data();
            core_fetchers[i].reset_all_data();
            mem_router.reset_all_data();
            main_V_reg.reset(0); main_V_reg.reset(1);
            metrics.reset_data();
        }
    }
};