#include "SNN_Accelerator.h"
#include "S_QKV_layer_param.h"

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
    // HÀM TÍNH TOÁN TRỰC TIẾP Q/K/V VỚI 1 TOKEN TRONG 1 CORE
    void Core_Compute_QKV(
        int core_id, int n_tokens, int local_oc_group, int num_oc_actual, int num_ic_actual,
        int ic_tile_start, const vector<char>& input_spikes, SQKV_param& p,
        bool pp_v_reg, bool pp_spike, bool ena_nz, bool ena_lb)
    {        
        // Chia cụm 64 global bitmap rồi tính
        for (int local_ic_start = 0; local_ic_start < num_ic_actual; local_ic_start += 64) {

            // global_ic_start = 0, 64, 128, ... chạy trên toàn N_in
            // nên phải = ic_tile_start + local_ic_start
            unsigned long long global_bitmap = core_fetchers[core_id].generate_64_global_bitmap(
                spike_rf[core_id].get_local_spike(pp_spike), p.T, p.C_in, ic_tile_start + local_ic_start);
            
            // Weight phải được nạp từ trước đó thông qua khối điều phối (Orchestrator)
            core_fetchers[core_id].fetch_data_and_compute(
                global_bitmap, local_oc_group, num_oc_actual, ic_tile_start + local_ic_start,
                p.C_in, core_mems[core_id],
                spike_rf[core_id].get_local_spike(pp_spike), NULL, p.T,
                core_array3d[core_id], pp_v_reg, ena_nz, ena_lb, false
            );
        }
    }

    void SQKV_Scheduler(
        const vector<char>& input_spikes, const vector<int>& bias,
        const vector<int8_t>& w_q, const vector<int8_t>& w_k, const vector<int8_t>& w_v,
        // Vector chứa num_heads output_qkv, mỗi output lại là vector<char> chứa spike
        vector<vector<char>>& output_q, vector<vector<char>>& output_k, vector<vector<char>>& output_v,
        SQKV_param& p, bool ena_nz, bool ena_lb)
    {
        // Tính toán sức chứa ic, oc_groups của SRAM và số oc mỗi lần tile
        // Chú ý là tính theo fan-in, luôn load đủ weight theo C_in lên SRAM trước rồi tile C_out
        for (int i = 0; i < Num_Cores; i++) core_mems[i].set_num_ic_oc_of_bank(p.C_in);
        // Số lượng oc mỗi lần tile để nạp Weight lên SRAM
        int num_oc_per_tile = core_mems[0].get_num_oc_group_per_bank() * A_H;

        // Tiling từng cụm oc_groups để load weight và tính toán QKV_gen trên toàn bộ C_in
        for (int oc_tile_start = 0; oc_tile_start < p.C_out; oc_tile_start += num_oc_per_tile) {
            // Số oc_group thật sự trong lần tile này
            int num_oc_group_actual = min(num_oc_per_tile / A_H, (p.C_out - oc_tile_start + A_H - 1) / A_H);

            // BƯỚC 1: Nạp trọng số theo mô hình chia nhóm cho Q, K, V
            // Cluster 0: Core 0(Q), 1(K), 2(V) | Cluster 1: Core 3(Q), 4(K), 5(V) | ...
            for (int core_id = 0; core_id < Num_Cores; core_id++) {
                int role = core_id % 3; // 0=Q, 1=K, 2=V
                
                const vector<int8_t>& current_weight = (role == 0) ? w_q : ((role == 1) ? w_k : w_v);
                core_mems[core_id].load_weight_DRAM_to_SRAM(oc_tile_start, p.C_in, p.C_out, current_weight, false);
            }

            // BƯỚC 2: Điều phối tính toán độc lập cho từng Token
            #pragma omp parallel for num_threads(Num_Cores)
            for (int core_id = 0; core_id < Num_Cores; core_id++) {
                int role = core_id % 3;         // vai trò Q/K/V
                int cluster_id = core_id / 3;

                // Cứ 3 tokens liền nhau thì chia đều cho 3 cluster
                // Cluster 0 (Core 0, 1, 2) tính Q, K, V cho token 0
                // Cluster 1 (Core 3, 4, 5) tính Q, K, V cho token 1
                // ... Đến lần tiếp theo Cluster 0 tính Q, K, V cho token 3
                bool pp_v_reg = 0;
                for (int n = cluster_id; n < p.N_tokens; n += 3) {

                    // BƯỚC 2.1. Nạp input spikes sau khi xác định role và vị trí n_tokens cụ thể
                    // Tiling C_in nếu cần, tận dụng 2 Spike_RF pp để lưu từng cụm 512 ic
                    /*  Với model SpikingFormer 8-768 thì C_in của QKV là 768, lưu đủ vào 2 thanh pp chứa max 1024 ic
                        Tổng quát nếu C_in > 1024 thì code này chạy sai, ta phải xét từng oc_groups và nạp đủ C_in lên SpikeRF
                        Tuy nhiên với C_in < 1024 thì code như vậy sẽ phải nạp lại input spikes như nhau lên từng cụm oc_group
                        -> Lặp lại nạp spike không cần thiết, vì thế trong phạm vi kích thước hiện tại ta code như này */

                    bool pp_spike = 0;  // luôn bắt đầu ghi từ pp = 0 khi bắt đầu vị trí n_tokens mới
                    for (int ic_tile_start = 0; ic_tile_start < p.C_in; ic_tile_start += max_ic_per_spike_rf) {
                        // phải set ic_tile_start của SpikeRF để hàm bên dưới điều phối nạp
                        spike_rf[core_id].set_input_position(-1, -1, ic_tile_start);
                        // Nạp dữ liệu của 1 TOKEN vào SpikeRF
                        spike_rf[core_id].load_spike_DRAM_to_SRAM_QKV(
                            input_spikes, n, p.N_tokens, p.C_in, p.T, pp_spike
                        );
                        pp_spike = !pp_spike;
                    }
                    
                    // BƯỚC 2.2. Sau khi nạp đủ spike của tất cả ic lên thì chia local_oc_groups
                    for (int local_oc_group = 0; local_oc_group < num_oc_group_actual; local_oc_group++) {
                        pp_spike = 0;   // bắt đầu lại từ 0 do kết thúc vòng lặp trước có thể pp_spike = 1

                        // Số oc thật sự còn lại trong oc_group đang tính
                        // Có thể < A_H nếu nằm ở oc_group cuối cùng khi C_out không chia hết cho A_H
                        int num_oc_actual = min(A_H, p.C_out - oc_tile_start - local_oc_group*A_H);
                        core_array3d[core_id].reset_v_reg(pp_v_reg);

                        // Đọc đủ spike của tất cả C_in đã nạp rồi tính
                        for (int ic_tile_start = 0; ic_tile_start < p.C_in; ic_tile_start += max_ic_per_spike_rf) {
                            int num_ic_actual = min(max_ic_per_spike_rf, p.C_in - ic_tile_start);
                            Core_Compute_QKV(
                                core_id, n, local_oc_group, num_oc_actual, num_ic_actual,
                                ic_tile_start, input_spikes, p, pp_v_reg, pp_spike, ena_nz, ena_lb
                            );
                            pp_spike = !pp_spike;   // Đảo spikeRF nếu cần để đọc nốt
                        }

                        // MU và nén output theo 2 chiều (Bidirectional Compressor)
                        for (int oh = 0; oh < num_oc_actual; oh++) {

                            // oc thật sự trong C_out của output, chưa chia ra các heads
                            int oc = oc_tile_start + local_oc_group * A_H + oh;
                            int head_idx = oc / p.d_heads;    // chỉ số head của oc hiện tại

                            // Tìm xem output hiện tại là Q/K/V và ở head nào để lưu đúng
                            vector<char>* current_output;
                            if (role == 0) current_output = &output_q[head_idx];
                            else if (role == 1) current_output = &output_k[head_idx];
                            else current_output = &output_v[head_idx];

                            int V_mem = 0; // Khởi tạo điện thế màng cho mỗi output channel mới
                            char spike_vector[A_D] = {0}; // Mảng tạm lưu các spikes của 1 output channel

                            for (int t = 0; t < p.T; t++) {
                                int psum = core_array3d[core_id].get_v_reg(t, oh, pp_v_reg);
                                
                                // leaky factor = 0.5 -> V_mem * 0,5 = V_mem >> 1
                                V_mem = (V_mem >> 1) + psum + bias[oc];
                                if (V_mem >= p.V_th) {
                                    spike_vector[t] = 1;
                                    V_mem -= p.V_th; // Soft reset
                                } else spike_vector[t] = 0;

                                int out_idx;
                                if (role == 1) {         // K: Nén Dọc [d_heads][N][T]
                                    out_idx = (oc % p.d_heads) * p.N_tokens * p.T + n * p.T + t;
                                } else {                 // Q, V: Nén Ngang [N][d_heads][T]
                                    out_idx = n * p.d_heads * p.T + (oc % p.d_heads) * p.T + t;
                                }
                                (*current_output)[out_idx] = spike_vector[t];
                            }
                        }

                        // Xong 1 token tại 1 local_oc_group thì đảo pp_v_reg ở core
                        pp_v_reg = !pp_v_reg;
                    }
                }
            }
        }
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
        cout << "Total load spike DRAM to SRAM: " << metrics.spike_load_DRAM_to_SRAM_count << "\n";
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