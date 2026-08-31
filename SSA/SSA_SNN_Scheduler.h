#include "SNN_Accelerator.h"
#include "SSA_layer_param.h"

class SNN_Hardware_Accelerator {
private:
    SpikeRF spike_rf[Num_Cores];
    MultiBankMemory core_mems[Num_Cores];
    ComputationArray3D core_array3d[Num_Cores];
    NonZeroDataFetcher core_fetchers[Num_Cores];
    AsyncCoreMemRouter mem_router;
    V_reg main_V_reg;
    StatisticalData metrics;

public:
    // Hàm tính toán nhân 1 hàng thứ j của S = K^T * V (lấy hàng thứ j của K^T nhân với V)
    // Chỉ tính trên 1 d_group vì một lần V_reg chỉ tính được như vậy
    void Core_Compute_token_j_S_Matrix (
        int core_id, int j, bool& pp_spike, // pp_spike phải truyền tham chiếu để sửa trực tiếp
        const vector<char>& K_local_bitmap, char Q_local_bitmap[A_D],
        SSA_param& p, int local_d_group, int num_oc_actual,
        bool pp_v_reg, bool ena_nz, bool ena_lb)
    {
        // Không được reset V_reg của core vì cần cộng dồn đủ các hàng thứ j của S
        // ứng với các vị trí global bitmap Q[i][j] = 1

        // Cần nạp một hàng j của K^T (vai trò input spikes) vào SpikeRF
        // K^T có layout [d_head][N_tokens][T] với N_tokens tương tự C_in trong conv
        // Nếu max_ic_per_spike_rf < N_tokens, ta chia tile để lưu vào ping-pong Spike_rf
        // Với model SpikingFormer 8-768 thì N_tokens = 196 < 512 nên thực tế không cần tile
        for (int n_tile_start = 0; n_tile_start < p.N_tokens; n_tile_start += max_ic_per_spike_rf) {

            // Tiling N_tokens và nạp K_spikes vào SpikeRF pp
            spike_rf[core_id].set_input_position(-1, -1, n_tile_start);
            spike_rf[core_id].load_spike_DRAM_to_SRAM_SSA(
                K_local_bitmap, j, p.N_tokens, p.T, pp_spike);

            // Quét qua từng cụm 64 tokens (tương đương 64 ic) của hàng K^T[j]
            int num_of_tokens_actual = min(p.N_tokens - n_tile_start, max_ic_per_spike_rf);
            for (int local_n_start = 0; local_n_start < num_of_tokens_actual; local_n_start += 64) {

                int local_ic_start = n_tile_start + local_n_start;
                // Tạo 64 global bitmap từ K^T
                unsigned long long global_bitmap = core_fetchers[core_id].generate_64_global_bitmap(
                    spike_rf[core_id].get_local_spike(pp_spike), p.T, p.N_tokens, local_ic_start);
                
                // Fetch weight từ Multibank lên 3D Array và tính toán
                // Hàm điều phối sẽ load V vào Multibank và K bitmap vào SpikeRF trước khi tính
                core_fetchers[core_id].fetch_data_and_compute(
                    global_bitmap, local_d_group, num_oc_actual, local_ic_start,
                    p.N_tokens, core_mems[core_id],
                    spike_rf[core_id].get_local_spike(pp_spike), Q_local_bitmap,
                    p.T, core_array3d[core_id], pp_v_reg, ena_nz, ena_lb, true);
            }

            pp_spike = !pp_spike;
        }
    }

    // Khối SAS: nhận hàng (token) thứ i của Q, sinh global bitmap và fetch hàng thứ j của K^T rồi tính
    void Core_SAS_and_compute (
        const vector<char>& Q_spikes, const vector<char>& K_spikes, int core_id, int i,
        int T, int N_tokens, int local_d_group, int d, int num_oc_actual,
        // pp_spike phải truyền tham chiếu để sửa trực tiếp trong hàm "Core_Compute_token_j_S_Matrix"
        bool& pp_spike, bool pp_v_reg, bool ena_SAS, bool ena_nz, bool ena_lb)
    {
        // Quét hàng Q[i] theo các cột j (chạy đủ d cột của hàng Q[i])
        for (int j = 0; j < d; j++) {

            // Quét Q[i][j] theo các timesteps
            char Q_ij_local_bitmap[A_D] = {0};
            for (int t = 0; t < T; t++) {
                Q_ij_local_bitmap[t] = Q_spikes[i * d * T + j * T + t];
            }

            bool Q_ij_global_bitmap = false; // global bitmap của Q[i][j]
            for (int t = 0; t < T; t++) {
                if (Q_ij_local_bitmap[t]) {Q_ij_global_bitmap = true; break;}
            }

            // SAS: Nếu Q[i][j] global bitmap = 1 thì tính hàng j của S, ngược lại skip
            if (Q_ij_global_bitmap) {
                Core_Compute_token_j_S_Matrix(
                    core_id, j, pp_spike, K_spikes, Q_ij_local_bitmap, p,
                    local_d_group, num_oc_actual, pp_v_reg, ena_nz, ena_lb);
                metrics.SSA_operation_by_SAS_count++; // // Số lần tính K^T[j] * V nếu có SAS
            } else {
                if (!ena_SAS) { // Nếu không bật SAS thì vẫn load Q, K tương ứng vào để tính
                    Core_Compute_token_j_S_Matrix(
                        core_id, j, pp_spike, K_spikes, Q_ij_local_bitmap, p,
                        local_d_group, num_oc_actual, pp_v_reg, ena_nz, ena_lb);
                }
                metrics.SSA_skip_by_SAS_count++; // Số lần skip K^T[j] * V nếu có SAS
            }
        }
    }

    // Hàm Orchestrator tính SSA hoàn chỉnh
    void SSA_Sim(
        const vector<vector<char>>& Q_spikes, const vector<vector<char>>& K_spikes,
        const vector<vector<char>>& V_spikes,
        vector<char>& SSA_output, SSA_param& p, bool ena_SAS, bool ena_nz, bool ena_lb)
    {
        for (int core_id = 0; core_id < Num_Cores; core_id++)
            core_mems[core_id].set_num_ic_oc_of_bank(p.N_tokens);
        int num_oc_per_tile = core_mems[0].get_num_oc_group_per_bank() * A_H;

        // Tính SSA trên từng head rồi hợp lại thành chiều C_out
        for (int head_idx = 0; head_idx < p.num_heads; head_idx++) {
            // Do d của các heads trong model lớn nên phải tiling (d tương tự oc)
            for (int oc_tile_start = 0; oc_tile_start < p.d; oc_tile_start += num_oc_per_tile) {

                // Số oc thật sự còn lại cần tile để nạp và tính trong head này
                int num_oc_tile_actual = min (num_oc_per_tile, p.d - oc_tile_start);

                // BƯỚC 1: Nạp V làm weight vào Multibank SRAM
                // Cấp phát vector chứa weight là ma trận V được đóng gói theo 8 timesteps về layout [N][d]
                vector<int8_t> packed_v (p.N_tokens * num_oc_tile_actual, 0);
                for (int n = 0; n < p.N_tokens; n++) {
                    for (int oc = 0; oc < num_oc_tile_actual; oc++) {
                        int actual_oc_in_this_head = oc_tile_start + oc;
                        // Đóng gói spikes của 8 timesteps tại vị trí V[n][oc] thành 1 số int8
                        // Dịch trái t lần spike V[n][oc][t] rồi OR với packed weight
                        // Như vậy spike của các timesteps được lưu lần lượt từ phải -> trái
                        for (int t = 0; t < p.T; t++)
                            packed_v[n*num_oc_tile_actual + oc]
                            |= V_spikes[head_idx][n*p.d*p.T + actual_oc_in_this_head*p.T + t] << t;
                    }
                }

                for (int core_id = 0; core_id < Num_Cores; core_id++) 
                    core_mems[core_id].load_weight_DRAM_to_SRAM(oc_tile_start, p.N_tokens, p.d, packed_v, true);

                // BƯỚC 2: Tính Q * K^T * V, sử dụng các cores song song cho các hàng (tokens) liền nhau của Q
                for (int core_id = 0; core_id < Num_Cores; core_id++) {
                    bool pp_v_reg = 0, pp_spike = 0;

                    for (int n_tokens = core_id; n_tokens < p.N_tokens; n_tokens += Num_Cores) {

                        // tính SSA cho từng cụm d_group (tương tự oc_group trong Sconv) đã nạp vào SRAM
                        // local_d_group = 0 ứng với A_H cột đầu tiên trong num_oc_tile_actual cột của V ...
                        for (int local_d_group = 0; A_H * local_d_group < num_oc_tile_actual; local_d_group++) {

                            int num_oc_actual_this_group = min (A_H, num_oc_tile_actual - A_H * local_d_group);

                            core_array3d[core_id].reset_v_reg(pp_v_reg);
                            Core_SAS_and_compute(Q_spikes[head_idx], K_spikes[head_idx],
                                core_id, n_tokens, p.T, p.N_tokens, local_d_group, p.d,
                                num_oc_actual_this_group, pp_spike, pp_v_reg, ena_SAS, ena_nz, ena_lb);
                            
                            // BƯỚC 3: Nhân scale và tính MU
                            for (int oc = 0; oc < num_oc_actual_this_group; oc++) {
                                int v_mem = 0;
                                for (int t = 0; t < p.T; t++) {

                                    int oc_output_final = head_idx * p.d + oc_tile_start + local_d_group * A_H + oc;
                                    int output_final_idx = oc_output_final * p.N_tokens * p.T + n_tokens * p.T + t;

                                    // leaky = 0.5, scale = 0.125
                                    v_mem = (v_mem >> 1) + core_array3d[core_id].get_v_reg(t, oc, pp_v_reg);
                                    if (v_mem >= p.V_th) {SSA_output[output_final_idx] = 1; v_mem -= p.V_th;}
                                    else SSA_output[output_final_idx] = 0;
                                }
                            }

                            pp_v_reg = !pp_v_reg; // đảo sau 1 local_d_group (tương tự 1 oc_group trong SConv)
                        }
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
        cout << "Total SSA operation w/o SAS module: " << metrics.SSA_operation_by_SAS_count << "\n";
        cout << "Total SSA operation w SAS module:   " << metrics.SSA_operation_by_SAS_count + metrics.SSA_skip_by_SAS_count << "\n";
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