#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "SSA_SNN_Scheduler.h"

// Hàm ghi metrics ra file CSV
void log_metrics_to_csv(std::ofstream& csv_file, int sparsity, const std::string& mode, const StatisticalData& metrics) {
    if (!csv_file.is_open()) return;
    csv_file << sparsity << "," 
             << mode << ","
             << metrics.SOP_count << "," 
             << metrics.cycle_count << ","
             << metrics.average_cycle_all_core << ","
             << metrics.weight_load_DRAM_to_SRAM_count << ","
             << metrics.weight_load_SRAM_to_array_count << ","
             << metrics.spike_load_DRAM_to_SRAM_count << ","
             << metrics.SSA_operation_by_SAS_count + metrics.SSA_skip_by_SAS_count << ","
             << metrics.SSA_operation_by_SAS_count << ","
             << metrics.average_num_WeightVector_load_per_cycle << "\n";
}

int main() {
    vector<int> input_sparsity = {30, 50, 70, 90};
    string data_dir = "SSA_input_data";
    string out_dir = "SSA_SNN_output";
    filesystem::create_directories(out_dir);

    vector<vector<char>> Q_spikes(p.num_heads);
    vector<vector<char>> K_spikes(p.num_heads);
    vector<vector<char>> V_spikes(p.num_heads);
    vector<char> SSA_output(p.C_out * p.H * p.W * p.T, 0);

    // Cấp phát động trên Heap
    SNN_Hardware_Accelerator* hw = new SNN_Hardware_Accelerator();
    
    ofstream csv_file("metrics_ssa.csv");
    csv_file << "Sparsity(%),Mode,SOP,Max_Cycle,Avg_Cycle,Weight_DRAM_loads,Weight_SRAM_loads,Spike_DRAM_load,Total_SSA_wo_SAS,Total_SSA_w_SAS,Avg_num_Weight_load_per_cycle\n";

    for (int sparsity : input_sparsity) {
        cout << "\n========================================================\n";
        cout << "Running SSA with Global Sparsity: " << sparsity << "%\n";
        cout << "========================================================\n";

        string cur_data_dir = data_dir + "/Sparsity_" + to_string(sparsity) + "%";
        string cur_out_dir = out_dir + "/Sparsity_" + to_string(sparsity) + "%";
        filesystem::create_directories(cur_out_dir);

        // Nạp data Q, K, V cho các heads
        for (int h = 0; h < p.num_heads; h++) {
            Q_spikes[h] = load_txt_spike(cur_data_dir + "/Input_Q_head_" + to_string(h) + ".txt", p.N_tokens * p.d * p.T);
            K_spikes[h] = load_txt_spike(cur_data_dir + "/Input_K_head_" + to_string(h) + ".txt", p.N_tokens * p.d * p.T);
            V_spikes[h] = load_txt_spike(cur_data_dir + "/Input_V_head_" + to_string(h) + ".txt", p.N_tokens * p.d * p.T);
        }

        // MODE 1: No SAS No NZ (Tuần tự cơ bản)
        cout << "[MODE 1] No SAS No NZ:\n";
        hw->reset_all_data();
        hw->SSA_Sim(Q_spikes, K_spikes, V_spikes, SSA_output, p, false, false, false);
        hw->count_metrics();
        log_metrics_to_csv(csv_file, sparsity, "No SAS No NZ", hw->get_metrics());
        save_txt_SSA_LIF_reshape_H_W(cur_out_dir + "/SSA_SNN_output_Mode_1.txt", SSA_output, p.C_out, p.H, p.W, p.T);

        // MODE 2: SAS Only
        cout << "[MODE 2] SAS Only:\n";
        hw->reset_all_data();
        hw->SSA_Sim(Q_spikes, K_spikes, V_spikes, SSA_output, p, true, false, false);
        hw->count_metrics();
        log_metrics_to_csv(csv_file, sparsity, "SAS Only", hw->get_metrics());
        save_txt_SSA_LIF_reshape_H_W(cur_out_dir + "/SSA_SNN_output_Mode_2.txt", SSA_output, p.C_out, p.H, p.W, p.T);

        // MODE 3: SAS + NZ
        cout << "[MODE 3] SAS + NZ:\n";
        hw->reset_all_data();
        hw->SSA_Sim(Q_spikes, K_spikes, V_spikes, SSA_output, p, true, true, false);
        hw->count_metrics();
        log_metrics_to_csv(csv_file, sparsity, "SAS + NZ", hw->get_metrics());
        save_txt_SSA_LIF_reshape_H_W(cur_out_dir + "/SSA_SNN_output_Mode_3.txt", SSA_output, p.C_out, p.H, p.W, p.T);

        // MODE 4: SAS + NZ + LB (Bật toàn bộ khối tăng tốc)
        cout << "[MODE 4] SAS + NZ + LB:\n";
        hw->reset_all_data();
        hw->SSA_Sim(Q_spikes, K_spikes, V_spikes, SSA_output, p, true, true, true);
        hw->count_metrics();
        log_metrics_to_csv(csv_file, sparsity, "SAS + NZ + LB", hw->get_metrics());
        save_txt_SSA_LIF_reshape_H_W(cur_out_dir + "/SSA_SNN_output_Mode_4.txt", SSA_output, p.C_out, p.H, p.W, p.T);
    }
    
    delete hw; 
    csv_file.close();
    cout << "\nAll output and metrics saved.\n";
    return 0;
}