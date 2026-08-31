#include <filesystem>   // Tạo thư mục lưu output
#include <fstream>
#include <string>
#include "S_QKV_SNN_Scheduler.h"

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
             << metrics.average_num_WeightVector_load_per_cycle << "\n";
}

int main() {
    vector<int> input_sparsity = {30, 50, 70, 90};
    string data_dir = "S_QKV_input_and_weight_data";
    string out_dir = "S_QKV_SNN_output_sparse";
    filesystem::create_directories(out_dir);

    vector<string> out_dir_sparse;
    for (int sparsity : input_sparsity) {
        string cur_dir = out_dir + "/Sparsity_" + to_string(sparsity) + "%";
        out_dir_sparse.push_back(cur_dir);
        filesystem::create_directories(cur_dir);
    }

    // Đọc weight chung cho tất cả sparsity
    vector<int8_t> weight_Q = load_txt_weight(data_dir + "/weight_q.txt", p.weight_size);
    vector<int8_t> weight_K = load_txt_weight(data_dir + "/weight_k.txt", p.weight_size);
    vector<int8_t> weight_V = load_txt_weight(data_dir + "/weight_v.txt", p.weight_size);
    vector<int> bias(p.bias_size, 0);
    // Khai báo trước các vector, ghi đè khi chạy từng sparsity
    vector<char> input_spikes(p.input_size, 0);
    vector<vector<char>> output_Q (p.num_heads, vector<char>(p.output_size_QKV_each_head, 0));
    vector<vector<char>> output_K (p.num_heads, vector<char>(p.output_size_QKV_each_head, 0));
    vector<vector<char>> output_V (p.num_heads, vector<char>(p.output_size_QKV_each_head, 0));

    // Cấp phát động trên Heap, tránh tràn stack do kích thước hw khá lớn
    SNN_Hardware_Scheduler* hw = new SNN_Hardware_Scheduler();
    
    // Tạo file CSV lưu metric của các sparsity và ghi tiêu đề cột
    ofstream csv_file("metrics_sqkv.csv");
    csv_file << "Sparsity(%),Mode,SOP,Max_Cycle,Avg_Cycle,Weight_DRAM_loads,Weight_SRAM_loads,Spike_DRAM_load, Avg_num_Weight_load_per_cycle\n";

    int i = 0;
    for (int sparsity : input_sparsity) {
        cout << "\nRunning S_QKV_Gen with Global Sparsity: " << sparsity << "%\n";
        // Nạp spikes với độ thưa tương ứng
        string input_dir = data_dir + "/input_spikes_sparsity_" + to_string(sparsity) + "%" + ".txt";
        input_spikes = load_txt_spike(input_dir, p.input_size);

        // Chạy Mode 1: No NZ
        cout << "[MODE 1] No NZ:\n";
        hw->reset_all_data();
        hw->SQKV_Scheduler(
            input_spikes, bias, weight_Q, weight_K, weight_V, 
            output_Q, output_K, output_V, p, false, false);
        hw->count_metrics();
        log_metrics_to_csv(csv_file, sparsity, "No NZ", hw->get_metrics());

        // Chạy Mode 2: NZ Only
        cout << "[MODE 2] NZ Only:\n";
        hw->reset_all_data();
        hw->SQKV_Scheduler(
            input_spikes, bias, weight_Q, weight_K, weight_V, 
            output_Q, output_K, output_V, p, true, false);
        hw->count_metrics();
        log_metrics_to_csv(csv_file, sparsity, "NZ Only", hw->get_metrics());

        // Chạy Mode 3: NZ + LB
        cout << "[MODE 3] NZ + LB:\n";
        hw->reset_all_data();
        hw->SQKV_Scheduler(
            input_spikes, bias, weight_Q, weight_K, weight_V, 
            output_Q, output_K, output_V, p, true, true);
        hw->count_metrics();
        log_metrics_to_csv(csv_file, sparsity, "NZ + LB", hw->get_metrics());
        
        // Lưu output SQKV của Mode 3 với từng head
        // Có thể so sánh 3 mode cho ra output khớp nhau bằng hàm main chạy riêng 1 file input
        for (int head_idx = 0; head_idx < p.num_heads; head_idx++) {
            string q_filename = "/SNN_output_Q_head_" + to_string(head_idx) + ".txt";
            string k_filename = "/SNN_output_K_head_" + to_string(head_idx) + ".txt";
            string v_filename = "/SNN_output_V_head_" + to_string(head_idx) + ".txt";

            save_txt_QV_LIF(out_dir_sparse[i] + q_filename, output_Q[head_idx], p.N_tokens, p.d_heads, p.T);
            save_txt_K_Transpose_LIF(out_dir_sparse[i] + k_filename, output_K[head_idx], p.N_tokens, p.d_heads, p.T);
            save_txt_QV_LIF(out_dir_sparse[i] + v_filename, output_V[head_idx], p.N_tokens, p.d_heads, p.T);
        }
        i++;
    }
    
    delete hw; csv_file.close();
    cout << "All output and metrics saved.\n";
    return 0;
}