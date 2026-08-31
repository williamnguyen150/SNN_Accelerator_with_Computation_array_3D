#include <filesystem>   // Tạo thư mục lưu output
#include <fstream>
#include <string>
#include "Sconv_SNN_Scheduler.h"

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
             << metrics.spike_load_DRAM_to_SRAM_count + metrics.router_shared_count << ","
             << metrics.spike_load_DRAM_to_SRAM_count << ","
             << metrics.average_num_WeightVector_load_per_cycle << "\n";
}

int main() {
    vector<int> input_sparsity = {30, 50, 70, 90};
    string data_dir = "Sconv_input_and_weight_data";
    string out_dir = "Sconv_SNN_output_sparse";
    filesystem::create_directories(out_dir);

    // Đọc weight chung cho tất cả sparsity
    vector<int8_t> weight = load_txt_weight(data_dir + "/weight.txt", p.weight_size);
    vector<int> bias(p.bias_size, 0);
    // Khai báo trước các vector, ghi đè khi chạy từng sparsity
    vector<char> input_spikes(p.input_size, 0);
    vector<int> output_v_mem(p.output_size, 0);
    vector<char> output_spikes(p.output_size, 0);

    // Cấp phát động trên Heap, tránh tràn stack do kích thước hw khá lớn
    SNN_Hardware_Scheduler* hw = new SNN_Hardware_Scheduler();
    
    // Tạo file CSV lưu metric của các sparsity và ghi tiêu đề cột
    ofstream csv_file("metrics_sconv.csv");
    csv_file << "Sparsity(%),Mode,SOP,Max_Cycle,Avg_Cycle,Weight_DRAM_loads,Weight_SRAM_loads,Spike_DRAM_load_wo_Router,Spike_DRAM_load_w_Router,Avg_num_Weight_load_per_cycle\n";

    for (int sparsity : input_sparsity) {
        cout << "\nRunning Sconv with Global Sparsity: " << sparsity << "%\n";

        // Nạp spikes với độ thưa tương ứng
        string input_dir = data_dir + "/input_spikes_sparsity_" + to_string(sparsity) + "%" + ".txt";
        input_spikes = load_txt_spike(input_dir, p.input_size);

        // Chạy Mode 1: No NZ
        cout << "[MODE 1] No NZ:\n";
        hw->reset_all_data();
        hw->SConv_Scheduler(input_spikes, weight, bias, output_v_mem, output_spikes, p, false, false);
        hw->count_metrics();
        log_metrics_to_csv(csv_file, sparsity, "No NZ", hw->get_metrics());

        // Chạy Mode 2: NZ Only
        cout << "[MODE 2] NZ Only:\n";
        hw->reset_all_data();
        hw->SConv_Scheduler(input_spikes, weight, bias, output_v_mem, output_spikes, p, true, false);
        hw->count_metrics();
        log_metrics_to_csv(csv_file, sparsity, "NZ Only", hw->get_metrics());

        // Chạy Mode 3: NZ + LB
        cout << "[MODE 3] NZ + LB:\n";
        hw->reset_all_data();
        hw->SConv_Scheduler(input_spikes, weight, bias, output_v_mem, output_spikes, p, true, true);
        hw->count_metrics();
        log_metrics_to_csv(csv_file, sparsity, "NZ + LB", hw->get_metrics());
        
        // Lưu output Sconv của Mode 3
        // Có thể so sánh 3 mode cho ra output khớp nhau bằng hàm main chạy riêng 1 file input
        string output_1_dir = out_dir + "/Sconv_SNN_output_1_" + to_string(sparsity) + "%" + ".txt";
        save_txt_Sconv_v_mem(output_1_dir, output_v_mem, p.C_out, p.H_out, p.W_out, p.T);
        
        string output_2_dir = out_dir + "/Sconv_SNN_output_2_" + to_string(sparsity) + "%" + ".txt";
        save_txt_Sconv_LIF(output_2_dir, output_spikes, p.C_out, p.H_out, p.W_out, p.T);
    }
    
    delete hw; csv_file.close();
    cout << "All output and metrics saved.\n";
    return 0;
}