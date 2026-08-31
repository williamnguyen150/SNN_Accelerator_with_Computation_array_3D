#include <filesystem>   // Tạo thư mục lưu output
#include <fstream>
#include <string>
#include "Sconv_SNN_Scheduler.h"

// Chạy 1 file input thì không in metrics ra csv, chỉ in ra console
// Lưu cả 6 output ứng với 3 mode

int main() {
    string data_dir = "Sconv_input_and_weight_data";
    string out_dir = "Sconv_SNN_output_3_mode";
    filesystem::create_directories(out_dir);

    // Thay path của input spike và weight muốn tính: chỉ chạy 1 input để so sánh output của 3 mode
    vector<char> input_spikes = load_txt_spike(data_dir + "/input_spikes_sparsity_90%.txt", p.input_size);
    vector<int8_t> weight = load_txt_weight(data_dir + "/weight.txt", p.weight_size);
    vector<int> bias(p.C_out, 0);
    vector<int> output_v_mem(p.output_size, 0);
    vector<char> output_spikes(p.output_size, 0);

    // Cấp phát động trên Heap, tránh tràn stack do kích thước hw khá lớn
    SNN_Hardware_Scheduler* hw = new SNN_Hardware_Scheduler();

    string output_1_dir = out_dir + "/Sconv_SNN_output_1_";
    string output_2_dir = out_dir + "/Sconv_SNN_output_2_";
    
    // Chạy Mode 1: No NZ
    cout << "\n[MODE 1] No NZ:\n";
    hw->reset_all_data();
    hw->SConv_Scheduler(input_spikes, weight, bias, output_v_mem, output_spikes, p, false, false);
    hw->count_metrics();
    hw->print_metrics();
    save_txt_Sconv_v_mem(output_1_dir + "Mode_1.txt", output_v_mem, p.C_out, p.H_out, p.W_out, p.T);
    save_txt_Sconv_LIF(output_2_dir + "Mode_1.txt", output_spikes, p.C_out, p.H_out, p.W_out, p.T);

    // Chạy Mode 2: NZ Only
    cout << "\n[MODE 2] NZ Only:\n";
    hw->reset_all_data();
    hw->SConv_Scheduler(input_spikes, weight, bias, output_v_mem, output_spikes, p, true, false);
    hw->count_metrics();
    hw->print_metrics();
    save_txt_Sconv_v_mem(output_1_dir + "Mode_2.txt", output_v_mem, p.C_out, p.H_out, p.W_out, p.T);
    save_txt_Sconv_LIF(output_2_dir + "Mode_2.txt", output_spikes, p.C_out, p.H_out, p.W_out, p.T);

    // Chạy Mode 3: NZ + LB
    cout << "\n[MODE 3] NZ + LB:\n";
    hw->reset_all_data();
    hw->SConv_Scheduler(input_spikes, weight, bias, output_v_mem, output_spikes, p, true, true);
    hw->count_metrics();
    hw->print_metrics();
    save_txt_Sconv_v_mem(output_1_dir + "Mode_3.txt", output_v_mem, p.C_out, p.H_out, p.W_out, p.T);
    save_txt_Sconv_LIF(output_2_dir + "Mode_3.txt", output_spikes, p.C_out, p.H_out, p.W_out, p.T);
    
    delete hw;
    cout << "All output saved.\n";
    return 0;
}