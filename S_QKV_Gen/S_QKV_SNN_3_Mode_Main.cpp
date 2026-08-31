#include <filesystem>   // Tạo thư mục lưu output
#include <fstream>
#include <string>
#include "S_QKV_SNN_Scheduler.h"

int main() {
    string data_dir = "S_QKV_input_and_weight_data";
    string out_dir = "S_QKV_SNN_output_3_mode";
    filesystem::create_directories(out_dir);
    for (int mode : {1, 2, 3}) {
        string cur_dir = out_dir + "/Mode_" + to_string(mode);
        filesystem::create_directories(cur_dir);
    }

    string input_dir = data_dir + "/input_spikes_sparsity_90%" + ".txt";
    vector<char> input_spikes = load_txt_spike(input_dir, p.input_size);

    vector<int8_t> weight_Q = load_txt_weight(data_dir + "/weight_q.txt", p.weight_size);
    vector<int8_t> weight_K = load_txt_weight(data_dir + "/weight_k.txt", p.weight_size);
    vector<int8_t> weight_V = load_txt_weight(data_dir + "/weight_v.txt", p.weight_size);
    vector<int> bias(p.bias_size, 0);

    vector<vector<char>> output_Q (p.num_heads, vector<char>(p.output_size_QKV_each_head, 0));
    vector<vector<char>> output_K (p.num_heads, vector<char>(p.output_size_QKV_each_head, 0));
    vector<vector<char>> output_V (p.num_heads, vector<char>(p.output_size_QKV_each_head, 0));

    // Cấp phát động trên Heap, tránh tràn stack do kích thước hw khá lớn
    SNN_Hardware_Scheduler* hw = new SNN_Hardware_Scheduler();
    
    // Chạy Mode 1: No NZ
    cout << "[MODE 1] No NZ:\n";
    hw->reset_all_data();
    hw->SQKV_Scheduler(
        input_spikes, bias, weight_Q, weight_K, weight_V, 
        output_Q, output_K, output_V, p, false, false);
    hw->count_metrics();
    hw->print_metrics();
    for (int head_idx = 0; head_idx < p.num_heads; head_idx++) {
        string q_filename = "/Mode_1/SNN_output_Q_head_" + to_string(head_idx) + ".txt";
        string k_filename = "/Mode_1/SNN_output_K_head_" + to_string(head_idx) + ".txt";
        string v_filename = "/Mode_1/SNN_output_V_head_" + to_string(head_idx) + ".txt";

        save_txt_QV_LIF(out_dir + q_filename, output_Q[head_idx], p.N_tokens, p.d_heads, p.T);
        save_txt_K_Transpose_LIF(out_dir + k_filename, output_K[head_idx], p.N_tokens, p.d_heads, p.T);
        save_txt_QV_LIF(out_dir + v_filename, output_V[head_idx], p.N_tokens, p.d_heads, p.T);
    }

    // Chạy Mode 2: NZ Only
    cout << "[MODE 2] NZ Only:\n";
    hw->reset_all_data();
    hw->SQKV_Scheduler(
        input_spikes, bias, weight_Q, weight_K, weight_V, 
        output_Q, output_K, output_V, p, true, false);
    hw->count_metrics();
    hw->print_metrics();
    for (int head_idx = 0; head_idx < p.num_heads; head_idx++) {
        string q_filename = "/Mode_2/SNN_output_Q_head_" + to_string(head_idx) + ".txt";
        string k_filename = "/Mode_2/SNN_output_K_head_" + to_string(head_idx) + ".txt";
        string v_filename = "/Mode_2/SNN_output_V_head_" + to_string(head_idx) + ".txt";

        save_txt_QV_LIF(out_dir + q_filename, output_Q[head_idx], p.N_tokens, p.d_heads, p.T);
        save_txt_K_Transpose_LIF(out_dir + k_filename, output_K[head_idx], p.N_tokens, p.d_heads, p.T);
        save_txt_QV_LIF(out_dir + v_filename, output_V[head_idx], p.N_tokens, p.d_heads, p.T);
    }

    // Chạy Mode 3: NZ + LB
    cout << "[MODE 3] NZ + LB:\n";
    hw->reset_all_data();
    hw->SQKV_Scheduler(
        input_spikes, bias, weight_Q, weight_K, weight_V, 
        output_Q, output_K, output_V, p, true, true);
    hw->count_metrics();
    hw->print_metrics();   
    for (int head_idx = 0; head_idx < p.num_heads; head_idx++) {
        string q_filename = "/Mode_3/SNN_output_Q_head_" + to_string(head_idx) + ".txt";
        string k_filename = "/Mode_3/SNN_output_K_head_" + to_string(head_idx) + ".txt";
        string v_filename = "/Mode_3/SNN_output_V_head_" + to_string(head_idx) + ".txt";

        save_txt_QV_LIF(out_dir + q_filename, output_Q[head_idx], p.N_tokens, p.d_heads, p.T);
        save_txt_K_Transpose_LIF(out_dir + k_filename, output_K[head_idx], p.N_tokens, p.d_heads, p.T);
        save_txt_QV_LIF(out_dir + v_filename, output_V[head_idx], p.N_tokens, p.d_heads, p.T);
    }
    
    delete hw;
    cout << "All output saved.\n";
    return 0;
}