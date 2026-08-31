#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "Sconv_layer_param.h"

using namespace std;

int8_t* load_txt_weight(const char* file, int size) {
    FILE* f = fopen(file, "r");
    if (!f) { printf("Khong the mo file %s\n", file); exit(1); }
    int8_t* a = (int8_t*)malloc(size * sizeof(int8_t));
    
    int temp; // Dùng biến int 4 bytes để hứng dữ liệu từ %d an toàn
    for (int i = 0; i < size; i++) {
        fscanf(f, "%d", &temp);
        a[i] = (int8_t)temp; // Ép kiểu về 1 byte
    }
    fclose(f);
    return a;
}

char* load_txt_spike(const char* file, int size) {
    FILE* f = fopen(file, "r");
    if (!f) { printf("Khong the mo file %s\n", file); exit(1); }
    char* a = (char*)malloc(size * sizeof(char));
    
    char c;
    int count = 0;
    // Đọc từng ký tự
    while (count < size && fscanf(f, " %c", &c) == 1) { 
        if (c == '0' || c == '1') {
            a[count++] = c - '0';
        }
    }
    fclose(f);
    return a;
}

void save_txt_spike(const char* file, char* output, int C_out, int H, int W, int T) {
    FILE* f = fopen(file, "w");
    if (f == NULL) {
        perror("Error opening file");
        return;
    }
    for (int oc = 0; oc < C_out; oc++) {
        for (int oh = 0; oh < H; oh++) {
            for (int ow = 0; ow < W; ow++) {
                for (int t = 0; t < T; t++) {
                    fprintf(f, "%d", output[oc*H*W*T + oh*W*T + ow*T + t]);
                }
                fprintf(f, " ");
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
    }
    fclose(f); 
}

void save_txt_v_mem(const char* file, int* output, int C_out, int H, int W, int T) {
    FILE* f = fopen(file, "w");
    if (f == NULL) {
        perror("Error opening file");
        return;
    }
    for (int oc = 0; oc < C_out; oc++) {
        for (int oh = 0; oh < H; oh++) {
            for (int ow = 0; ow < W; ow++) {
                for (int t = 0; t < T; t++) {
                    fprintf(f, "%d ", output[oc*H*W*T + oh*W*T + ow*T + t]);
                }
                fprintf(f, " ");
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
    }
    fclose(f); 
}

void conv2d(
    // các tham số
    char* input, int8_t* weight, int* bias,
    int* output_v_mem, char* output_spikes,
    long long* SOP_count,
    Sconv_param* p
) {
    *SOP_count = 0;

    // Tính từng pixel của output_mem qua các timesteps
    for (int t = 0; t < (*p).T; t++){
        for (int oc = 0; oc < (*p).C_out; oc++){
            for (int i = 0; i < (*p).H_out; i++){
                for (int j = 0; j < (*p).W_out; j++) {

                    int sum = (bias == NULL) ? 0 : bias[oc];

                    // Duyệt qua từng pixel input và kernel tương ứng
                    for (int ic = 0; ic < (*p).C_in; ic++){
                        for (int ki = 0; ki < (*p).K1; ki++){
                            for (int kj = 0; kj < (*p).K2; kj++) {

                                // Tính chỉ số của pixel trong input cần tính
                                int ih = i*(*p).stride + ki - (*p).padding;
                                int iw = j*(*p).stride + kj - (*p).padding;

                                if (ih >= 0 && ih < (*p).H && iw >= 0 && iw < (*p).W) {
                                    // Tìm chỉ số pixel cần tính trong mảng input và weight
                                    int in_idx = ic*(*p).H*(*p).W*(*p).T + ih*(*p).W*(*p).T + iw*(*p).T + t;
                                    int w_idx  = oc*(*p).C_in*(*p).K1*(*p).K2 + ic*(*p).K1*(*p).K2 + ki*(*p).K2 + kj;
                                    // Cộng dồn (WA)
                                    if (input[in_idx]) {
                                        sum += weight[w_idx];
                                        (*SOP_count)++;
                                    }
                                }
                            } // kết thúc tính tại kj
                        } // kết thúc tính tại ki, xong phép chập trên 1 channel input
                    } // kết thúc tính tại ic, xong phép chập trên 1 pixel output

                    // Gán P_sum vào output_v_mem
                    output_v_mem[oc*(*p).H_out*(*p).W_out*(*p).T + i*(*p).W_out*(*p).T + j*(*p).T + t] = sum;
                }
            }
        }
    }

    // MU và phát spikes
    for (int oc = 0; oc < (*p).C_out; oc++){
        for (int i = 0; i < (*p).H_out; i++){
            for (int j = 0; j < (*p).W_out; j++) {
                int v_mem = 0;
                for (int t = 0; t < (*p).T; t++) {
                    int out_idx = oc*(*p).H_out*(*p).W_out*(*p).T + i*(*p).W_out*(*p).T + j*(*p).T + t;
                    
                    // cộng dồn leaky p_sum từ timestep trước đó
                    v_mem = (v_mem >> 1) + output_v_mem[out_idx];
                    
                    // v_mem thực sự sau leaky và integrate
                    output_v_mem[out_idx] = v_mem; 
                    
                    // fire, hoàn thành mô hình LIF
                    output_spikes[out_idx] = (v_mem >= (*p).V_th) ? 1 : 0;
                    v_mem -= (v_mem >= (*p).V_th) ? (*p).V_th : 0;
                }
            }
        }
    }
}

int main() {
    vector<int> input_sparsity = {30, 50, 70, 90};
    string data_dir = "Sconv_input_and_weight_data";
    string out_dir = "Sconv_serial_output_sparse";
    filesystem::create_directories(out_dir);

    // Các mảng output dùng chung và ghi đè giữa các mode
    int* output_v_mem = (int*)calloc(p.output_size, sizeof(int));
    char* output_spikes = (char*)calloc(p.output_size, sizeof(char));
    long long* SOP_count = (long long*)calloc(1, sizeof(long long));

    // Nạp weight 1 lần dùng chung 3 mode
    string weight_file = data_dir + "/weight.txt";
    int8_t* weight = load_txt_weight(weight_file.c_str(), p.weight_size);

    for (int sparsity : input_sparsity) {
        cout << "\nRunning Serial Sconv with Global Sparsity: " << sparsity << "%\n";

        // Load input tương ứng với sparsity
        string input_file = data_dir + "/input_spikes_sparsity_" + to_string(sparsity) + "%.txt";
        char* input_spikes = load_txt_spike(input_file.c_str(), p.input_size);

        // Tính toán tuần tự (Golden output) với &p trỏ đến parameter của layer conv
        conv2d(input_spikes, weight, NULL, output_v_mem, output_spikes, SOP_count, &p);
        cout << "Total SOP (Golden): " << *SOP_count << "\n";

        // Lưu output
        string out1_file = out_dir + "/Sconv_serial_output_1_" + to_string(sparsity) + "%.txt";
        string out2_file = out_dir + "/Sconv_serial_output_2_" + to_string(sparsity) + "%.txt";
        save_txt_v_mem(out1_file.c_str(), output_v_mem, p.C_out, p.H_out, p.W_out, p.T);
        save_txt_spike(out2_file.c_str(), output_spikes, p.C_out, p.H_out, p.W_out, p.T);

        free(input_spikes);
    }

    free(weight); 
    free(output_spikes); 
    free(output_v_mem); 
    free(SOP_count);

    cout << "\nAll golden outputs saved.\n";
    return 0;
}