#pragma once
#include <iostream>

using namespace std;

struct SQKV_param {
    int T; int C_in; int H; int W;
    int C_out; int num_heads;
    int V_th; float leaky_factor;

    int N_tokens = H*W;
    int d_heads = C_out / num_heads;
    int input_size = T * C_in * N_tokens;
    int weight_size = C_out * C_in;
    int bias_size = C_out;
    int output_size_QKV_each_head = N_tokens * d_heads * T;
};

SQKV_param p = {
    .T = 8, .C_in = 66, .H = 14, .W = 14,
    .C_out = 66, . num_heads = 6,
    .V_th = 40, .leaky_factor = 0.5f
};