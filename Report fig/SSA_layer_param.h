#pragma once
#include <iostream>

using namespace std;

struct SSA_param {
    int T; int H; int W;
    int C_out; int num_heads;
    int V_th; float leaky_factor;
    float scale;

    int N_tokens = H*W;
    // Dimension (số oc) của mỗi head trong phép Attention, PHẢI CHIA HẾT
    int d = C_out / num_heads;
    int SSA_final_size = C_out * N_tokens * T;
    int SSA_head_size = SSA_final_size / num_heads;
};

SSA_param p = {
    .T = 8, .H = 14, .W = 14,
    .C_out = 66, .num_heads = 6,
    .V_th = 4, .leaky_factor = 0.5f,
    .scale = 0.125f
};