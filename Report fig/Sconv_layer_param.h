#pragma once
#include <iostream>

using namespace std;

struct Sconv_param {
    int T; int C_in; int H; int W;
    int C_out; int K1; int K2;
    int padding; int stride;
    int V_th; float leaky_factor;

    int input_size = T * C_in * H * W;
    int weight_size = C_out * C_in * K1 * K2;
    int bias_size = C_out * C_in * K1 * K2;
    int H_out = (H + 2 * padding - K1) / stride + 1;
    int W_out = (W + 2 * padding - K2) / stride + 1;
    int output_size = C_out * H_out * W_out * T;
};

Sconv_param p = {
    .T = 8, .C_in = 32, .H = 28, .W = 28,
    .C_out = 66, .K1 = 3, .K2 = 3,
    .padding = 1, .stride = 2,
    .V_th = 254, .leaky_factor = 0.5f
};