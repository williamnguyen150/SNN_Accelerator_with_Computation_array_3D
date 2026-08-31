import numpy as np
import os

T, C_in, H, W = 8, 512, 28, 28
C_out, K1, K2 = 768, 3, 3

input_size = T * C_in * H * W
weight_size = C_out * C_in * K1 * K2

np.random.seed(42)

# Tạo thư mục lưu input
data_dir = "Sconv_input_and_weight_data"
os.makedirs(data_dir, exist_ok=True)

# 1. TẠO VÀ LƯU WEIGHT THEO ĐÚNG SHAPE [C_out][C_in][K1][K2]
weight_shape = (C_out, C_in, K1, K2)
weight = np.clip(np.random.normal(0, 40, size=weight_shape), -128, 127).astype(np.int8)

with open(f"{data_dir}/weight.txt", "w") as f:
    for oc in range(C_out):
        for ic in range(C_in):
            for k1 in range(K1):
                for k2 in range(K2):
                    f.write(f"{weight[oc, ic, k1, k2]} ")
                f.write("\n") # Xuống dòng hết 1 hàng ngang của Kernel
            f.write("\n")     # Xuống dòng hết 1 channel
        f.write("\n")         # Xuống dòng hết 1 output channel

print(f"Đã lưu weight.txt với shape {weight_shape}")

# 2. TẠO VÀ LƯU SPIKES THEO CÁC ĐỘ THƯA GLOBAL BITMAP
sparsities = [30, 50, 70, 90]

for sparsity in sparsities:
    p_zero = sparsity / 100.0  # Xác suất pixel bằng 0 trên TOÀN BỘ timesteps
    p_one = 1.0 - p_zero
    
    # Tạo Global Mask [C_in, H, W]
    global_mask = np.random.choice([0, 1], size=(C_in, H, W), p=[p_zero, p_one])
    # Tạo spike ban đầu cho mọi vị trí
    spikes = np.random.choice([0, 1], size=(C_in, H, W, T), p=[0.8, 0.2]).astype(np.int8)
    # Tạo spike hoàn chỉnh
    for c in range(C_in):
        for h_idx in range(H):
            for w_idx in range(W):
                if global_mask[c, h_idx, w_idx] == 1:
                    # Nếu mask = 1 mà ngẫu nhiên sinh ra toàn 0 thì phải bắt ép có ít nhất 1 spike
                    if np.sum(spikes[c, h_idx, w_idx, :]) == 0:
                        t_rand = np.random.randint(0, T)
                        spikes[c, h_idx, w_idx, t_rand] = 1
                else:
                    # Nếu mask = 0 thì tắt mọi spike ở tất cả timesteps
                    spikes[c, h_idx, w_idx, :] = 0
                    
    # Ghi ra file theo format [C_in][H][W][T]
    with open(f"{data_dir}/input_spikes_sparsity_{sparsity}%.txt", "w") as f:
        for c in range(C_in):
            for h_idx in range(H):
                for w_idx in range(W):
                    for t in range(T):
                        f.write(f"{spikes[c, h_idx, w_idx, t]}")
                    f.write(" ")
                f.write("\n")
            f.write("\n")
            
    print(f"Đã lưu input_spikes.txt với global sparsity ~ {sparsity}%")