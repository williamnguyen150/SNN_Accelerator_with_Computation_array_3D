import numpy as np
import os

T, C_in, H, W = 8, 768, 14, 14
N_tokens = H*W
C_out = 768
input_size = T * C_in * N_tokens
weight_size = C_out * C_in

np.random.seed(42)

data_dir = "S_QKV_input_and_weight_data"
os.makedirs(data_dir, exist_ok=True)

weight_shape = (C_out, C_in)

# Sinh 3 ma trận weight riêng biệt cho Q, K, V
for name in ['q', 'k', 'v']:
    weights = np.clip(np.random.normal(0, 40, size=weight_shape), -128, 127).astype(np.int8)
    with open(f"{data_dir}/weight_{name}.txt", "w") as f:
        for oc in range(C_out):
            for ic in range(C_in):
                f.write(f"{weights[oc, ic]} ")
            f.write("\n") # Xuống dòng hết 1 output channel
    print(f"Đã lưu weight_{name}.txt")

sparsities = [30, 50, 70, 90]

for sparsity in sparsities:
    p_zero = sparsity / 100.0  # Xác suất pixel bằng 0 trên TOÀN BỘ timesteps
    p_one = 1.0 - p_zero
    
    # Tạo Global Mask [C_in, N_tokens]
    global_mask = np.random.choice([0, 1], size=(C_in, N_tokens), p=[p_zero, p_one])
    # Tạo spike ban đầu cho mọi vị trí
    spikes = np.random.choice([0, 1], size=(C_in, N_tokens, T), p=[0.8, 0.2]).astype(np.int8)
    
    # Tạo spike hoàn chỉnh dựa trên Global Mask
    for c in range(C_in):
        for n in range(N_tokens):
            if global_mask[c, n] == 1:
                # Nếu mask = 1 mà ngẫu nhiên sinh ra toàn 0 thì phải bắt ép có ít nhất 1 spike
                if np.sum(spikes[c, n, :]) == 0:
                    t_rand = np.random.randint(0, T)
                    spikes[c, n, t_rand] = 1
            else:
                # Nếu mask = 0 thì tắt mọi spike ở tất cả timesteps
                spikes[c, n, :] = 0
                
    # Ghi ra file theo format [C_in][H][W][T] giống output Sconv
    # Khi tính QKV sẽ tự gộp 2 chiều [H][W] thành [N_tokens]
    with open(f"{data_dir}/input_spikes_sparsity_{sparsity}%.txt", "w") as f:
        for c in range(C_in):
            for h in range(H):
                for w in range(W):
                    for t in range(T):
                        f.write(f"{spikes[c, h * W + w, t]}")
                    f.write(" ")
                f.write("\n")
            f.write("\n")
        f.write("\n")
            
    print(f"Đã lưu input_spikes.txt với global sparsity ~ {sparsity}%")