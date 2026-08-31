import numpy as np
import os

T = 8
H, W = 14, 14
N_tokens = H * W
C_out = 768
num_heads = 12
d_head = C_out // num_heads

np.random.seed(42)
data_dir = "SSA_input_data"
os.makedirs(data_dir, exist_ok=True)

# Độ thưa của V cố định
v_sparsity = 70
v_p_zero = v_sparsity / 100.0
v_p_one = 1.0 - v_p_zero
# Độ thưa của Q, K thay đổi
sparsities = [30, 50, 70, 90]

print("Bắt đầu sinh dữ liệu ngẫu nhiên cho SSA...")

for sparsity in sparsities:
    p_zero = sparsity / 100.0
    p_one = 1.0 - p_zero
    # Tạo thư mục cho từng độ thưa
    cur_dir = os.path.join(data_dir, f"Sparsity_{sparsity}%")
    os.makedirs(cur_dir, exist_ok=True)
    
    for h in range(num_heads):
        # -----------------------------------------------------------
        # 1. SINH VÀ GHI MA TRẬN Q (Layout: [N_tokens][d_head][T])
        # -----------------------------------------------------------
        global_mask = np.random.choice([0, 1], size=(N_tokens, d_head), p=[p_zero, p_one])
        spikes = np.random.choice([0, 1], size=(N_tokens, d_head, T), p=[0.8, 0.2]).astype(np.int8)
        
        # Ép mask
        for n in range(N_tokens):
            for c in range(d_head):
                if global_mask[n, c] == 1:
                    if np.sum(spikes[n, c, :]) == 0:
                        spikes[n, c, np.random.randint(0, T)] = 1
                else:
                    spikes[n, c, :] = 0
        
        # Ghi file
        with open(os.path.join(cur_dir, f"Input_Q_head_{h}.txt"), "w") as f:
            for n in range(N_tokens):
                for c in range(d_head):
                    # Ghi 8 timesteps dính liền
                    f.write("".join(spikes[n, c].astype(str)))
                    f.write(" ")
                f.write("\n") # Xuống dòng khi hết 1 Token

        # ----------------------------------------------------------------------------
        # 2. SINH VÀ GHI MA TRẬN V (Layout: [N_tokens][d_head][T]) VỚI ĐỘ THƯA CỐ ĐỊNH
        # ----------------------------------------------------------------------------
        global_mask = np.random.choice([0, 1], size=(N_tokens, d_head), p=[v_p_zero, v_p_one])
        spikes = np.random.choice([0, 1], size=(N_tokens, d_head, T), p=[0.8, 0.2]).astype(np.int8)
        
        # Ép mask
        for n in range(N_tokens):
            for c in range(d_head):
                if global_mask[n, c] == 1:
                    if np.sum(spikes[n, c, :]) == 0:
                        spikes[n, c, np.random.randint(0, T)] = 1
                else:
                    spikes[n, c, :] = 0
        
        # Ghi file
        with open(os.path.join(cur_dir, f"Input_V_head_{h}.txt"), "w") as f:
            for n in range(N_tokens):
                for c in range(d_head):
                    # Ghi 8 timesteps dính liền
                    f.write("".join(spikes[n, c].astype(str)))
                    f.write(" ")
                f.write("\n") # Xuống dòng khi hết 1 Token

        # -----------------------------------------------------------
        # 3. SINH VÀ GHI MA TRẬN K^T (Layout: [d_head][N_tokens][T])
        # -----------------------------------------------------------
        global_mask_k = np.random.choice([0, 1], size=(d_head, N_tokens), p=[p_zero, p_one])
        spikes_k = np.random.choice([0, 1], size=(d_head, N_tokens, T), p=[0.8, 0.2]).astype(np.int8)
        
        for c in range(d_head):
            for n in range(N_tokens):
                if global_mask_k[c, n] == 1:
                    if np.sum(spikes_k[c, n, :]) == 0:
                        spikes_k[c, n, np.random.randint(0, T)] = 1
                else:
                    spikes_k[c, n, :] = 0
                    
        with open(os.path.join(cur_dir, f"Input_K_head_{h}.txt"), "w") as f:
            for c in range(d_head):
                for n in range(N_tokens):
                    f.write("".join(spikes_k[c, n].astype(str)))
                    f.write(" ")
                f.write("\n") # Xuống dòng khi hết 1 Channel

    print(f" -> Đã lưu xong dữ liệu Spikes Q, K, V cho Sparsity {sparsity}%")

print("\nHoàn tất khởi tạo dữ liệu SSA!")