import numpy as np
import os

T = 8
C_out = 768
num_heads = 12
d = C_out // num_heads
H = 14
W = 14
N_tokens = H * W
V_th = 20
leaky_factor = 0.5
scale = 0.125

def load_spike_file(filename, shape):
    if not os.path.exists(filename):
        print(f"    [LỖI] Không tìm thấy file {filename}")
        return np.zeros(shape, dtype=np.int32)
        
    with open(filename, 'r') as f:
        raw_text = f.read()
    
    # Chỉ lấy đúng ký tự '0' và '1'
    bits = [int(char) for char in raw_text if char in '01']
    
    # Ép về Numpy Array và Reshape theo Layout mong muốn
    arr = np.array(bits, dtype=np.int32)
    
    if arr.size != np.prod(shape):
        print(f"    [CẢNH BÁO] Kích thước file {filename} ({arr.size}) không khớp với shape {shape}")
        
    return arr.reshape(shape)

sparsities = [30, 50, 70, 90]
data_dir = "SSA_input_data"
out_dir = "SSA_golden_output"

# Đảm bảo thư mục output gốc tồn tại
os.makedirs(out_dir, exist_ok=True)

for sparsity in sparsities:
    print(f"\n--- Đang xử lý Golden Output cho Sparsity {sparsity}% ---")
    
    cur_data_dir = f"{data_dir}/Sparsity_{sparsity}%"
    
    final_output = []
    
    for h in range(num_heads):
        Q = load_spike_file(f"{cur_data_dir}/Input_Q_head_{h}.txt", (N_tokens, d, T))
        V = load_spike_file(f"{cur_data_dir}/Input_V_head_{h}.txt", (N_tokens, d, T))
        K_T = load_spike_file(f"{cur_data_dir}/Input_K_head_{h}.txt", (d, N_tokens, T))

        SSA_Psum = np.zeros((N_tokens, d, T), dtype=np.int32)
        for t in range(T):
            Q_t = Q[:, :, t]      # Kích thước (N, d)
            K_T_t = K_T[:, :, t]  # Kích thước (d, N)
            V_t = V[:, :, t]      # Kích thước (N, d)
            S_t = np.dot(K_T_t, V_t)
            Psum_t = np.dot(Q_t, S_t)
            SSA_Psum[:, :, t] = Psum_t

        # Mảng chứa output cuối cùng của head này
        SSA_output = np.zeros((N_tokens, d, T), dtype=np.int32)
        # Quá trình cập nhật điện thế màng (MU)
        for n in range(N_tokens):
            for c in range(d):
                v_mem = 0
                for t in range(T):
                    # v_mem >> 1 để leaky 0.5, Psum >> 3 để nhân scale 0.125
                    v_mem = (v_mem >> 1) + SSA_Psum[n, c, t]
                    
                    if v_mem >= V_th:
                        SSA_output[n, c, t] = 1
                        v_mem -= V_th  # Soft reset
                    else:
                        SSA_output[n, c, t] = 0

        final_output.append(SSA_output)

    final_array = np.array(final_output)
    
    # Hoán đổi chiều d và N về (num_heads, d, N_tokens, T)
    final_array= final_array.transpose(0, 2, 1, 3)

    # Gộp num_heads và d thành C_out, tách N_tokens ra thành H, W
    # Layout cuối cùng: (C_out, H, W, T)
    final_output_reshaped = final_array.reshape(C_out, H, W, T)

    output_filename = f"{out_dir}/SSA_golden_output_{sparsity}%.txt"

    # Ghi ra file với định dạng [C_out][H][W][T]
    with open(output_filename, "w") as f:
        for c in range(C_out):
            for h_idx in range(H):
                for w_idx in range(W):
                    for t in range(T):
                        f.write(f"{final_output_reshaped[c, h_idx, w_idx, t]}")
                    f.write(" ")
                f.write("\n")
            f.write("\n")
            
    print(f"Đã lưu thành công {output_filename}")

print("\nHoàn tất sinh toàn bộ Golden Outputs cho SSA!")