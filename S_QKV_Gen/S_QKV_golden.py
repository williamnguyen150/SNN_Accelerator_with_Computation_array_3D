import numpy as np
import os

T = 8
C_in = 768
H, W = 14, 14
N_tokens = H * W
C_out = 768
num_heads = 12
d_head = C_out // num_heads
V_th = 381

def load_spike_file(filename, shape):
    """ Hàm bóc tách từng ký tự '0', '1' chống lỗi dính bit """
    if not os.path.exists(filename):
        print(f"[LỖI] Không tìm thấy file {filename}")
        return np.zeros(shape, dtype=np.int8)
        
    with open(filename, 'r') as f:
        raw_text = f.read()
    
    # Chỉ bóc đúng ký tự 0 và 1
    bits = [int(char) for char in raw_text if char in '01']
    arr = np.array(bits, dtype=np.int8)
    
    if arr.size != np.prod(shape):
        print(f"[CẢNH BÁO] {filename} có {arr.size} bit, nhưng shape yêu cầu {np.prod(shape)}")
        
    return arr.reshape(shape)

def load_weight_file(filename, shape):
    """ Hàm đọc trọng số nguyên chuẩn có khoảng trắng """
    if not os.path.exists(filename):
        print(f"[LỖI] Không tìm thấy file {filename}")
        return np.zeros(shape, dtype=np.int32)
    
    arr = np.loadtxt(filename, dtype=np.int32)
    return arr.reshape(shape)

def run_lif_neuron(Psum, v_th):
    C, N, Time = Psum.shape
    Spike_out = np.zeros_like(Psum, dtype=np.int8)
    
    for c in range(C):
        for n in range(N):
            v_mem = 0
            for t in range(Time):
                # Dùng >> 1 để leaky 0,5
                v_mem = (v_mem >> 1) + Psum[c, n, t]
                
                if v_mem >= v_th:
                    Spike_out[c, n, t] = 1
                    v_mem -= v_th
                else:
                    Spike_out[c, n, t] = 0

    return Spike_out

data_dir = "S_QKV_input_and_weight_data"
out_dir_base = "S_QKV_golden_output_sparse"

# Trọng số layout: [C_out][C_in] - Chỉ cần load 1 lần cho mọi độ thưa
W_q = load_weight_file(f"{data_dir}/weight_q.txt", (C_out, C_in))
W_k = load_weight_file(f"{data_dir}/weight_k.txt", (C_out, C_in))
W_v = load_weight_file(f"{data_dir}/weight_v.txt", (C_out, C_in))

sparsities = [30, 50, 70, 90]
for sparsity in sparsities:
    print(f"\n--- Đang xử lý Golden Output cho Sparsity {sparsity}% ---")
    
    # Tạo thư mục tương ứng cho từng độ thưa
    out_dir = os.path.join(out_dir_base, f"Sparsity_{sparsity}%")
    os.makedirs(out_dir, exist_ok=True)
    
    # Nạp Spike Input
    input_file = f"{data_dir}/input_spikes_sparsity_{sparsity}%.txt"
    X_4D = load_spike_file(input_file, (C_in, H, W, T))
    
    # Trộn trục H và W thành N_tokens để tính chập 1x1 (Linear)
    X = X_4D.reshape(C_in, N_tokens, T)

    # Phép nhân ma trận: (C_out, C_in) x (C_in, N_tokens, T) -> Kết quả: (C_out, N_tokens, T)
    Psum_Q = np.tensordot(W_q, X, axes=([1], [0]))
    Psum_K = np.tensordot(W_k, X, axes=([1], [0]))
    Psum_V = np.tensordot(W_v, X, axes=([1], [0]))

    # Cập nhật LIF Neuron
    Spikes_Q = run_lif_neuron(Psum_Q, V_th)
    Spikes_K = run_lif_neuron(Psum_K, V_th)
    Spikes_V = run_lif_neuron(Psum_V, V_th)

    # Reshape để phân chia dimension C_out thành (num_heads, d_head)
    Spikes_Q_heads = Spikes_Q.reshape(num_heads, d_head, N_tokens, T)
    Spikes_K_heads = Spikes_K.reshape(num_heads, d_head, N_tokens, T)
    Spikes_V_heads = Spikes_V.reshape(num_heads, d_head, N_tokens, T)

    # Ghi file cho từng Head
    for h in range(num_heads):
        # 1. Ghi ma trận Q (transpose để lưu theo thứ tự [n, c, t])
        Q_h = Spikes_Q_heads[h].transpose(1, 0, 2)
        with open(f"{out_dir}/Golden_output_Q_head_{h}.txt", "w") as f:
            for n in range(N_tokens):
                for c in range(d_head):
                    for t in range(T):
                        f.write(f"{Q_h[n, c, t]}")
                    f.write(" ")
                f.write("\n")

        # 2. Ghi ma trận K (lưu theo thứ tự [c, n, t])
        K_h = Spikes_K_heads[h]
        with open(f"{out_dir}/Golden_output_K_head_{h}.txt", "w") as f:
            for c in range(d_head):
                for n in range(N_tokens):
                    for t in range(T):
                        f.write(f"{K_h[c, n, t]}")
                    f.write(" ")
                f.write("\n")

        # 3. Ghi ma trận V (transpose để lưu theo thứ tự [n, c, t])
        V_h = Spikes_V_heads[h].transpose(1, 0, 2)
        with open(f"{out_dir}/Golden_output_V_head_{h}.txt", "w") as f:
            for n in range(N_tokens):
                for c in range(d_head):
                    for t in range(T):
                        f.write(f"{V_h[n, c, t]}")
                    f.write(" ")
                f.write("\n")

print("\nALl Golden Outputs Saved!")