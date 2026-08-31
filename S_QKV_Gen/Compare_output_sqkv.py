import numpy as np
import os
import warnings

# Tắt cảnh báo dòng trống của thư viện Numpy
warnings.filterwarnings("ignore", category=UserWarning)

num_heads = 12
matrices = ['Q', 'K', 'V']

def compare_files(file1, file2):
    if not os.path.exists(file1):
        print(f"    [LỖI] Thiếu file: {file1}")
        return 1.0
    if not os.path.exists(file2):
        print(f"    [LỖI] Thiếu file: {file2}")
        return 1.0
    
    arr1 = np.loadtxt(file1, dtype=str)
    arr2 = np.loadtxt(file2, dtype=str)
    
    if arr1.shape != arr2.shape:
        print(f"    [LỖI] Lệch Shape: {file1} ({arr1.shape}) vs {file2} ({arr2.shape})")
        return 1.0
        
    # Tính tỷ lệ sai lệch trung bình (nếu giống nhau 100% thì trả về 0.0)
    return np.mean(arr1 != arr2)

# ==============================================================================
# 1. SO SÁNH GOLDEN (PYTHON) VÀ TEST (C++ SNN ACCELERATOR) Ở CÁC ĐỘ THƯA
# ==============================================================================
golden_dir = "S_QKV_golden_output_sparse"
test_dir = "S_QKV_SNN_output_sparse"

print("=====================================================")
print("1. COMPARE GOLDEN (PYTHON) VS C++ SNN ACCELERATOR")
print("=====================================================")

for sparsity in [30, 50, 70, 90]:
    print(f"\nCompare output with input sparsity {sparsity}%:")
    
    for mat in matrices:
        total_diff = 0
        for h in range(num_heads):
            f_golden = f"{golden_dir}/Sparsity_{sparsity}%/Golden_output_{mat}_head_{h}.txt"
            f_test = f"{test_dir}/Sparsity_{sparsity}%/SNN_output_{mat}_head_{h}.txt"
            
            total_diff += compare_files(f_golden, f_test)
            
        avg_diff = total_diff / num_heads
        
        print(f"  - Different of LIF neuron output {mat}: {avg_diff}")

# ==============================================================================
# 2. SO SÁNH CHÉO CÁC OUTPUT GIỮA 3 MODE CỦA ACCELERATOR
# ==============================================================================
mode_dir = "S_QKV_SNN_output_3_mode"

print("\n=====================================================")
print("2. COMPARE OUTPUT BETWEEN 3 HARDWARE MODES")
print("=====================================================")

for mode_compare in [[1,2], [2,3], [3,1]]:
    mode_x, mode_y = mode_compare[0], mode_compare[1]
    print(f"\nCompare Mode {mode_x} vs Mode {mode_y}:")
    
    for mat in matrices:
        total_diff = 0
        for h in range(num_heads):
            f_mode_x = f"{mode_dir}/Mode_{mode_x}/SNN_output_{mat}_head_{h}.txt"
            f_mode_y = f"{mode_dir}/Mode_{mode_y}/SNN_output_{mat}_head_{h}.txt"
            
            total_diff += compare_files(f_mode_x, f_mode_y)
            
        avg_diff = total_diff / num_heads
        
        print(f"  - Different of LIF neuron output {mat} Mode {mode_x} and Mode {mode_y}: {avg_diff}")