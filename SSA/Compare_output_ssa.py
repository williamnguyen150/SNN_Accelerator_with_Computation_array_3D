import numpy as np
import os
import warnings

# Tắt cảnh báo dòng trống của thư viện Numpy (UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

sparsities = [30, 50, 70, 90]
golden_dir = "SSA_golden_output"
test_dir = "SSA_SNN_output"

mode_names = {
    1: "No SAS No NZ",
    2: "SAS Only",
    3: "SAS + NZ",
    4: "SAS + NZ + LB"
}

print("===================================================================")
print("1. COMPARE GOLDEN (PYTHON) VS C++ SNN ACCELERATOR (4 MODES)")
print("===================================================================")

for sparsity in sparsities:
    print(f"\n--- Checking Sparsity {sparsity}% ---")
    golden_file = f"{golden_dir}/SSA_golden_output_{sparsity}%.txt"
    
    for mode in range(1, 5):
        test_file = f"{test_dir}/Sparsity_{sparsity}%/SSA_SNN_output_Mode_{mode}.txt"
        diff = compare_files(golden_file, test_file)
        print(f"  [Mode {mode}] {mode_names[mode]:<15} vs Golden: {diff}")


print("\n===================================================================")
print("2. COMPARE OUTPUT BETWEEN HARDWARE MODES (CROSS-CHECK)")
print("===================================================================")

for sparsity in sparsities:
    print(f"\n--- Cross-checking Sparsity {sparsity}% ---")
    
    pairs = [(1, 2), (2, 3), (3, 4)]
    
    for mode_x, mode_y in pairs:
        file_x = f"{test_dir}/Sparsity_{sparsity}%/SSA_SNN_output_Mode_{mode_x}.txt"
        file_y = f"{test_dir}/Sparsity_{sparsity}%/SSA_SNN_output_Mode_{mode_y}.txt"
        diff = compare_files(file_x, file_y)
        print(f"  Mode {mode_x} vs Mode {mode_y}: {diff}")

print("\nAll done!")