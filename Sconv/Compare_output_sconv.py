import numpy as np

# So sánh các output_sparse golden (Sconv serial) và test (với SNN Accelerator ở mode 3)
golden_dir = "Sconv_serial_output_sparse"
test_dir = "Sconv_SNN_output_sparse"

for sparsity in [30, 50, 70, 90]:
    print(f"\nCompare output with input sparsity {sparsity} %:")

    golden_output_1 = np.loadtxt(f"{golden_dir}/Sconv_serial_output_1_{sparsity}%.txt")
    test_output_1 = np.loadtxt(f"{test_dir}/Sconv_SNN_output_1_{sparsity}%.txt")
    diff1 = np.abs(golden_output_1 - test_output_1)

    print("Different of membrane potential output:", diff1.mean())

    golden_output_2 = np.loadtxt(f"{golden_dir}/Sconv_serial_output_2_{sparsity}%.txt")
    test_output_2 = np.loadtxt(f"{test_dir}/Sconv_SNN_output_2_{sparsity}%.txt")
    diff2 = np.abs(golden_output_2 - test_output_2)

    print("Different of LIF neuron output:", diff2.mean())

# So sánh các output của cùng 1 file input ở 3 Mode với nhau
output_dir = "Sconv_SNN_output_3_mode"
print("\nCompare output between 3 Mode:")

for mode_compare in [[1,2], [2,3], [3,1]]:
    output_1_mode_x = np.loadtxt(f"{output_dir}/Sconv_SNN_output_1_Mode_{mode_compare[0]}.txt")
    output_1_mode_y = np.loadtxt(f"{output_dir}/Sconv_SNN_output_1_Mode_{mode_compare[1]}.txt")
    diff1 = np.abs(output_1_mode_x - output_1_mode_y)
    print(f"Different of membrane potential output Mode {mode_compare[0]} and Mode {mode_compare[1]}: {diff1.mean()}")

    output_2_mode_x = np.loadtxt(f"{output_dir}/Sconv_SNN_output_2_Mode_{mode_compare[0]}.txt")
    output_2_mode_y = np.loadtxt(f"{output_dir}/Sconv_SNN_output_2_Mode_{mode_compare[1]}.txt")
    diff2 = np.abs(output_2_mode_x - output_2_mode_y)
    print(f"Different of LIF neuron output Mode {mode_compare[0]} and Mode {mode_compare[1]}: {diff2.mean()}")