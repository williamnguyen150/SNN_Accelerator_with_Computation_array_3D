import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def format_metric_value(value):
    """Format số có dấu phẩy mỗi 3 chữ số; giữ 2 số lẻ cho số thực."""
    if pd.isna(value):
        return "0"
    value = float(value)
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def add_value_labels(ax, bars, values, offset_ratio=0.015, fontsize=9):
    """
    Ghi giá trị lên đầu cột. Nếu label quá rộng so với khoảng cách giữa
    các cột, tự động xoay label 90 độ để tránh chèn lên nhau.
    """
    values = list(values)
    max_value = max([abs(float(v)) for v in values], default=0)
    offset = max_value * offset_ratio if max_value > 0 else 0.1

    texts = []
    for bar, value in zip(bars, values):
        yval = bar.get_height()
        text = ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + offset,
            format_metric_value(value),
            ha='center',
            va='bottom',
            fontsize=fontsize,
            rotation=0
        )
        texts.append(text)

    # Đo độ rộng thật của chữ trên canvas để quyết định có cần xoay hay không.
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()

    centers = sorted(bar.get_x() + bar.get_width() / 2 for bar in bars)
    if len(centers) > 1:
        min_center_distance = min(b - a for a, b in zip(centers, centers[1:]))
        p0 = ax.transData.transform((0, 0))[0]
        p1 = ax.transData.transform((min_center_distance, 0))[0]
        safe_width_px = abs(p1 - p0) * 0.82
    else:
        safe_width_px = float('inf')

    need_rotation = any(
        text.get_window_extent(renderer=renderer).width > safe_width_px
        for text in texts
    )

    if need_rotation:
        for text in texts:
            text.set_rotation(90)
            text.set_ha('center')
            text.set_va('bottom')

    return need_rotation

csv_file = "metrics_ssa.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Không tìm thấy file {csv_file}.")
    exit()

# Xóa khoảng trắng thừa ở tên cột (nếu có)
df.columns = df.columns.str.strip()

# Các Mode và mức Sparsity
modes = ["No SAS No NZ", "SAS Only", "SAS + NZ", "SAS + NZ + LB"]
# Lọc ra các mode thực tế có trong file CSV để tránh lỗi nếu file thiếu dữ liệu
modes = [m for m in modes if m in df['Mode'].values]
sparsities = sorted(df['Sparsity(%)'].unique())

# Tách riêng 2 cột Total và Actual SSA operations để vẽ riêng, các cột còn lại vẽ chung
metrics_to_plot = [col for col in df.columns if col not in ['Sparsity(%)', 'Mode', 'Total_SSA_wo_SAS', 'Total_SSA_w_SAS']]

# Tạo thư mục để lưu ảnh biểu đồ
output_dir = "SSA_charts_output"
os.makedirs(output_dir, exist_ok=True)

# VẼ CÁC BIỂU ĐỒ THÔNG SỐ CHUNG (MỖI ẢNH 1 THÔNG SỐ CỦA 4 MODES)
print("Bắt đầu vẽ các biểu đồ")
bar_width = 0.2
x_indexes = np.arange(len(sparsities))
# 4 màu cho 4 Mode: Xanh dương, Cam, Xanh lá, Đỏ
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 

for metric in metrics_to_plot:
    plt.figure(figsize=(12, 6))
    
    # Vẽ cột cho từng Mode
    all_bars = []
    all_values = []
    for i, mode in enumerate(modes):
        mode_data = df[df['Mode'] == mode]

        # Lấy giá trị tương ứng với từng mức sparsity
        values = []
        for sparsity in sparsities:
            val = mode_data[mode_data['Sparsity(%)'] == sparsity][metric].values
            values.append(val[0] if len(val) > 0 else 0)

        # Tính toán vị trí x để 4 cột đứng cạnh nhau cân xứng
        x_pos = x_indexes + (i - 1.5) * bar_width
        bars = plt.bar(x_pos, values, width=bar_width, label=mode, color=colors[i], edgecolor='black')
        all_bars.extend(bars)
        all_values.extend(values)

    # Tự động thêm dấu phẩy và xoay dọc nếu label quá rộng
    add_value_labels(plt.gca(), all_bars, all_values, offset_ratio=0.015, fontsize=8)

    # Trang trí
    metric_name = metric.replace('_', ' ') 
    plt.xlabel('Global Input Sparsity (%)', fontsize=12, fontweight='bold')
    plt.ylabel(metric_name, fontsize=12, fontweight='bold')
    plt.title(f'SSA: Comparison of {metric_name} across 4 Hardware Modes', fontsize=14, fontweight='bold')
    plt.xticks(x_indexes, [f'{s}%' for s in sparsities], fontsize=11)
    plt.legend(title="Hardware Mode", fontsize=10, title_fontsize='11')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Mở rộng trục Y thêm 25% để không bị lẹm text xoay dọc
    max_y = df[metric].max()
    plt.ylim(0, max_y * 1.25 if max_y > 0 else 1) 
    
    plt.tight_layout()
    filename = f"{output_dir}/Chart_SSA_{metric}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f" -> Đã lưu: {filename}")

# VẼ BIỂU ĐỒ HIỆU QUẢ CỦA MODULE SAS (TOTAL OPS VS ACTUAL OPS)
if 'Total_SSA_wo_SAS' in df.columns and 'Total_SSA_w_SAS' in df.columns:
    plt.figure(figsize=(10, 6))
    sas_bar_width = 0.35
    sas_x_indexes = np.arange(len(sparsities))
    
    # Chỉ đọc dữ liệu từ Mode 2 (SAS Only), thực tế 4 Modes phải cho ra cùng 1 kết quả
    mode2_data = df[df['Mode'] == 'SAS Only']
    total_ops = []
    actual_ops = []
    
    for sparsity in sparsities:
        row = mode2_data[mode2_data['Sparsity(%)'] == sparsity]
        if not row.empty:
            total_ops.append(row['Total_SSA_wo_SAS'].values[0])
            actual_ops.append(row['Total_SSA_w_SAS'].values[0])
        else:
            total_ops.append(0)
            actual_ops.append(0)

    # Vẽ 2 cột so sánh
    bars1 = plt.bar(sas_x_indexes - sas_bar_width/2, total_ops, width=sas_bar_width, 
                    label='Total SSA Operations w/o SAS', color='#7f7f7f', edgecolor='black')
    bars2 = plt.bar(sas_x_indexes + sas_bar_width/2, actual_ops, width=sas_bar_width, 
                    label='Total SSA Operations w SAS', color='#2ca02c', edgecolor='black')

    # Tự động thêm dấu phẩy và xoay dọc nếu label quá rộng
    add_value_labels(
        plt.gca(), list(bars1) + list(bars2), total_ops + actual_ops,
        offset_ratio=0.015, fontsize=9
    )

    plt.xlabel('Global Input Q & K Sparsity (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Operations', fontsize=12, fontweight='bold')
    plt.title('Impact of Sparse Attention Skipper (SAS) on Operation Count', fontsize=14, fontweight='bold')
    plt.xticks(sas_x_indexes, [f'{s}%' for s in sparsities], fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    max_ops = max(total_ops) if total_ops else 1
    plt.ylim(0, max_ops * 1.15)
    
    plt.tight_layout()
    filename = f"{output_dir}/Chart_SSA_Efficiency_SAS_Module.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f" -> Đã lưu: {filename}")
else:
    print(" -> Không tìm thấy cột Total_SSA_wo_SAS/Total_SSA_w_SAS trong CSV.")

# VẼ BIỂU ĐỒ BẤT ĐỒNG BỘ 9 CORES (MAX VS AVG CYCLE) Ở 50% VÀ 90%
if 'Max_Cycle' in df.columns and 'Avg_Cycle' in df.columns:
    sparsities_to_plot = [50, 90]
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('SSA Asynchronous Core Execution: Max vs Avg Cycle', fontsize=16, fontweight='bold', y=0.98)

    async_bar_width = 0.35
    async_x_indexes = np.arange(len(modes))
    color_max = '#d62728'  # Đỏ
    color_avg = '#1f77b4'  # Xanh dương

    for idx, sparsity in enumerate(sparsities_to_plot):
        ax = axes[idx]
        df_sp = df[df['Sparsity(%)'] == sparsity]
        
        max_cycles = []
        avg_cycles = []
        
        for mode in modes:
            row = df_sp[df_sp['Mode'] == mode]
            if not row.empty:
                max_cycles.append(row['Max_Cycle'].values[0])
                avg_cycles.append(row['Avg_Cycle'].values[0])
            else:
                max_cycles.append(0)
                avg_cycles.append(0)
                
        # Vẽ 2 cột
        bar1 = ax.bar(async_x_indexes - async_bar_width/2, max_cycles, width=async_bar_width, 
                      label='Max Cycle', color=color_max, edgecolor='black')
        bar2 = ax.bar(async_x_indexes + async_bar_width/2, avg_cycles, width=async_bar_width, 
                      label='Avg Cycle', color=color_avg, edgecolor='black')
        
        # Tự động thêm dấu phẩy và xoay dọc nếu label quá rộng
        add_value_labels(
            ax, list(bar1) + list(bar2), max_cycles + avg_cycles,
            offset_ratio=0.015, fontsize=9
        )
        
        # Trang trí Subplot
        ax.set_title(f'SSA Global Q, K Sparsity: {sparsity}%', fontsize=13, fontweight='bold')
        ax.set_xticks(async_x_indexes)
        # Bẻ chữ label các mode để không bị đè vào nhau do tên dài
        ax.set_xticklabels(modes, fontsize=10, rotation=0) 
        ax.set_ylabel('Hardware Cycles', fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        if idx == 1:
            ax.legend(fontsize=10, loc='upper right')
        
        max_overall = max(max_cycles) if max_cycles else 0
        ax.set_ylim(0, max_overall * 1.2 if max_overall > 0 else 1)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    filename = f"{output_dir}/Chart_SSA_Async_Max_vs_Avg_Cycle.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f" -> Đã lưu thành công: {filename}")
else:
    print(" -> Bỏ qua phần bất đồng bộ do không tìm thấy cột Max_Cycle/Avg_Cycle trong CSV.")

print(f"\nHoàn tất! Toàn bộ biểu đồ SSA đã được lưu vào thư mục '{output_dir}'.")