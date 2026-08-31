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

# Đọc dữ liệu từ file CSV
csv_file = "metrics_sconv.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Không tìm thấy file {csv_file}.")
    exit()

# Xóa khoảng trắng thừa ở tên cột (nếu có)
df.columns = df.columns.str.strip()

# Các Mode và mức Sparsity
modes = ["No NZ", "NZ Only", "NZ + LB"]
sparsities = sorted(df['Sparsity(%)'].unique())

# Lấy danh sách các thông số cần vẽ (bỏ qua 2 cột Sparsity và Mode, Spike Dram load vẽ chung để thể hiện hiệu năng router)
metrics_to_plot = [col for col in df.columns if col not in ['Sparsity(%)', 'Mode', 'Spike_DRAM_load_wo_Router', 'Spike_DRAM_load_w_Router']]

# Tạo thư mục để lưu ảnh biểu đồ
output_dir = "Sconv_charts_output"
os.makedirs(output_dir, exist_ok=True)

# Thông số của biểu đồ cột
bar_width = 0.25
x_indexes = np.arange(len(sparsities))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Xanh, Cam, Xanh lá cho 3 mode

print("Bắt đầu vẽ biểu đồ...")

# Vẽ và lưu từng biểu đồ cho từng metric
for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    
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

        # Tính toán vị trí x để các cột đứng cạnh nhau
        x_pos = x_indexes + (i - 1) * bar_width
        bars = plt.bar(x_pos, values, width=bar_width, label=mode, color=colors[i], edgecolor='black')
        all_bars.extend(bars)
        all_values.extend(values)

    # Tự động thêm dấu phẩy và xoay dọc nếu label quá rộng
    add_value_labels(plt.gca(), all_bars, all_values, offset_ratio=0.01, fontsize=9)

    # Trang trí biểu đồ
    metric_name = metric.replace('_', ' ') # Xóa dấu gạch dưới để in ra đẹp hơn
    plt.xlabel('Global Input Sparsity (%)', fontsize=12, fontweight='bold')
    plt.ylabel(metric_name, fontsize=12, fontweight='bold')
    plt.title(f'Comparison of {metric_name} across Operating Modes', fontsize=14, fontweight='bold')
    
    # Cài đặt nhãn cho trục x
    plt.xticks(x_indexes, [f'{s}%' for s in sparsities], fontsize=11)
    
    # Thêm lưới, legend và tối ưu không gian
    plt.legend(title="Hardware Mode", fontsize=10, title_fontsize='11')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Mở rộng giới hạn trục Y một chút để không bị lấp mất chữ số trên đỉnh cột
    plt.ylim(0, df[metric].max() * 1.15) 
    
    plt.tight_layout()
    
    # Lưu ảnh ra thư mục
    filename = f"{output_dir}/Chart_{metric}.png"
    plt.savefig(filename, dpi=300) # dpi=300 để ảnh nét căng
    plt.close()
    print(f" -> Đã lưu: {filename}")

# Vẽ riêng biểu đồ Router Efficiency: Spike Dram load w và wo router
if 'Spike_DRAM_load_wo_Router' in df.columns and 'Spike_DRAM_load_w_Router' in df.columns:
    print("\nBắt đầu vẽ biểu đồ hiệu quả của Async Core Mem Router...")
    plt.figure(figsize=(10, 6))
    
    router_bar_width = 0.35
    router_x_indexes = np.arange(len(sparsities))
    
    # CHỈ ĐỌC DATA TỪ MODE 3 (NZ + LB)
    mode3_data = df[df['Mode'] == 'NZ + LB']
    
    wo_router_loads = []
    w_router_loads = []
    
    for sparsity in sparsities:
        row = mode3_data[mode3_data['Sparsity(%)'] == sparsity]
        if not row.empty:
            wo_router_loads.append(row['Spike_DRAM_load_wo_Router'].values[0])
            w_router_loads.append(row['Spike_DRAM_load_w_Router'].values[0])
        else:
            wo_router_loads.append(0)
            w_router_loads.append(0)

    # Vẽ 2 cột so sánh
    bars1 = plt.bar(router_x_indexes - router_bar_width/2, wo_router_loads, width=router_bar_width, 
                    label='Spike DRAM Load (w/o Router)', color='#7f7f7f', edgecolor='black') # Màu xám
    bars2 = plt.bar(router_x_indexes + router_bar_width/2, w_router_loads, width=router_bar_width, 
                    label='Spike DRAM Load (w/ Router)', color='#2ca02c', edgecolor='black')   # Màu xanh lá

    # Tự động thêm dấu phẩy và xoay dọc nếu label quá rộng
    add_value_labels(
        plt.gca(), list(bars1) + list(bars2),
        wo_router_loads + w_router_loads,
        offset_ratio=0.015, fontsize=10
    )

    plt.xlabel('Global Input Sparsity (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Memory Accesses', fontsize=12, fontweight='bold')
    plt.title('Impact of Async Core Mem Router on Spike DRAM Load', fontsize=14, fontweight='bold')
    plt.xticks(router_x_indexes, [f'{s}%' for s in sparsities], fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    max_ops = max(wo_router_loads) if wo_router_loads else 1
    plt.ylim(0, max_ops * 1.15)
    
    plt.tight_layout()
    filename = f"{output_dir}/Chart_Router_Efficiency.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f" -> Đã lưu thành công: {filename}")

# Vẽ riêng biểu đồ Asynchronous cores: so sánh max cycle và avg cycle ở 3 mode với sparsity 50%, 90%
if 'Max_Cycle' in df.columns and 'Avg_Cycle' in df.columns:
    sparsities_to_plot = [50, 90]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Asynchronous Core Execution: Max vs Avg Cycle of All Cores', fontsize=16, fontweight='bold', y=0.98)

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
        bar1 = ax.bar(async_x_indexes - async_bar_width/2, max_cycles, width=async_bar_width, label='Max Cycle', color=color_max, edgecolor='black')
        bar2 = ax.bar(async_x_indexes + async_bar_width/2, avg_cycles, width=async_bar_width, label='Avg Cycle', color=color_avg, edgecolor='black')
        
        # Tự động thêm dấu phẩy và xoay dọc nếu label quá rộng
        add_value_labels(
            ax, list(bar1) + list(bar2), max_cycles + avg_cycles,
            offset_ratio=0.015, fontsize=10
        )
        
        # Trang trí Subplot
        ax.set_title(f'Global Input Sparsity: {sparsity}%', fontsize=13, fontweight='bold')
        ax.set_xticks(async_x_indexes)
        ax.set_xticklabels(modes, fontsize=12)
        ax.set_ylabel('Hardware Cycles', fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        if idx == 1: # Đặt legend ở hình bên phải
            ax.legend(fontsize=10, loc='upper right')
        
        max_overall = max(max_cycles) if max_cycles else 0
        ax.set_ylim(0, max_overall * 1.15 if max_overall > 0 else 1)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88) # Tránh đè lên Suptitle

    filename = f"{output_dir}/Chart_Async_Max_vs_Avg_Cycle.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f" -> Đã lưu thành công: {filename}")

print(f"\nHoàn tất! Các biểu đồ đã được lưu vào thư mục '{output_dir}'.")