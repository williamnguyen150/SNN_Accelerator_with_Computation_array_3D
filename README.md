# Phân Tích Chi Tiết Cơ Chế Hoạt Động — SNN_Accelerator.h

> Tài liệu tổng hợp toàn bộ phân tích về kiến trúc, cơ chế đánh địa chỉ, luồng xử lý, hệ thống phân cấp bộ nhớ và các tối ưu hoá của mô hình phần cứng mô phỏng SNN Accelerator cho mạng Spiking Transformers.

--------
## Mục lục
1.  [Hằng số & Tham số toàn cục](#1-hằng-số--tham-số-toàn-cục)
2.  [SpikeRF](#2-spikerf--spike-register-file--ping-pong-buffering)
3.  [V_reg & WeightVector](#3-v_reg--weightvector--cấu-trúc-dữ-liệu-cơ-bản)
4.  [MultiBankMemory](#4-multibankmemory--sram-đa-băng)
5.  [ComputationArray3D](#5-computationarray3d--mảng-cây-cộng-có-mặt-nạ-tính-toán-wa-3d)
6.  [NonZeroDataFetcher](#6-nonzerodatafetcher--bộ-kéo-dữ-liệu-khác-không)
7.  [AsyncCoreMemRouter](#7-asynccorememrouter--bộ-định-tuyến-spike)
8.  [StatisticalData & Tiện ích File IO](#8-statisticaldata--tiện-ích-file-io)
9.  [Tóm Tắt Các Kiến Trúc Hardware Accelerator](#9-tóm-tắt-các-kiến-trúc-hardware-accelerator)
10. [Multi-Mode Unified Scheduler](#10-multi-mode-unified-scheduler---bộ-lập-lịch-tính-toán-đa-chế-độ)
--------

## 1. Hằng số & Tham số Toàn Cục

```cpp
constexpr int A_D = 8;  // Chiều sâu: 8 Timesteps
constexpr int A_H = 8;  // Chiều cao: 8 Output Channels (1 Vector chứa 8 số)
constexpr int A_W = 8;  // Chiều rộng: 8 Input Channels (8 Bank SRAM)
constexpr int max_WeightVector_per_bank = 256;  // Mỗi bank chứa tối đa 256 WeightVector
constexpr int max_ic_per_spike_rf = 512;        // Mỗi Spike RF chứa tối đa 512 ic
constexpr int N = 4;    // Cửa số tìm kiếm của Load Balancer
constexpr int Num_Cores = 9;
```

| Tham số | Giá trị | Ý nghĩa & Mapping Phần Cứng |
|---------|---------|-----------------------------|
| `A_D` | 8 | Số timestep. Đây là chiều thời gian, cho phép tái sử dụng trọng số nhiều lần trong cùng 1 cycle |
| `A_H` | 8 | Số output channel tích luỹ song song (Fan-out). Quy định kích thước của 1 `WeightVector` |
| `A_W` | 8 | Số input channel xử lý đồng thời = số lượng SRAM Bank (Fan-in) |
| `max_WeightVector_per_bank` | 256 | Sức chứa tối đa của 1 bank SRAM trọng số. Đủ chứa toàn bộ N_in của SpikingFormer 8-768 |
| `max_ic_per_spike_rf` | 512 | Sức chứa Input Channel tối đa của SpikeRF trên một mặt Ping hoặc Pong |
| `N` | 4 | Cửa sổ tìm kiếm của bộ Load Balancer (Quét 4 bank lân cận để mượn việc) |
| `Num_Cores` | 9 | Cụm 9 lõi xử lý song song, tối ưu mapping Kernel Convolution 3x3 hoặc chia thành 3 cụm (Clusters) để xử lý Q-K-V |

---

## 2. SpikeRF — Spike Register File & Ping-Pong Buffering

**Chức năng:** Khối lưu trữ cục bộ các spike đầu vào cần thiết cho tính toán. Đây là **bộ nhớ cận tính toán (near-compute memory)** gắn liền với mỗi core, giúp giảm tối đa băng thông (bandwidth) truy cập DRAM.

```cpp
class SpikeRF {
    // Vị trí của spike trong IFM mà core đang lưu
    int h_in = -1;
    int w_in = -1;

    // Nếu N_in > max_ic_per_spike_rf thì cần tile
    // Với kích thước model SpikingFormer 8-768 thì chỉ có QKV cần tile
    int ic_tile_start = -1;

    // 2 SpikeRF ping-pong, mỗi RF chứa 512 ic qua A_D timestep
    char local_spike[2][max_ic_per_spike_rf][A_D] = {0};
    ...
};
```

### 2.1. Cơ chế Ping-Pong Buffering (Chiều [2])
Mảng local_spike được thiết kế dưới dạng 3 chiều, trong đó chiều đầu tiên là [2] đại diện cho biến pp (Ping-Pong).
Cơ chế Double-Buffering (Bộ đệm kép) này hoạt động như sau:
- Khối AC 3D thực hiện tính toán Tích lũy Trọng số (Weight Accumulation - WA) trên dữ liệu của bộ đệm Ping (pp = 0).
- Cùng lúc đó, khối Router chạy ngầm, nạp sẵn (prefetch) dữ liệu của toạ độ/token tiếp theo từ DRAM vào bộ đệm Pong (pp = 1).
- Khi AC tính xong, hệ thống chỉ việc lật cờ (pp = !pp). Việc này giúp che giấu hoàn toàn độ trễ (Hide Latency) của giao tiếp bộ nhớ ngoài.

### 2.2. Kích thước sử dụng thực tế (Memory Footprint)
- Mỗi core sở hữu 1 SpikeRF với dung lượng vật lý (trên SRAM):
- Size = 2 (Ping-Pong) * 512 (ic) * 8 (timesteps) * 1 bit = 1 KB.
- Với 9 cores, tổng dung lượng SpikeRF toàn chip là 9 KB.

### 2.3. Chiến lược Tiling & Cấp phát (Tham số ic_tile_start)
Ở mạng SpikingFormer 8-768, số kênh đầu vào (C_in) của lớp QKV là 768. Vì 768 > 512 (max_ic_per_spike_rf), bộ nhớ không thể chứa toàn bộ input trong 1 lần nạp. 
Biến 'ic_tile_start' giúp đánh dấu vị trí bắt đầu cắt lớp (Tiling):
- Lần nạp 1: ic_tile_start = 0, load 512 kênh đầu tiên.
- Lần nạp 2: ic_tile_start = 512, load 256 kênh còn lại.

Tuỳ thuộc vào toán tử, cách dữ liệu được nạp vào SpikeRF sẽ khác nhau:

| Toán tử |     Nguồn Load     |           Hàm phụ trách           |
|---------|--------------------|-----------------------------------|
| `SConv` | DRAM / Core kề bên | `load_spike_DRAM_to_SRAM_SConv()` |
    |
    |
    └──> Tái sử dụng dữ liệu. Dùng `borrow_spike_rf_next_core()` để mượn spike từ core hàng xóm khi cửa sổ conv 3x3 trượt đi
|`QKV_Gen`|   DRAM (Direct)    |  `load_spike_DRAM_to_SRAM_QKV()`  |
    |
    |
    └──> Đọc dữ liệu 1D tại 1 tokens. Phải tiling làm 2 đợt do C_in = 768
|  `SSA`  |   DRAM (Direct)    |  `load_spike_DRAM_to_SRAM_SSA()`  |
    |
    |
    └──> Mỗi core tính một hàng thứ i của Q. Module SAS: Nạp từng hàng thứ j của K^T (cỡ N = 196 < 512) nếu global bitmap Q[i][j] theo 8 timesteps = 1. Chuyển sang giá trị j mới thì đảo pp_spike để nạp và vẫn tính trên cùng 1 oc_group (8 oc liền nhau), cho đến khi hết hàng Q[i] thì chuyển sang Q[i+9]. Các cores đảm nhận tính toán từng cụm 9 tokens (9 hàng) trong output Q.

---

## 3. V_reg & WeightVector — Cấu Trúc Dữ Liệu Cơ Bản

Lưu trữ trọng số phục vụ trực tiếp tại các đầu vào của mảng tính toán cây cộng 3D (3D adder trees).
Lưu trữ điện thế màng trong quá trình LIF: Leaky/Rò rỉ điện thế - Integrate/Cộng dồn trọng số - Fire/Phát xung

```cpp
struct V_reg {
    // 2 Register file ping-pong chứa membrane potential của A_H oc qua A_D timesteps
    int RF[2][A_D][A_H] = {0};
};

struct WeightVector {
    int8_t w[A_H] = {0}; // vector chứa 8 weight đơn lẻ ứng với 8 oc theo chiều dọc
};
```

### 3.1. Thanh ghi Điện thế màng (V_reg)
- Chức năng: Lưu giữ tổng thành phần (Partial Sums - Psum) của các điện thế màng neuron trước khi xuất qua hàm kích hoạt LIF.
- Cấu trúc 3 chiều ([pp][Timesteps][Output_Channels]): `V_reg` trải dài theo chiều A_D để lưu 8 trạng thái thời gian cùng lúc, và A_H để lưu 8 output channel tính song song.
- Ping-Pong trong V_reg: Tương tự SpikeRF, `V_reg` cũng có `pp`. Cập nhật màng (Membrane Update - MU) tốn thời gian. Nhờ có bộ đệm kép, core có thể vừa tính Tích lũy trọng số (WA) cho khối oc_group sau, vừa tiến hành xử lý tuần tự MU cho khối oc_group trước.
- Dung lượng lưu trữ: 2 (pp) * 8 (A_H weight each vector, also A_H oc each group) * 8 (A_D timesteps) * 4 byte (each Psum value is an int32 number) = 512 bytes.

### 3.2. Vector Trọng số (WeightVector)
- Chức năng: Đơn vị đọc tối thiểu (atomic read) từ bộ nhớ SRAM đa băng (MultiBankMemory).
- Kích thước: Trọng số được lượng tử hóa về 8-bit (`int8_t`). Mỗi struct chiếm đúng 8 bytes, tương ứng với 8 Output Channel dọc theo chiều Fan-out.
- Khi kết hợp cả 8 Bank (A_W), tại một chu kỳ xung nhịp (clock cycle), SRAM có khả năng cung cấp mảng `fetched_vectors[8]` chứa tối đa 8 x 8 = 64 trọng số cho toàn bộ mảng AC 3D.

--- 

## 4. MultiBankMemory — SRAM Đa Băng

Bộ nhớ trọng số (Weight Memory) nội bộ của từng core được thiết kế dưới dạng SRAM đa băng (Multi-bank SRAM) để tránh hiện tượng xung đột truy cập (bank conflict) khi cấp dữ liệu cho mảng 3D.

```cpp
class MultiBankMemory {
private:
    WeightVector banks[A_W][max_WeightVector_per_bank];
    int num_ic_per_bank = 0;
    int num_oc_group_per_bank = 0;
    int weight_load_DRAM_to_SRAM_count = 0;
};
```

### 4.1. Kích thước vật lý và Cấu trúc lưu trữ
- Phần cứng của mỗi core sở hữu 8 bank độc lập (A_W = 8).
- Sức chứa mỗi bank: 256 (WeightVector) * 8 bytes (8 Weight trong 1 WeightVector) = 2 KB.
- Tổng dung lượng Weight SRAM mỗi core: 8 * 2 KB = 16 KB. 
- Sức chứa này được thiết kế có chủ đích để nạp vừa vặn toàn bộ Input Channel (N_in) của bất kỳ layer nào trong mô hình SpikingFormer 8-768 (tối đa 768 kênh), giúp tránh việc phải tải lại trọng số nhiều lần.
- Cấu trúc và địa chỉ:
    banks[bank_idx][local_addr].w[oc]
    │       │           │         │
    │       │           │         └──> output channel offset trong 1 WeightVector (0, 1, ..., 7)
    │       │           └──> địa chỉ cụ thể trong 256 WeightVector của bank
    │       └──> bank index
    └──> mảng 8 bank SRAM

### 4.2. Chiến lược Cấp phát động (Dynamic Tiling)
Thay vì hardcode địa chỉ cố định, hàm `set_num_ic_oc_of_bank()` tính toán linh hoạt cấu trúc lưu trữ dựa trên N_in của toán tử hiện tại. Ưu tiên hàng đầu là tính theo fan-in, tức nạp và gom hết weight theo các ic với cùng 1 oc_group để tính. Nếu không nạp hết được N_out thì Scheduler sẽ điều phối tiling N_out.

```cpp
void set_num_ic_oc_of_bank(int N_in) {
    num_ic_per_bank = (N_in + A_W - 1) / A_W;
    num_oc_group_per_bank = max_WeightVector_per_bank / num_ic_per_bank;
}
```

Cấu trúc lưu trữ Weight từ DRAM phân bổ vào 8 Bank SRAM được minh họa qua ví dụ N_in = 128, N_out = 128. Mỗi ô trong sơ đồ là 1 WeightVector chứa 8 giá trị int8_t (tương ứng 8 Output Channels).

```
|             | Bank 0                           | Bank 1                           | ... | Bank 7                             |
|-------------|----------------------------------|----------------------------------|-----|------------------------------------|
| Channel     | ic0      | ic8      |...| ic120  | ic1      | ic9      |...| ic121  | ... | ic7      | ic15     |...| ic127    |
|-------------|----------|----------|---|--------|----------|----------|---|--------|-----|----------|----------|---|----------|
| Oc_group0   | W0,0     | W0,8     |...| W0,120 | W0,1     | W0,9     |...| W0,121 | ... | W0,7     | W0,15    |...| W0,127   |
| (oc0-oc7)   | ...      | ...      |...| ...    | ...      | ...      |...| ...    | ... | ...      | ...      |...| ...      |
|             | W7,0     | W7,8     |...| W7,120 | W7,1     | W7,9     |...| W7,121 | ... | W7,7     | W7,15    |...| W7,127   |
|-------------|----------|----------|---|--------|----------|----------|---|--------|-----|----------|----------|---|----------|
| Oc_group1   | W8,0     | W8,8     |...| W8,120 | W8,1     | W8,9     |...| W8,121 | ... | W8,7     | W8,15    |...| W8,127   |
| (oc8-oc15)  | ...      | ...      |...| ...    | ...      | ...      |...| ...    | ... | ...      | ...      |...| ...      |
|             | W15,0    | W15,8    |...| W15,120| W15,1    | W15,9    |...| W15,121| ... | W15,7    | W15,15   |...| W15,127  |
|-------------|----------|----------|---|--------|----------|----------|---|--------|-----|----------|----------|---|----------|
| ...         | ...      | ...      |...| ...    | ...      | ...      |...| ...    | ... | ...      | ...      |...| ...      |
|-------------|----------|----------|---|--------|----------|----------|---|--------|-----|----------|----------|---|----------|
| Oc_group15  | W120,0   | W120,8   |...|W120,120| W120,1   | W120,9   |...|W120,121| ... | W120,7   | W120,15  |...| W120,127 |
| (oc120-127) | ...      | ...      |...| ...    | ...      | ...      |...| ...    | ... | ...      | ...      |...| ...      |
|             | W127,0   | W127,8   |...|W127,120| W127,1   | W127,9   |...|W127,121| ... | W127,7   | W127,15  |...| W127,127 |
```

### 4.3. Đa hình Dữ liệu (Polymorphic Data Layout)
Hàm `load_weight_DRAM_to_SRAM ()`hỗ trợ nạp nhiều loại định dạng trọng số khác nhau từ DRAM vào SRAM, tuỳ thuộc vào bản chất toán tử:
- SConv / QKV: Trọng số thực sự (Weights). Layout trên DRAM là [C_out][C_in]. Cắt theo cụm C_in, rải đều vào 8 bank.
- SSA: Ma trận V đã đóng gói (Packed). Layout trên DRAM là [N_tokens][d]. Ma trận V vốn là ma trận Spike, nhưng được nén 8 timesteps vào 1 số int8_t để đưa vào SRAM trọng số.

---

## 5. ComputationArray3D — Mảng Cây Cộng Có Mặt Nạ Tính Toán WA 3D

Đây là trái tim của bộ tăng tốc, thực hiện phép tính Tích luỹ Trọng số (Weight Accumulation - WA) mà không cần bộ nhân (MAC), phản ánh chính xác sức mạnh tiết kiệm năng lượng của mạng SNN.

```cpp
class ComputationArray3D {
private:
    V_reg v_reg;                // Khối thanh ghi Psum nội bộ của mảng 3d
    long long SOP_count = 0;    // Số phép cộng dồn (ứng với 1 weight có xung đơn lẻ)

public:
    void compute_WA_one_cycle(...) {}
};
```

### 5.1. Bản chất 3 Chiều (3D Array Mapping)
Đoạn code sử dụng 3 vòng lặp lồng nhau, map trực tiếp với kiến trúc cây cộng phần cứng (Masked Adder Tree):
1. Chiều Thời gian (t = 0 đến A_D): Xử lý 8 timesteps trong cùng một hàm. Phần cứng tận dụng dữ liệu trọng số vừa fetch ra từ SRAM để tính cho toàn bộ 8 timesteps mà không cần load lại, giảm thiểu hao phí năng lượng (Weight reuse among timesteps).
2. Chiều Fan-in (i = 0 đến A_W): Từ khoá pragma GCC unroll 8 chỉ thị trình biên dịch mở phẳng vòng lặp, mô phỏng 8 cổng vào của cây cộng (Adder Tree) hoạt động đồng thời trong 1 cycle.
3. Chiều Fan-out (oh = 0 đến A_H): Từ khoá pragma omp simd mô phỏng tập lệnh vector. Một trọng số cộng dồn vào nhiều kênh output cùng lúc.

```
         t=0  t=1  t=2  t=3  t=4  t=5  t=6  t=7      WeightVector
         ───  ───  ───  ───  ───  ───  ───  ───      ────────────
Bank 0:  [1]  [0]  [1]  [0]  [0]  [1]  [0]  [0]   ×   w0[8 oc]
Bank 1:  [0]  [0]  [0]  [1]  [0]  [0]  [0]  [0]   ×   w1[8 oc]
Bank 2:  [1]  [1]  [0]  [0]  [1]  [0]  [0]  [0]   ×   w2[8 oc]
  ...
Bank 7:  [0]  [0]  [0]  [0]  [0]  [0]  [0]  [1]   ×   w7[8 oc]
                                                        ↓
                                    V_reg: RF[ping-pong][t = 0:7][oh = 0:7]
```

### 5.2. Mạch cộng có điều kiện (Masked Accumulation)
Câu lệnh `if (bank_busy[i] && spike_mask[i][t])` chính là mấu chốt của SNN:
- Khác với mạng ANN thông thường phải dùng bộ nhân (output += input * weight), SNN nhận đầu vào là chuỗi nhị phân (0 hoặc 1). 
- Phần cứng chỉ cần một bộ MUX (Multiplexer) hoặc cờ logic điều khiển bộ cộng. Nếu spike = 1, cộng thẳng trọng số vào thanh ghi màng (V_reg). Nếu spike = 0, bỏ qua (Zero-skipping).

### 5.3. Mạch giải nén cho SSA (SSA Unpacking Logic)

```cpp
is_SSA ? (fetched_vectors[i].w[oh] >> t) & 1 : fetched_vectors[i].w[oh]
```

Trong toán tử Self-Attention (Q * K_T * V), ma trận V bản chất là các chuỗi spike (0 hoặc 1) trải dài qua 8 timesteps. 
- Thay vì lưu V như một bộ spike khổng lồ tốn diện tích, Scheduler đã đóng gói (pack) 8 timesteps này thành 1 số int8_t (mỗi bit ứng với 1 timestep) và nạp vào MultiBankMemory dưới danh nghĩa Weight chính thức, được reuse theo T.
- Khi mảng 3D tính toán tại timestep t, nó dùng phép dịch bit (>> t) và AND với 1 (& 1) để trích xuất ra đúng bit (spike) của timestep đó.

---

## 6. NonZeroDataFetcher — Bộ Kéo Dữ Liệu Khác Không

Chức năng: "bộ não" điều phối của mỗi core, chịu trách nhiệm:
- khai thác tính thưa thớt (sparsity) của mạng SNN
- Cân bằng tải (load balancing) giữa các bank SRAM

```cpp
class NonZeroDataFetcher {
private:
    int cycle_count = 0;
    long long weight_load_SRAM_to_array_count = 0;
};
```

### 6.1. Quá trình tạo và tách Bitmap (Nhóm 64 Kênh/Tokens)
Bộ kéo dữ liệu sử dụng định dạng bitmap 2 giai đoạn (`Two-stage bitmap dataflow`) để xác định vị trí các spike khác không.
Do mỗi core tính toán 1 pixel riêng biệt (trong SConv) hoặc 1 cụm Token riêng (trong QKV/SSA), spike_vector trong DRAM được xem như có layout [N_in][T].
Bộ Fetcher quét dọc theo N_in (tức C_in hoặc N_tokens):
1. `generate_64_global_bitmap()`: Input lấy từ Spike RF thay vì chọc vào spike mem chung của 9 cores. Quét một cụm tối đa 64 kênh. Chỉ tạo bitmap cho các ic thật sự còn lại, chưa chắc là full 64 ic.
2. `split_global_bitmap_to_banks()`: Tách 64-bit global bitmaps thành 8 cụm 8 global bitmaps (gọi là `bank_bitmaps`), phân bổ về cho 8 bank. Ví dụ với 64 kênh đầu tiên:
    Bank 0: [ic0=1, ic8=0, ic16=1, ic24=0, ic32=0, ic40=1, ic48=0, ic56=0]  → 3 việc
    Bank 1: [ic1=0, ic9=0, ic17=0, ic25=1, ic33=0, ic41=0, ic49=0, ic57=0]  → 1 việc
    Bank 2: [ic2=1, ic10=1, ic18=0, ic26=0, ic34=1, ic42=0, ic50=1, ic58=0] → 4 việc
    Bank 3: [ic3=0, ic11=0, ic19=0, ic27=0, ic35=0, ic43=0, ic51=0, ic59=0] → 0 việc ← RẢNH
    Bank 4: [ic4=1, ic12=0, ic20=0, ic28=0, ic36=0, ic44=0, ic52=0, ic60=0] → 1 việc
    Bank 5: [ic5=0, ic13=0, ic21=0, ic29=0, ic37=0, ic45=0, ic53=0, ic61=1] → 1 việc
    Bank 6: [ic6=0, ic14=1, ic22=1, ic30=1, ic38=1, ic46=0, ic54=0, ic62=0] → 4 việc
    Bank 7: [ic7=0, ic15=0, ic23=0, ic31=0, ic39=0, ic47=0, ic55=0, ic63=0] → 0 việc ← RẢNH
3. `address_generator()`: Bộ sinh địa chỉ. Lặp đến khi bit thứ i từ phải sang bằng 1 thì reset nó về 0 và lưu lại địa chỉ -> vị trí ic tương ứng có global bit = 1. Cơ chế này mô phỏng chính xác mạch mã hóa ưu tiên (`Priority Encoder`) trên phần cứng.

### 6.2. Cân bằng tải (Load Balancer)
Dễ thấy xảy ra tình trạng bank có nhiều dữ liệu thì quá tải, bank không có dữ liệu lại ngồi chờ, dẫn đến trong 1 chu kỳ chỉ có ít cổng của adder tree thật sự tham gia tính toán.
Bộ cân bằng tải can thiệp bằng cách cho phép "vay mượn" công việc:
- Quét cửa sổ tìm kiếm lân cận (N banks). Bank nào còn global bit = 1 (chuỗi bitmap != 0) thì đăng ký mượn.
- Chốt quyền mượn theo thứ tự ưu tiên (khi nhiều bank cùng mượn 1 bank thì bank chỉ số nhỏ hơn được mượn trước).
- Phải kiểm tra lại xem neighbor còn data không, phòng trường hợp bank này đã bị idle_bank có ưu tiên cao hơn mượn rồi.

### 6.3. `load_WeightVector_SRAM_to_array` — Nạp Weight Vào Mảng Tính Toán
Tổng quát cho 2 trường hợp, hoặc load từ bank SRAM của chính mình, hoặc mượn từ bank hàng xóm để đưa vào cổng adder tree của bank mình
```cpp
void load_WeightVector_SRAM_to_array(
    int bank_idx,   // chỉ số bank cần lấy weight
    int borrow_idx, // chỉ số bank đi mượn: nếu tự load mà không mượn thì borrow_idx = bank_idx
    ...)
```

### 6.4. Ba chế độ hoạt động (fetch_data_and_compute)
Hàm điều phối chính `fetch_data_and_compute()` có thể hoạt động dưới 3 Mode tùy thuộc vào các cờ ena_nz và ena_lb:
- **MODE 1 (NO NZ, NO LB)**: Tính toán tuần tự từng ic, không dùng đến `Two Stage Bitmap`. Vòng lặp chạy cố định, không kiểm tra độ thưa thớt để fetch mà chỉ kiểm tra ở cổng MUX để đưa vào adder tree.
    ```
    Đếm số input channel còn lại: channels_to_scan = min(64, N_in - global_ic_start)
    Chia channels_to_scan thành các block, mỗi block tối đa A_W ic
    for each block:
        ① Tất cả 8 bank đọc weight theo từng ic lần lượt (0->7, 8->15, ...)
        ② Tạo spike_mask từ SpikeRF
        ③ cycle_count++
        ④ arr3d.compute_WA_one_cycle(weights, busy = true, spike_mask, pp_v_reg)
    ```
- **MODE 2 (NZ)** và **MODE 3 (NZ + LB)**: Chạy vòng lặp while đến khi tất cả global bitmap được reset về 0 (tức biến all_empty = 0) thì kết thúc. Ở Mode 3, bộ load_balancer được kích hoạt để tối ưu số chu kỳ tính toán.
    ```
    while (còn bank nào có chuỗi 8 global bitmaps ≠ 0):
        ① Mỗi bank tự pop 1 việc từ bank_bitmaps bởi address_generator()
        ② Bank rảnh → load_balancer() vay từ neighbor nếu bật cờ ena_lb -> bank busy
        ③ Tạo spike_mask từ SpikeRF cho các bank busy
        ④ cycle_count++
        ⑤ arr3d.compute_WA_one_cycle(weights, busy, spike_mask, pp_v_reg)
    ```
- Thông số thống kê `average_num_WeightVector_load_per_cycle` ở Mode 1 > Mode 3 > Mode 2:
    + Mode 1: tất cả Weight được fetch dù không đóng góp vào output (spike = 0) nên giá trị này bằng A_H nếu C_in chia hết cho A_H và nhỏ hơn A_H một chút nếu không chia hết
    + Mode 2, 3: chỉ Weight nào thật sự đóng góp vào output mới được fetch nên giá trị này nhỏ hơn ở Mode 1 khá nhiều, đồng thời phụ thuộc độ thưa của input
    + Mode 3 có balancer nên load được nhiều Weight trong 1 cycle hơn Mode 2, do đó giá trị ở Mode 2 là bé nhất
-> Phản ánh độ thưa của input (Mode 2, 3 so với Mode 1) và hiệu năng phần cứng (Mode 3 so với Mode 2)

### 6.5. Tích hợp Sparse Attention Skipper (SAS)
Trong hàm `fetch_data_and_compute()`, mặt nạ kích hoạt (spike_mask) được phân nhánh linh hoạt theo cờ is_SSA:
- Với **Sconv, QKV_gen**: mask đơn giản là local bitmap tức input spikes của từng ic theo từng timesteps.
- Với **SSA**: mask = local bitmap của K_T AND Q. Tức là AND giữa local_spike_1 (K_T) và local_spike_2 (Q).
-> Ý nghĩa phần cứng: Phép toán AND nhị phân này mô phỏng module SAS. Cây cộng WA 3D chỉ hoạt động khi cả bitmap Q và K chuyển vị đều có xung, giúp giảm thiểu rất nhiều số phép toán SOP.

---

## 7. AsyncCoreMemRouter — Bộ Định Tuyến Spike

Chức năng: Quản lý và định tuyến luồng dữ liệu Spike từ DRAM vào SpikeRF của từng core. Hoạt động trong SConv Scheduler để tái sử dụng dữ liệu cho phép tích chập, giúp giảm số lần truy cập bộ nhớ ngoài Spike_Mem DRAM.

```cpp
class AsyncCoreMemRouter {
private:
    int router_shared_count = 0;
public:
    void get_spikes_with_router(...) {
    }
};
```

### 7.1. Cơ chế mượn dữ liệu nội bộ (Core-to-Core)
Thay vì cả 9 cores đều phải chọc ra DRAM để đọc dữ liệu, Router cho phép các cores chia sẻ dữ liệu với nhau:
- Quét xem core lân cận nào có dữ liệu tọa độ này:
    ```cpp
    for (int i = 0; i < 9; i++) {
        if (spike_rf[i].get_prev_h_in() == h_in
            && spike_rf[i].get_prev_w_in() == w_in
            && spike_rf[i].get_prev_ic_tile_start() == c_in_start)
        { found = true; found_id = i; break; }
    }
    ```
-> Phải làm vậy do ở vị trí biên hoặc ở ic_tile_start mới thì các core không mượn được.
-> Với model SpikeFormer 8-768 thì C_in max trong Sconv <= 512 nên không cần tile C_in, chỉ khi ở vị trí biên của OFM thì mới không mượn được.
- Sau khi quét tìm xong mới được gán `h_in, w_in, c_in_start` mới. Nếu gán trước sẽ mất tọa độ cũ, dẫn đến việc nhận diện sai khi chia sẻ dữ liệu.
- Với các vị trí còn lại không ở biên thì:
    + stride = 1: core 0 mượn 1, 1 mượn 2, 2 tự load, 3 mượn 4, 4 mượn 5, ...
    + stride = 2: core 0 mượn 2, 1 tự load, 2 tự load, 3 mượn 5, 4 tự load, ...
    + Đây là vị trí mượn đương nhiên, nhưng không gán sẵn như vậy mà vẫn thông qua vòng lặp quét mượn phía trên
- Nếu tìm được spike hợp lệ TỪ CORE KHÁC thì đi mượn (`borrow_spike_rf_next_core()` để copy từ core hàng xóm).
- Không found thì tự load từ spike mem bên ngoài DRAM.

### 7.2. Xử lý dữ liệu khi chuyển OC Group (Output Channel Group)
Trong quá trình tính toán, một vị trí của Input Feature Map (h_in, w_in) dùng chung cho tất cả oc_groups đã nạp lên Multi-bank WMEM, ứng với một vị trí tương ứng của Output Feature Map (h_out, w_out). Router xử lý trường hợp này cực kỳ khéo léo:
- Không được reset Spike_RF vì khi chuyển oc_group mà vẫn đang ở 1 vị trí của FM thì Spike_RF phải giữ nguyên để tính tiếp. Việc reset chỉ được thực hiện trong Sconv Scheduler.
- Vòng lặp quét mượn vẫn chạy lại khi gọi `get_spikes_with_router()` -> tìm được spike TỪ CHÍNH MÌNH: Khi đó Spike_RF giữ nguyên, không cần đi mượn hoặc load lại từ DRAM. Nhờ vậy, năng lượng tiêu thụ cho việc dịch chuyển dữ liệu được tối ưu đến mức tối thiểu.

---

## 8. StatisticalData & Tiện ích File IO

Chức năng: Dữ liệu thống kê các thông số hoạt động của Accelerator để phản ánh hiệu suất và thời gian thực thi (Latency). Đồng thời, các tiện ích xuất file giúp lưu output để kiểm tra tính chính xác (compare with golden output), cũng như để lưu làm input cho layer tiếp theo xử lý.

### 8.1. StatisticalData - Chỉ số Hiệu năng (Hardware Metrics)
`Struct StatisticalData` thu thập các số liệu quan trọng nhất để đánh giá kiến trúc:
- SOP_count (Synaptic Operations): Tổng số phép cộng thực thụ trên mảng 3D.
- weight_load_DRAM_to_SRAM_count: Số lần các core phải truy cập DRAM để nạp Weight vào Multi-bank SRAM.
- weight_load_SRAM_to_array_count: Số lần đưa một WeightVector vào các cổng của adder tree ở một core.
- cycle_count: Đại diện cho độ trễ (Latency) của hệ thống. Do 9 cores hoạt động bất đồng bộ (Async), tổng thời gian hoàn thành của hệ thống sẽ bằng số cycle của core chạy chậm nhất (hàm max).
- average_cycle_all_core: Giá trị trung bình về Latency của 9 cores, dùng để làm nổi bật tính song song bất đồng bộ (Async) giữa các cores khi so sánh với cycle_count (chắc chắn nhỏ hơn cycle_count).
- average_num_WeightVector_load_per_cycle: Số WeightVector trung bình được load trong mỗi cycle. Dùng để phản ánh hiệu năng của NZ Fetcher và Balancer, cũng như độ thưa input như phân tích ở **mục 6.4**.
- spike_load_DRAM_to_SRAM_count: Số lần các core phải truy cập DRAM để lấy Input spike.
- router_shared_count: Số lần mượn Input spike nội bộ giữa các cores. Chỉ số này càng cao chứng tỏ AsyncCoreMemRouter hoạt động càng hiệu quả, giúp giảm tải băng thông DRAM.
- SSA_skip_by_SAS_count: Số lượng phép toán K^T[j] * V bị module SAS lược bỏ khi tính Spiking Self-Attention nhờ module SAS.

### 8.2. Tiện ích File IO
Bao gồm các hàm load input & weight from file và save output to file.

---

## 9. Tóm Tắt Các Kiến Trúc Hardware Accelerator

| Kiến trúc              | Chức năng chính                                                                   |
|------------------------|-----------------------------------------------------------------------------------|
| **SpikeRF**            | Bộ nhớ cục bộ lưu input spike cho 1 output position                               |
| **V_reg**              | Các thanh ghi tích lũy trọng số cho A_H output channels và A_D timesteps          |
| **MultiBankMemory**    | A_W bank SRAM, mỗi bank = 1 input channel                                         |
| **ComputationArray3D** | Mảng cây cộng có mặt nạ 3D (8t × 8ic × 8oc = 512 ops/cycle)                       |
| **NonZeroDataFetcher** | Bộ kéo dữ liệu khác 0 khai thác sparsity, cân bằng tải, tối thiểu lượng tính toán |
| **AsyncCoreMemRouter** | Bộ định tuyến dữ liệu Spike cho SConv                                             |

---

## 10. Multi-Mode Unified Scheduler - Bộ Lập Lịch Tính Toán Đa Chế Độ

Chức năng: Lớp bao bọc ngoài cùng (Orchestrator) chịu trách nhiệm ánh xạ (mapping) các vòng lặp của vào kiến trúc phần cứng. Nó chia nhỏ dữ liệu (tiling), quản lý việc nạp/xuất với DRAM và điều khiển nhịp điệu của các cờ Ping-Pong (pp_v_reg, pp_spike) để mô phỏng che giấu độ trễ.

### 10.1. Sconv Scheduler

- Đặc thù phần cứng của **SConv**: Khai thác tối đa 9 Cores để xử lý song song không gian (Spatial) của cửa sổ tích chập (Kernel). 
- Với Kernel 3x3, phân vai cố định: Core 0 xử lý vị trí (kh=0, kw=0), Core 1 xử lý (kh=0, kw=1) ... Core 8 xử lý (kh=2, kw=2).
- Tại cùng một tọa độ OFM, các cores cùng lúc quét 9 pixel đầu vào (được Router cấp phát/cho mượn linh hoạt) và cộng dồn kết quả vào `main_V_reg` để ra giá trị WA cuối cùng.
- Data Layout:
            INPUT         *          WEIGHT       =          OUTPUT
    [C_in][H_in][W_in][T] * [C_out][C_in][K1][K2] = [C_out][H_out][W_out][T]

Data-flow chi tiết:

```
0. Tính toán sức chứa của MultiBankMemory, xác định số lượng oc_groups có thể nạp vào SRAM trong 1 lần (num_oc_tile).
0. for mỗi cụm oc_groups tile (`oc_tile_start` = 0 -> C_out):

    1. Nạp Trọng số (Weight Load):
       - 9 Cores tách trọng số từ DRAM ứng với vị trí pixel (kh, kw) của nó
       - Nạp vào MultiBank SRAM dưới layout [C_out][C_in]

    2. for mỗi tọa độ của OFM (h_out, w_out):
        for mỗi `local_oc_group` nằm trong cụm SRAM đã nạp (`local_oc_group` = 0 -> `num_oc_groups`):
            
            2.0. Reset pp_v_reg hiện tại ở từng cores

            2.1. for mỗi cụm C_in tile (nếu C_in > 512 thì phải cắt lớp nạp nhiều lần vào cùng 1 pp_spike):
                - Cấp phát Spike (Router Fetching):
                    + Từng Core tự tính tọa độ (h_in, w_in) của mình dựa trên hệ số stride, padding
                    + Nếu tọa độ hợp lệ: Gọi mem_router để đi mượn core kề bên hoặc load từ DRAM nạp vào SpikeRF
                    + Input layout: [C_in][T]
                    + Nếu tọa độ rơi vào padding: Reset SpikeRF về 0

                - for mỗi block 64 kênh đầu vào (từ SpikeRF đưa xuống Fetcher): tính WA
                    + Sinh 64-bit global_bitmaps từ SpikeRF
                    + Gọi core_fetchers.fetch_data_and_compute() để mảng 3D tính WA
            
            2.2. Gom P_sum và MU:
                - for mỗi oc thật sự trong local_d_group:
                    + Reset pp_v_reg ở thanh ghi tổng main_V_reg
                    + Gom toàn bộ V_reg của 9 Cores cộng dồn vào main_V_reg (Integrate)
                    + Tính rò màng (Leaky): v_mem = (v_mem >> 1) + main_V_reg + bias[oc]
                    + Nếu v_mem >= V_th: Phát xung Output Spike = 1 (Fire) và trừ đi V_th (Soft reset)
                    + Lưu v_mem và Output Spike ra DRAM theo layout [C_out][H_out][W_out][T]

            2.3. !pp_v_reg: Đã tính xong cho local_oc_group hiện tại. Lật cờ để local_oc_group tiếp theo tính WA trong khi group trước đó tính MU ở pp cũ. Code phần mềm không mô phỏng được sự song song này khi chạy, chỉ mô tả qua việc đảo cờ pp.
        
        // Xong tất cả oc_groups

        !pp_spike: Đã tính xong toàn bộ num_oc_groups trong 1 lần tile cho tọa độ (h_out, w_out) hiện tại. Cửa sổ Conv chuẩn bị trượt sang pixel tiếp theo. Lật cờ để Router nạp spike của tọa độ mới vào thanh ghi mới.
    
    // Xong tất cả tọa độ (h_out, w_out) trên 2 chiều (H_out, W_out) của OFM, ở tất cả oc_groups của lần tile hiện tại

// Xong toàn bộ OFM
```

### 10.2. S_QKV Scheduler

- Đặc thù phần cứng của **S_QKV_Gen**: Tạo ra 3 ma trận Q, K, V. Chia 9 cores thành 3 cụm (Clusters) để xử lý song song.
- Phân vai cố định: Cluster 0 (Core 0, 1, 2), Cluster 1 (Core 3, 4, 5), Cluster 2 (Core 6, 7, 8). `Mỗi cụm xử lý 1 Token` (1 điểm ảnh đã flatten). Trong mỗi cụm, Core 0 tính Q, Core 1 tính K, Core 2 tính V (`role = core_id % 3`).
- Cứ 3 tokens liền nhau thì chia đều cho 3 clusters. Áp dụng `Bidirectional Compressor` (bộ nén 2 chiều) ngay trong lúc ghi output để chuyển vị ma trận K thành K^T mà không tốn cycle.
- Data Layout:
                        INPUT                    *     WEIGHT    =                 OUTPUT
    [C_in][H_in][W_in][T] -> [C_in][N_tokens][T] * [C_out][C_in] = [Num_heads][d_heads][N_tokens][T] (K^T)
                                                                   [Num_heads][N_tokens][d_heads][T] (Q, V)
Trong đó C_out được tách thành [Num_heads][d_heads]: multi-head Self Attention, mỗi head chứa d_heads kênh output

Data-flow chi tiết:

```
0. Tính toán sức chứa của MultiBankMemory, xác định số lượng oc_groups có thể nạp vào SRAM trong 1 lần (num_oc_per_tile).
0. for mỗi cụm oc_groups tile (oc_tile_start = 0 -> C_out):

    1. Nạp Trọng số (Weight Load):
       - 9 Cores nạp trọng số Q, K hoặc V từ DRAM tùy thuộc vào vai trò (role) của core
       - Nạp Weight vào MultiBank SRAM dưới layout [C_out][C_in]

    2. for mỗi Token xử lý song song trên 9 cores (n = cluster_id -> N_tokens, bước nhảy = 3):
        
        2.1. Nạp Spike (Load Input Tokens):
           - pp_spike = 0 -> khởi tạo mặc định ở mỗi token
           - for mỗi cụm C_in tile (C_in = 768 > 512 nên phải dùng cả 2 thanh pp để lưu đủ):
               + Gọi spike_rf.load_spike_DRAM_to_SRAM_QKV() nạp 1 Token từ DRAM vào SpikeRF (không thể mượn từ core kề bên)
               + !pp_spike

        2.2. for mỗi local_oc_group nằm trong cụm tile đã nạp lên SRAM:
           - pp_spike = 0 -> khởi tạo lại để bắt đầu tính
           - Reset pp_v_reg hiện tại ở core

           2.2.1. for mỗi cụm C_in tile (Đọc dữ liệu đã nạp để tính toán):
               - for mỗi block 64 kênh đầu vào: tính WA
                   + Sinh 64-bit global_bitmaps từ SpikeRF
                   + Gọi core_fetchers.fetch_data_and_compute() để mảng 3D tính WA
               - !pp_spike: Đảo thanh ghi SpikeRF để đọc cụm C_in tile tiếp theo

           2.2.2. Gom P_sum, MU và Bidirectional Compressor:
               - for mỗi oc thật sự trong local_oc_group:
                   + Tìm xem output hiện tại là Q/K/V và ở head nào để lưu đúng
                   + Lấy psum trực tiếp từ core_array3d (vì mỗi core tính riêng Q, K, V, không cần gom 9 cores như SConv)
                   + Tính rò rỉ (Leaky): V_mem = (V_mem >> 1) + psum + bias[oc]
                   + Nếu V_mem >= V_th: Phát xung (Output Spike = 1) và trừ đi V_th (Soft reset)
                   + Bidirectional Compressor: 
                     * Nếu role == 1 (K): Lưu Output Spike ra DRAM theo layout Nén Dọc [d_heads][N][T] để tự động tạo K^T
                     * Nếu role == 0, 2 (Q, V): Lưu ra DRAM theo layout Nén Ngang [N][d_heads][T]

           2.2.3. !pp_v_reg: Đã tính xong cho local_oc_group hiện tại. Lật cờ để tính WA cho group tiếp theo trong khi group cũ làm MU.
           // Xong 1 oc_group trên 1 token
        
        // Xong tất cả oc_groups của 1 Token
    
    // Xong tất cả Tokens của lần tile hiện tại
    
// Xong toàn bộ OFM
```

### 10.3. SSA Scheduler

- Đặc thù phần cứng của SSA: Tính Q * K^T * V. Ma trận V (theo 8 timesteps) được đóng gói (packed) thành int8 nạp vào MultiBank SRAM dưới vai trò Trọng số (Weight). Ma trận K^T đóng vai Input spikes nạp vào SpikeRF.
- Phân vai: 9 Cores tính toán song song các hàng (tokens) liền nhau của ma trận Q tại một head, mỗi lần 1 hàng thứ i. Xong toàn bộ thì chuyển head và tính tương tự.
- `Sparse Attention Skipper` (SAS): Ma trận Q đóng vai trò bộ lọc tầng ngoài. Core quét Q[i][j] với j = 0 -> d, i cố định, nếu bằng 0 trên tất cả timesteps (global bitmap Q = 0) thì bỏ qua việc nạp và tính K^T[j] * V, giúp giảm lượng lớn phép tính SSA.
=> `Q global bitmap kiểm soát nạp K^T lên SpikeRF`
=> `Q local bitmap AND K^T local bitmap (input spikes) làm mask chính thức` cho adder tree, kiểm soát nạp weight từ SRAM lên array3d
- Data Layout:
    SAS (Sparse-Attention-Skipper)  *      INPUT (K^T)     *       WEIGHT (V)     =        OUTPUT
         [Num_heads][N][d][T]       * [Num_heads][d][N][T] * [Num_heads][N][d][T] = [Num_heads][N][d][T]
                                                                                        |       |  |
                                                                                        |_______|__|
                                                                                            |   |
                                                                                         ___|  _|__
                                                                                        |     |    |
                                                                                        v     V    V
                                                                                    [C_out][H_out][W_out][T]
Trong đó tính SSA trên từng head rồi gộp 2 chiều [Num_heads][d] -> [C_out] ở final output
[N] đóng vai trò input channel (N_in), tách thành [H_out][W_out] ở final output
[d] đóng vai trò output channel (N_out) trong từng head

Data-flow chi tiết:

```
0. Tính toán sức chứa của MultiBankMemory (dựa trên N_tokens hay N_in, ngắn gọn là N).
0. for mỗi head (head_idx = 0 -> num_heads):
    
    1. for mỗi cụm d_groups tile (oc_tile_start = 0 -> d, do d khá lớn tương đương C_out):

        1.1. Nạp Trọng số (Packed V Load):
           - Đóng gói spikes của 8 timesteps tại V[n][oc] thành 1 số int8 (dịch trái t lần).
           - 9 Cores nạp V vào MultiBank SRAM dưới layout [N_tokens][d] tương đương [N_in][N_out].

        1.2. for mỗi Token xử lý song song trên 9 cores (n_tokens = core_id -> N_tokens, bước nhảy = 9):
            
            for mỗi local_d_group nằm trong cụm SRAM đã nạp:
                
                1.2.1. Reset pp_v_reg hiện tại ở core
                
                1.2.2. SAS và Tính toán (Core_SAS_and_compute):
                   - pp_spike = 0
                   - for mỗi cột j của hàng Q[i] (j = 0 -> d):
                       + Kiểm tra global bitmap của Q[i][j] (quét 8 timesteps)
                       + Nếu Q[i][j] == 1 hoặc không bật SAS:
                           * for mỗi cụm N_tokens tile (nếu N_tokens > 512):
                               - Gọi spike_rf.load_spike_DRAM_to_SRAM_SSA() nạp hàng K^T[j] vào SpikeRF
                               - for mỗi block 64 tokens (từ SpikeRF):
                                   > Sinh 64-bit global_bitmaps từ K^T
                                   > Gọi core_fetchers.fetch_data_and_compute() để mảng 3D tính WA (mask = Q AND K^T)
                               - !pp_spike
                       + Nếu Q[i][j] == 0 và bật SAS: Skip toàn bộ bước nạp và tính toán trên

                1.2.3. Tính MU và Nhân Scale:
                   - for mỗi oc thật sự trong local_d_group:
                       + Lấy psum từ core_array3d
                       + Tính rò rỉ và nhân scale: v_mem = (v_mem >> 1) + psum
                       + Nếu v_mem >= V_th: Phát xung (Output Spike = 1) và trừ đi V_th
                       + Lưu Output Spike ra DRAM theo layout [C_out][N_tokens][T]
                       + Hàm save_txt_SSA_LIF_reshape_H_W() được gọi trong main() sẽ tách [N_tokens] -> [H][W] là output final
                
                1.2.4. !pp_v_reg: Đã tính xong cho local_d_group hiện tại. Lật cờ để group tiếp theo tính WA trong khi group cũ làm MU.
            
            // Xong tất cả d_group trong 1 lần tile của 1 token

        // Xong tất cả các Tokens trong 1 lần tile d_groups
        
    // Xong tất cả d_groups của 1 head
    
// Xong toàn bộ các heads, xong OFM
```

---

## End
--------
