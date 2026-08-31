#pragma once
#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <vector>
#include <omp.h>
#include <cstdint>

using namespace std;

constexpr int A_D = 8;  // Chiều sâu: 8 Timesteps
constexpr int A_H = 8;  // Chiều cao: 8 Output Channels (1 Vector chứa 8 số)
constexpr int A_W = 8;  // Chiều rộng: 8 Input Channels (8 Bank SRAM)
constexpr int max_WeightVector_per_bank = 256;  // Mỗi bank chứa tối đa 256 WeightVector
constexpr int max_ic_per_spike_rf = 512;        // Mỗi Spike RF chứa tối đa 512 ic
constexpr int N = 4;    // Cửa số tìm kiếm của Load Balancer
constexpr int Num_Cores = 9;

class SpikeRF {
private:
    // Vị trí của spike trong IFM mà core đang lưu
    int h_in = -1;
    int w_in = -1;

    // Nếu N_in > max_ic_per_spike_rf thì cần tile
    // Với kích thước model SpikingFormer 8-768 thì chỉ có QKV cần tile
    int ic_tile_start = -1;

    // Số lần load spike từ Spk_Mem vào Spike RF của core
    int spike_load_DRAM_to_SRAM_count = 0;

    // 2 SpikeRF ping-pong, mỗi RF chứa 512 ic qua A_D timestep
    // Sconv: load spike từ DRAM hoặc mượn core hàng xóm, đảo pp giữa các input position
    // QKV, SSA: luôn phải load spike từ DRAM, không mượn được của Core hàng xóm
    // đồng thời C_in của QKV = 768 > 512 nên dùng luôn 2 thanh pp để tile và lưu
    char local_spike[2][max_ic_per_spike_rf][A_D] = {0};

public:
    void reset_spike_rf(bool pp){
        for(int c = 0; c < max_ic_per_spike_rf; c++)
            for(int t = 0; t < A_D; t++)
                local_spike[pp][c][t] = 0;
    }

    void set_input_position(int h_in, int w_in, int ic_tile_start){
        this->h_in = h_in; this->w_in = w_in; this->ic_tile_start = ic_tile_start;
    }

    void load_spike_DRAM_to_SRAM_SConv(
        // input_spike có layout [C_in][H][W][T]
        // Một lần load chỉ load vị trí (h_in, w_in) lên Core
        const vector<char>& input_spikes, int H, int W, int C_in, int T, bool pp
    ) {
        for (int ic = 0; ic < max_ic_per_spike_rf && ic_tile_start + ic < C_in; ic++) {
            int ic_actual = ic_tile_start + ic; // ic thật sự trong input nếu cần tile để nạp
            for (int t = 0; t < T; t++) {
                // h_in và w_in lưu position hiện tại cần load để tính ở Core
                // Sẽ được Router điều phối và chốt trước khi nạp vào SpikeRF
                int in_idx = ic_actual * H * W * T + h_in * W * T + w_in * T + t;
                local_spike[pp][ic][t] = input_spikes[in_idx];
            }
        }
        spike_load_DRAM_to_SRAM_count++;
    }

    void load_spike_DRAM_to_SRAM_QKV(
        // input_spike có layout [C_in][N_tokens][T]
        // N_tokens = H * W chuyển output Sconv về chuỗi 1D làm input cho QKV_gen
        // Một lần load chỉ load vị trí n_tokens lên Core
        const vector<char>& input_spikes, int n_tokens, int N_tokens, int C_in, int T, bool pp
    ) {
        for (int ic = 0; ic < max_ic_per_spike_rf && ic_tile_start + ic < C_in; ic++) {
            int ic_actual = ic_tile_start + ic; // ic thật sự trong input nếu cần tile để nạp
            for (int t = 0; t < T; t++) {
                int in_idx = ic_actual * N_tokens * T + n_tokens * T + t;
                local_spike[pp][ic][t] = input_spikes[in_idx];
            }
        }
        spike_load_DRAM_to_SRAM_count++;
    }

    void load_spike_DRAM_to_SRAM_SSA(
        // input_spike là ma trận K^T có layout [d][N][T], N_token đóng vai trò N_in
        // Một lần load chỉ load hàng thứ j của K^T (ứng với global bitmap Q[i][j] = 1) lên Core
        const vector<char>& input_spikes, int j, int N_tokens, int T, bool pp
    ) {
        for (int n = 0; n < max_ic_per_spike_rf && ic_tile_start + n < N_tokens; n++) {
            int n_actual = ic_tile_start + n;
            for (int t = 0; t < T; t++) {
                int in_idx = j * N_tokens * T + n_actual * T + t;
                local_spike[pp][n][t] = input_spikes[in_idx];
            }
        }
        spike_load_DRAM_to_SRAM_count++;
    }

    void borrow_spike_rf_next_core(
        int core_id, const SpikeRF& borrow_SpikeRF,
        int N_in, int T, bool pp)
    {
        for (int ic = 0; ic < max_ic_per_spike_rf && ic_tile_start + ic < N_in; ic++) {
            for (int t = 0; t < T; t++) {
                // ping/pong của core đang xét sẽ copy spikes từ pong/ping của core hàng xóm
                local_spike[pp][ic][t] = borrow_SpikeRF.local_spike[!pp][ic][t];
            }
        }
    }

    int get_prev_h_in() {return h_in;}
    int get_prev_w_in() {return w_in;}
    int get_prev_ic_tile_start() {return ic_tile_start;}
    int get_spike_load_DRAM_to_SRAM_count() {return spike_load_DRAM_to_SRAM_count;}

    // Getter trả về mảng 2D khi có tham số pp cụ thể
    const char (*get_local_spike(bool pp) const)[A_D] {
        return local_spike[pp];
    }

    void reset_all_data() {
        h_in = -1; w_in = -1; 
        ic_tile_start = -1;
        spike_load_DRAM_to_SRAM_count = 0;
        reset_spike_rf(0); reset_spike_rf(1);
    }
};

struct V_reg {
    // 2 Register file ping-pong chứa membrane potential của A_H oc qua A_D timesteps
    int RF[2][A_D][A_H] = {0};
    void reset(bool pp) {
        for(int t = 0; t < A_D; t++) {
            for(int c = 0; c < A_H; c++) RF[pp][t][c] = 0;
        }
    }
};

struct WeightVector {
    int8_t w[A_H] = {0}; // vector chứa 8 weight đơn lẻ ứng với 8 oc theo chiều dọc
    void reset() {
        for(int i = 0; i < A_H; i++) w[i] = 0;
    }
};

class MultiBankMemory {
private:
    // Chia Core_mem thành A_W = 8 banks, mỗi bank chứa 256 WeightVector (tức tối đa 2048 ic)
    // Kích thước mỗi bank: 256 * 8 * 1 byte (weight kiểu int8) = 2k Bytes
    // Kích thước này đủ chứa toàn bộ N_in của tất cả layer trong model SpikingFormer 8-768
    WeightVector banks[A_W][max_WeightVector_per_bank];
    int num_ic_per_bank = 0;
    int num_oc_group_per_bank = 0;

    // Một lần load 16kB từ DRAM lên 8 bank SRAM ở 1 core thì biến đếm + 1
    int weight_load_DRAM_to_SRAM_count = 0;

public:
    void set_num_ic_oc_of_bank(int N_in) {
        // Phải tính toán theo fan-in tức gom hết các ic với cùng oc_group để tính
        // Ưu tiên lấy toàn bộ N_in chia đủ vào các bank, rồi tiling N_out để load và tính
        // Ví dụ N_in = 128 thì mỗi bank chứa 128/8 = 16 ic và 256/16 = 16 oc_groups
        // => Cách lưu weight như thế nào phụ thuộc kích thước cụ thể của N_in, N_out
        num_ic_per_bank = (N_in + A_W - 1) / A_W;
        num_oc_group_per_bank = max_WeightVector_per_bank / num_ic_per_bank;
    }

    // Load toàn bộ weight trong 1 lần tile từ DRAM vào multi-bank SRAM:
    // Với Sconv: hàm điều phối sẽ tách kh, kw về mỗi core để đưa weight array về layout 2 chiều [C_out][C_in]
    // Với QKV_gen: weight array có layout 2 chiều [C_out][C_in]
    // Với SSA: weight array là ma trận V được nén chiều timesteps về layout 2 chiều [N_tokens][d]
    // d và C_out tổng quát lại thành N_out, N_tokens và C_in tổng quát lại thành N_in
    void load_weight_DRAM_to_SRAM(
        int oc_tile_start, int N_in, int N_out,
        const vector<int8_t>& weight_array, bool is_SSA)
    {
        for (int ic = 0; ic < N_in; ic++) {
            for (int oc = 0; oc < (num_oc_group_per_bank * A_H) && oc_tile_start + oc < N_out; oc++) {
                // chỉ số bank là (ic % A_W): bank thứ x chứa ic 8*0 + x, 8*1 + x, ...
                int ic_offset = ic / A_W;
                int oc_group_offset = oc / A_H;
                int local_addr = oc_group_offset * num_ic_per_bank + ic_offset;
                // oc offset trong mảng w của WeightVector là (oc % A_H)
                // oc thật sự là oc + oc_tile_start, ic là ic thật sự do chắc chắn load được hết N_in lên SRAM
                banks[ic % A_W][local_addr].w[oc % A_H] = weight_array[
                    is_SSA ? ic * N_out + (oc_tile_start + oc) : (oc_tile_start + oc) * N_in + ic];
            }
        }
        weight_load_DRAM_to_SRAM_count++;
    }

    // Hàm lấy 1 WeightVector từ SRAM đưa vào array theo chỉ số bank và addr cụ thể trong bank
    // Việc load này do Non-zero Data Fetcher điều phối
    WeightVector get_WeightVector_from_SRAM(int bank_idx, int ic_offset, int oc_group_offset) {
        int local_addr = oc_group_offset * num_ic_per_bank + ic_offset;
        return banks[bank_idx][local_addr];
    }

    int get_num_ic_per_bank() {return num_ic_per_bank;}
    int get_num_oc_group_per_bank() {return num_oc_group_per_bank;}
    int get_weight_load_DRAM_to_SRAM_count(){return weight_load_DRAM_to_SRAM_count;}

    void reset_all_data(){
        for (int i = 0; i < A_W; i++)
            for (int j = 0; j < max_WeightVector_per_bank; j++)
                banks[i][j].reset();
        num_ic_per_bank = 0;
        num_oc_group_per_bank = 0;
        weight_load_DRAM_to_SRAM_count = 0;
    }
};

class ComputationArray3D {
private:
    V_reg v_reg;                // Khối thanh ghi Psum nội bộ của mảng 3d
    long long SOP_count = 0;    // Số phép cộng dồn (ứng với 1 weight có xung đơn lẻ)

public:
    // Hàm tính WA sau khi có đủ fetched weight và spikes trong 1 cycle chip
    void compute_WA_one_cycle(
        WeightVector fetched_vectors[A_W], bool bank_busy[A_W],
        char spike_mask[A_W][A_D], bool pp, bool is_SSA,
        // Nếu C_out % A_H != 0 thì oc_group cuối cùng sẽ không đủ A_H oc
        // Số oc thật sự còn lại sẽ được tính toán trong hàm điều phối
        int num_oc_actual)
    {
        for (int t = 0; t < A_D; t++) {
            // Unroll 8 Bank để giả lập các bộ cộng nằm ngang (A_W)
            #pragma GCC unroll 8
            for (int i = 0; i < A_W; i++) {

                // Chỉ cộng nếu có mask kích hoạt
                // Với Sconv, QKV: mask là local bitmap hay input spikes
                // Với SSA: mask là local bitmap Q AND K^T
                if (bank_busy[i] && spike_mask[i][t]) {
                    // 1 spike mask bật các cổng MUX lấy weight dọc theo chiều A_H
                    // Nên so sánh lại với A_H phòng khi tính sai giá trị num_oc_actual > A_H
                    SOP_count += min(num_oc_actual, A_H);

                    #pragma omp simd
                    for (int oh = 0; oh < min(num_oc_actual, A_H); oh++) {
                        // Nếu tính SSA:
                        // Mỗi fetched_vector chứa A_H weight dạng int8, ghép từ 8 timesteps của V tại mỗi vị trí
                        // Weight này sẽ không được tái sử dụng theo chiều A_D trong 3D array khi tính SSA
                        // mà tại mỗi t, ta chỉ lấy 1 bit ở vị trí t tương ứng rồi chèn thêm 7 bit 0 ở trước
                        v_reg.RF[pp][t][oh] +=
                            is_SSA ? (fetched_vectors[i].w[oh] >> t) & 1 : fetched_vectors[i].w[oh];
                    }
                }
            }
        }
    }

    long long get_SOP_count() {return SOP_count;}

    void reset_v_reg(bool pp) {v_reg.reset(pp);}

    int get_v_reg(int t, int oh, bool pp){return v_reg.RF[pp][t][oh];}

    void reset_all_data() {
        reset_v_reg(0); reset_v_reg(1);
        SOP_count = 0;
    }
};

class NonZeroDataFetcher {
private:
    int cycle_count = 0;
    long long weight_load_SRAM_to_array_count = 0;

public:
    unsigned long long generate_64_global_bitmap(
        // input từ spike RF thay vì chọc vào spike mem chung của 9 cores
        const char local_spike_data[max_ic_per_spike_rf][A_D],
        // global_ic_start = 0, 64, 128, ... tính trên toàn N_in
        int T, int N_in, int global_ic_start)
    {
        unsigned long long global_bitmap = 0; // biến kiểu long long tương đương chuỗi 64 bits
        int channels_to_scan = min(64, N_in - global_ic_start);
        
        // Chỉ tạo bitmap cho các ic thật sự còn lại, chưa chắc là full 64 ic
        for (int c = 0; c < channels_to_scan; c++) {
            int actual_c = global_ic_start + c;
            for (int t = 0; t < T; t++) {
                // actual_c có thể > max_ic nên phải chia lấy dư
                if (local_spike_data[actual_c % max_ic_per_spike_rf][t]) {
                    global_bitmap |= (1ULL << c);
                    break;
                }
            }
        }
        return global_bitmap;
    }

    void split_global_bitmap_to_banks(
        unsigned long long global_bitmap, unsigned char bank_bitmaps[A_W]
    ) {
        for (int c = 0; c < 64; c++) {
            if ((global_bitmap >> c) & 1){
                int bank_id = c % A_W;          // Lấy dư khi chia cho 8 để biết thuộc bank nào
                int local_bit_idx = c / A_W;    // Vị trí 0-7 bên trong bank đó
                bank_bitmaps[bank_id] |= 1<<local_bit_idx;
            }
        }
    }

    int address_generator(unsigned char& bank_bitmaps) {
        if (!bank_bitmaps) return -1;
        else {
            int i = 0;
            // lặp đến khi bit thứ i từ phải sang bằng 1 (tức bank_bitmaps>>i AND 1 = 1)
            while (!((bank_bitmaps>>i)&1)) i++;
            // reset bit thứ i về 0 (local bitmap AND với NOT(1<<i))
            bank_bitmaps &= ~(1<<i);
            return i;
        }
    }

    void load_WeightVector_SRAM_to_array(
        int bank_idx,   // chỉ số bank cần lấy weight
        int borrow_idx, // chỉ số bank đi mượn: nếu tự load mà không mượn thì borrow_idx = bank_idx
        int local_ic, int local_oc_group, // ic và oc_group offset trong bank
        MultiBankMemory& mem,
        WeightVector fetched_weights[A_W], bool bank_busy[A_W], int fetched_c_in[A_W]
    ) {
        // Load WeightVector từ SRAM vào mảng tạm thời
        fetched_weights[bank_idx] = mem.get_WeightVector_from_SRAM(borrow_idx, local_ic, local_oc_group);

        weight_load_SRAM_to_array_count++;
        bank_busy[bank_idx] = true;
        // ic thật sự: bank 0 (0, 8, ...), bank 1 (1, 9, ...), ...
        fetched_c_in[bank_idx] = A_W * local_ic + borrow_idx;
    }

    void load_balancer(
        unsigned char bank_bitmaps[A_W],
        int local_oc_group,     // 0, 1, ..., num_oc_group_per_bank
        // global_ic_start = 0, 64, ... tính trên toàn N_in, nạp đủ vào W_mem SRAM
        int global_ic_start,
        MultiBankMemory& mem,
        WeightVector fetched_weights[A_W], bool bank_busy[A_W], int fetched_c_in[A_W]
    ) {
        int request[A_W];
        for (int i = 0; i < A_W; i++) request[i] = -1;

        for (int idle_bank = 0; idle_bank < A_W; idle_bank++) {
            if (!bank_busy[idle_bank]) {
                // Quét cửa sổ tìm kiếm lân cận (N = 4)
                for (int n = 1; n <= N; n++) {
                    int neighbor = (idle_bank - n + A_W) % A_W;
                    // Bank nào còn global bit = 1 (chuỗi bitmap != 0) thì đăng ký mượn
                    if (bank_bitmaps[neighbor]) {
                        request[idle_bank] = neighbor;
                        break;
                    }
                }
            }
        }

        // Chốt quyền mượn theo thứ tự ưu tiên (bank chỉ số nhỏ hơn được mượn trước)
        for (int idle_bank = 0; idle_bank < A_W; idle_bank++) {
            if (!bank_busy[idle_bank] && request[idle_bank] != -1) {
                int neighbor = request[idle_bank];
                // Kiểm tra lại xem neighbor còn data không
                // Phòng trường hợp bank này đã bị idle_bank có ưu tiên cao hơn mượn rồi
                if (bank_bitmaps[neighbor]) {
                    // Chốt quyền mượn, tính ic_offset trong bank được mượn
                    int borrow_local_ic = global_ic_start / A_W + address_generator(bank_bitmaps[neighbor]);
                    load_WeightVector_SRAM_to_array(
                        idle_bank, neighbor, borrow_local_ic, local_oc_group, mem, fetched_weights, bank_busy, fetched_c_in);
                }
            }
        }
    }

    void fetch_data_and_compute(
        unsigned long long& global_bitmap,    // chuỗi 64 bit
        int local_oc_group,                   // 0, 1, ..., num_oc_group_per_bank
        // Nếu C_out % A_H != 0 thì oc_group cuối cùng sẽ không đủ A_H oc
        // Số oc thật sự còn lại sẽ được tính toán trong hàm điều phối
        int num_oc_actual,
        // global_ic_start = 0, 64, ... tính trên toàn N_in, nạp đủ vào W_mem SRAM
        int global_ic_start,
        int N_in, MultiBankMemory& mem,
        // Input spike trong Sconv, QKV hoặc là spike của K^T trong SSA
        const char local_spike_1[max_ic_per_spike_rf][A_D],
        // NULL (không dùng đến) trong Sconv, QKV hoặc là spike của Q trong SSA
        const char local_spike_2[A_D],
        int T, ComputationArray3D& arr3d, bool pp_v_reg,
        bool ena_nz, bool ena_lb, bool is_SSA)
    {
        // MODE 1: NO NZ NO LB. TÍNH TOÁN TUẦN TỰ TỪNG IC, KHÔNG DÙNG ĐẾN TWO STAGE BITMAP
        if (!ena_nz) {
            // Số kênh thật sự còn phải tính trong cụm tối đa 64 kênh đang xét
            int channels_to_scan = min(64, N_in - global_ic_start);
            int num_blocks = (channels_to_scan + A_W - 1) / A_W; // Số block 8-bank cần chạy

            // Chia channels_to_scan thành các block, mỗi block tối đa A_W ic
            for (int ic_block = 0; ic_block < num_blocks; ic_block++) {

                // Mảng chứa các vector trọng số được lấy ra trong cycle hiện tại để đưa vào 3D array
                WeightVector fetched_weights[A_W];
                // Mảng đánh dấu bank nào đã xuất dữ liệu, bank nào rảnh
                bool bank_busy[A_W] = {false};
                // Mảng ghi lại các ic actual đã được fetch weight vào 3D array
                int fetched_c_in[A_W] = {0};

                for (int i = 0; i < A_W; i++) { // i là index của bank
                    // Chỉ nạp Weight nếu kênh này thực sự tồn tại (nhỏ hơn channels_to_scan)
                    if (ic_block * A_W + i < channels_to_scan) {
                        int local_ic = global_ic_start / A_W + ic_block;   // ic_offset trong bank
                        load_WeightVector_SRAM_to_array(
                            i, i, local_ic, local_oc_group, mem, fetched_weights, bank_busy, fetched_c_in);
                    }
                }

                // Tạo spike_mask cho các vector weights vừa được lấy ra
                char spike_mask[A_W][A_D] = {0};
                for (int i = 0; i < A_W; i++) {
                    if (bank_busy[i]) { // Chỉ tạo mask nếu bank có dữ liệu hợp lệ
                        for (int t = 0; t < T; t++) {
                            // fetched_c_in[i] ghi lại ic actual, phải chia lấy dư cho max_ic_per_spike_rf
                            int fetched_c_in_offset = fetched_c_in[i] % max_ic_per_spike_rf;
                            
                            // Với SSA: mask = local bitmap của K^T AND Q
                            // Với Sconv, QKV_gen: mask đơn giản là local bitmap của input spike
                            spike_mask[i][t] = is_SSA ?
                            (local_spike_1[fetched_c_in_offset][t] && local_spike_2[t]) : local_spike_1[fetched_c_in_offset][t];
                        }
                    }
                }

                // Một lần load các WeightVectors lên mảng 3D và tính toán mất 1 cycle chip
                cycle_count++;
                // Đưa toàn bộ weight và spike_mask đã fetch từ SRAM vào mảng 3D rồi tính WA
                arr3d.compute_WA_one_cycle(fetched_weights, bank_busy, spike_mask, pp_v_reg, is_SSA, num_oc_actual);

                // Quay lại thực hiện ic_block tiếp theo
            }
        } else {
            // MODE 2, 3: ENABLE NZ DATA FETCHER
            unsigned char bank_bitmaps[A_W] = {0};
            // Split 64 global bitmaps thành 8 cụm 8 local bitmaps
            split_global_bitmap_to_banks(global_bitmap, bank_bitmaps);

            // load weight và tính WA trong mảng 3D đến khi tất cả global bitmap = 0
            while (true) {
                // Đầu tiên check xem cả 64 input channel có trống hay không?
                unsigned char all_empty = 0;
                for (int i = 0; i < A_W; i++) all_empty |= bank_bitmaps[i];
                // Nếu tất cả input spike là 0 (tức all_empty = 0) thì kết thúc
                if (!all_empty) break;
                
                // Mảng chứa các vector trọng số được lấy ra trong cycle hiện tại để đưa vào 3D array
                WeightVector fetched_weights[A_W];
                // Mảng đánh dấu bank nào đã xuất dữ liệu, bank nào rảnh
                bool bank_busy[A_W] = {false};
                // Mảng ghi lại các ic actual đã được fetch weight vào 3D array
                int fetched_c_in[A_W] = {0};

                // Các bank tự xử lý dữ liệu: borrow_idx = bank_idx = i
                for (int i = 0; i < A_W; i++) { // i là index của bank
                    int addr = address_generator(bank_bitmaps[i]);
                    if (addr != -1) {
                        int local_ic = global_ic_start / A_W + addr;   // ic_offset trong bank
                        load_WeightVector_SRAM_to_array(
                            i, i, local_ic, local_oc_group, mem, fetched_weights, bank_busy, fetched_c_in);
                    }
                }

                // MODE 3: NZ AND LB
                if (ena_lb) {
                    load_balancer(bank_bitmaps, local_oc_group, global_ic_start, mem,
                        fetched_weights, bank_busy, fetched_c_in);
                }

                // Lấy local bitmaps cho 8 vector weights vừa được lấy ra
                char spike_mask[A_W][A_D] = {0};
                for (int i = 0; i < A_W; i++) {
                    if (bank_busy[i]) {
                        for (int t = 0; t < T; t++) {
                            // fetched_c_in[i] ghi lại ic actual, phải chia lấy dư cho max_ic_per_spike_rf
                            int fetched_c_in_offset = fetched_c_in[i] % max_ic_per_spike_rf;

                            // Với SSA: mask = local bitmap của K^T AND Q
                            // Với Sconv, QKV_gen: mask đơn giản là local bitmap của input spike
                            spike_mask[i][t] = is_SSA ?
                            (local_spike_1[fetched_c_in_offset][t] && local_spike_2[t]) : local_spike_1[fetched_c_in_offset][t];
                        }
                    }
                }
                
                // Một lần load các WeightVectors lên mảng 3D và tính toán mất 1 cycle chip
                cycle_count++;
                // Đưa toàn bộ weight và input spikes (local bitmaps) đã fetch vào mảng 3D rồi tính WA
                arr3d.compute_WA_one_cycle(fetched_weights, bank_busy, spike_mask, pp_v_reg, is_SSA, num_oc_actual);

                // Quay lại vòng lặp, fetch weight và local spike rồi tính đến hết 64 global bitmaps
            }
        }
    }

    int get_cycle_count() {return cycle_count;}
    long long get_weight_load_SRAM_to_array_count() {return weight_load_SRAM_to_array_count;}

    void reset_all_data() {
        cycle_count = 0;
        weight_load_SRAM_to_array_count = 0;
    }
};

class AsyncCoreMemRouter {
private:
    int router_shared_count = 0;

public:
    void get_spikes_with_router(
        int core_id, int c_in_start, int h_in, int w_in, bool pp,
        const vector<char>& input_spikes, SpikeRF spike_rf[Num_Cores],
        int T, int C_in, int H, int W)
    {
        // Không được reset Spike_RF vì khi chuyển oc_group mà vẫn đang ở 1 vị trí (h,w) của OFM
        // thì Spike_RF phải giữ nguyên để tính tiếp
        bool found = false; int found_id = -1;

        // Quét xem core lân cận nào có dữ liệu tọa độ này
        // Phải làm vậy do ở vị trí biên hoặc ở ic_tile_start mới thì các core không mượn được
        // Với các vị trí còn lại thì core 0 mượn 1, 1 mượn 2, 2 tự load, 3 mượn 4, ... (stride = 1)
        for (int i = 0; i < 9; i++) {
            if (spike_rf[i].get_prev_h_in() == h_in
                && spike_rf[i].get_prev_w_in() == w_in
                && spike_rf[i].get_prev_ic_tile_start() == c_in_start)
            { found = true; found_id = i; break; }
        }

        // Sau khi quét tìm xong mới được gán h_in, w_in, c_in_start mới cho core này
        // Và phải gán position mới trước khi chốt mượn hoặc load từ ngoài DRAM
        spike_rf[core_id].set_input_position(h_in, w_in, c_in_start);

        // Nếu tìm được spike hợp lệ TỪ CORE KHÁC thì đi mượn
        if (found && found_id != core_id) {
            // Copy local_spike từ core hàng xóm
            spike_rf[core_id].borrow_spike_rf_next_core(core_id, spike_rf[found_id], C_in, T, pp);
            router_shared_count++;
        } else if (!found) {
            // không found thì tự load từ spike mem bên ngoài DRAM
            spike_rf[core_id].load_spike_DRAM_to_SRAM_SConv(input_spikes, H, W, C_in, T, pp);
        }
        // Trường hợp còn lại: mượn của chính mình xảy ra khi chuyển oc_group mà vẫn đang ở 1 vị trí (h,w) của OFM
        // Khi đó Spike_RF giữ nguyên, không cần đi mượn hoặc load lại từ DRAM, do đó không được reset từ trước
    }

    int get_router_shared_count(){return router_shared_count;}

    void reset_all_data() {router_shared_count = 0;}
};

struct StatisticalData {
    long long SOP_count = 0;
    int weight_load_DRAM_to_SRAM_count = 0;
    long long weight_load_SRAM_to_array_count = 0;
    int cycle_count = 0;
    int average_cycle_all_core = 0;
    float average_num_WeightVector_load_per_cycle = 0;
    int spike_load_DRAM_to_SRAM_count = 0;
    int router_shared_count = 0;
    int SSA_operation_by_SAS_count = 0;
    int SSA_skip_by_SAS_count = 0;

    void count_average_num_WeightVector_load_per_cycle(){
        float val = ((float)weight_load_SRAM_to_array_count / Num_Cores) / average_cycle_all_core;
        // làm tròn đến 2 chữ số sau dấu phẩy
        average_num_WeightVector_load_per_cycle = round(val*100.0f) / 100.0f;
    }
    
    void reset_data(){
        SOP_count = 0;
        weight_load_DRAM_to_SRAM_count = 0;
        weight_load_SRAM_to_array_count = 0;
        cycle_count = 0;
        average_cycle_all_core = 0;
        average_num_WeightVector_load_per_cycle = 0;
        spike_load_DRAM_to_SRAM_count = 0;
        router_shared_count = 0;
        SSA_operation_by_SAS_count = 0;
        SSA_skip_by_SAS_count = 0;
    }
};

vector<int8_t> load_txt_weight(const string& file, int size) {
    vector<int8_t> a(size, 0);
    ifstream f(file);
    if (!f.is_open()) {
        cerr << "Khong the mo file " << file << "\n";
        exit(1);
    }
    int temp; // Sử dụng biến trung gian kiểu int
    for (int i = 0; i < size; i++) {
        f >> temp;
        a[i] = (int8_t)temp; // Ép kiểu về int8 để làm weight
    }
    return a;
}

vector<char> load_txt_spike(const string& file, int size) {
    vector<char> a(size, 0);
    ifstream f(file);
    if (!f.is_open()) {
        cerr << "Khong the mo file " << file << "\n";
        exit(1);
    }
    char c;
    int count = 0;
    // Đọc từng KÝ TỰ, không đọc kiểu int do spike của các timestep ghi liền nhau
    while (count < size && f >> c) { 
        if (c == '0' || c == '1') {
            a[count++] = c - '0'; // Đổi từ ký tự mã ASCII '0'/'1' sang giá trị số 0/1
        }
    }
    return a;
}

void save_txt_Sconv_LIF(const string& file, const vector<char>& output, int C_out, int H, int W, int T) {
    ofstream f(file);
    if (!f.is_open()) {
        cerr << "Loi khi tao file " << file << "\n";
        return;
    }
    // In theo định dạng [C_out][H][W][T]
    for (int oc = 0; oc < C_out; oc++) {
        for (int oh = 0; oh < H; oh++) {
            for (int ow = 0; ow < W; ow++) {
                for (int t = 0; t < T; t++) {
                    f << (int)output[oc*H*W*T + oh*W*T + ow*T + t];
                }
                f << " ";
            }
            f << "\n";
        }
        f << "\n";
    }
    f.close();
}

void save_txt_QV_LIF(const string& file, const vector<char>& output, int N, int d, int T) {
    ofstream f(file);
    if (!f.is_open()) {
        cerr << "Loi khi tao file " << file << "\n";
        return;
    }
    // In theo định dạng [N][d][T] (Nén ngang)
    for (int n = 0; n < N; n++) {         // Quét qua từng Token (Batch)
        for (int c = 0; c < d; c++) {     // Quét qua từng Kênh của Token đó
            for (int t = 0; t < T; t++) { // Quét qua 8 Timesteps (Vector 8-bit)
                // Tính index 1D theo thứ tự: Token -> Channel -> Timestep
                int idx = n * d * T + c * T + t;
                f << (int)output[idx];
            }
            f << " "; // In dấu cách giữa các Channel (d)
        }
        f << "\n"; // Xuống dòng khi hết 1 Token -> Mỗi dòng là 1 Token
    }
    f << "\n";
    f.close();
}

void save_txt_K_Transpose_LIF(const string& file, const vector<char>& output, int N, int d, int T) {
    ofstream f(file);
    if (!f.is_open()) {
        cerr << "Loi khi tao file " << file << "\n";
        return;
    }
    // In theo định dạng [d][N][T] (Nén dọc)
    for (int c = 0; c < d; c++) {         // Quét qua từng Token (Batch)
        for (int n = 0; n < N; n++) {     // Quét qua từng Kênh của Token đó
            for (int t = 0; t < T; t++) { // Quét qua 8 Timesteps (Vector 8-bit)
                // Tính index 1D theo thứ tự: Channel -> Token -> Timestep
                int idx = c * N * T + n * T + t;
                f << (int)output[idx];
            }
            f << " "; // In dấu cách giữa các Token (N)
        }
        f << "\n"; // Xuống dòng khi hết 1 channel (d) -> Mỗi dòng là 1 channel
    }
    f << "\n";
    f.close();
}

void save_txt_SSA_LIF_reshape_H_W(const string& file, const vector<char>& output, int C_out, int H, int W, int T) {
    ofstream f(file);
    if (!f.is_open()) {
        cerr << "Loi khi tao file " << file << "\n";
        return;
    }
    // Reshape định dạng [C_out][N][T] -> [C_out][H][W][T]
    for (int oc = 0; oc < C_out; oc++) {
        for (int oh = 0; oh < H; oh++) {
            for (int ow = 0; ow < W; ow++) {
                for (int t = 0; t < T; t++) {
                    f << (int)output[oc*H*W*T + (oh*W+ow)*T + t];
                }
                f << " ";
            }
            f << "\n";
        }
        f << "\n";
    }
    f.close();
}

void save_txt_Sconv_v_mem(const string& file, const vector<int>& output, int C_out, int H, int W, int T) {
    ofstream f(file);
    if (!f.is_open()) {
        cerr << "Loi khi tao file " << file << "\n";
        return;
    }
    // In theo định dạng [C_out][H][W][T]
    for (int oc = 0; oc < C_out; oc++) {
        for (int oh = 0; oh < H; oh++) {
            for (int ow = 0; ow < W; ow++) {
                for (int t = 0; t < T; t++) {
                    f << output[oc*H*W*T + oh*W*T + ow*T + t] << " ";
                }
                f << " ";
            }
            f << "\n";
        }
        f << "\n";
    }
    f.close();
}