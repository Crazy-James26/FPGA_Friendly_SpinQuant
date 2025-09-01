#include <iostream>
#include <vector>
#include <random>
#include <gflags/gflags.h>

DEFINE_string(bitstream, "", ""/*path to bitstream file, run csim if empty*/);

#include "config.h"
#include "SpinQuant_Decoding_flatten.h"

#include <algorithm>
#include <numeric>
#include <cmath>


void SpinQuant_Decoding_test(int argc, char* argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

    cout << "Token Block Parallel: " << T_BLOCK_PARALLEL << endl;
    cout << "Token QKVO FFN LM_HEAD Block Parallel: " << T_QKVO_FFN_BLOCK_PARALLEL << endl;
    cout << "Dec QKVO FFN LM_HEAD Weight Parallel: " << DEC_QKVO_FFN_W_PARALLEL << endl;
    cout << "Dec MHA Head Parallel: " << DEC_HEAD_PARALLEL << endl;
    cout << "Dec MHA K Weight Parallel: " << DEC_K_PARALLEL << endl;
    cout << "Dec MHA V Weight Parallel: " << DEC_V_PARALLEL << endl;
    
    // random seeds
    vector<float, tapa::aligned_allocator<float>> rand_seeds_mmap(MAX_DEC_SEQ_LEN);

    // decoder token idx
    vector<int, tapa::aligned_allocator<int>> sampled_token_idx_mmap(MAX_DEC_SEQ_LEN);

    // vocab library
    vector<hls::vector<float, T_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_BLOCK_PARALLEL>>> vocab_lib(
        VOCAB_SIZE * HIDDEN_DIM / T_BLOCK_PARALLEL
    );
    cout << "vocab_lib size: " << vocab_lib.size() << endl;

    // Input/Output mmap
    static vector<hls::vector<float, T_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_BLOCK_PARALLEL>>> io_mmap(
        (MAX_DEC_SEQ_LEN * (DECODER_LAYER_NUM + 1) + 1) * HIDDEN_DIM / T_BLOCK_PARALLEL
    );
    cout << "io_mmap size: " << io_mmap.size() << endl;

    // Linear Layer QKVO weight mmap
    cout << "w_qkvo_FFN_size: " << w_qkvo_FFN_size << endl;

    vector<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>>> w_qkvo_FFN_mmaps[T_QKVO_FFN_BLOCK_PARALLEL];
    for (int i = 0; i < T_QKVO_FFN_BLOCK_PARALLEL; i++) {
        w_qkvo_FFN_mmaps[i].resize(w_qkvo_FFN_size);
    }

    // Linear Layer QKVO weight_s_sum mmap
    cout << "w_s_qkvo_FFN_size: " << w_s_qkvo_FFN_size << endl;
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> w_s_sum_qkvo_FFN_mmap(
        w_s_qkvo_FFN_size
    );

    // KV caches
    vector<hls::vector<ap_int<8>, DEC_K_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_K_PARALLEL>>> k_caches[DEC_HEAD_PARALLEL];
    for(int i = 0; i < DEC_HEAD_PARALLEL; i++){
        k_caches[i].resize(DECODER_LAYER_NUM * KV_HEAD_NUM/DEC_HEAD_PARALLEL * MAX_SUM_SEQ_LEN/DEC_K_PARALLEL * HEAD_DIM);
    }
    cout << "k_caches size: " << k_caches[0].size() << endl;

    vector<hls::vector<ap_int<8>, DEC_V_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_V_PARALLEL>>> v_caches[DEC_HEAD_PARALLEL];
    for(int i = 0; i < DEC_HEAD_PARALLEL; i++){
        v_caches[i].resize(DECODER_LAYER_NUM * (KV_HEAD_NUM/DEC_HEAD_PARALLEL * HEAD_DIM/DEC_V_PARALLEL) * MAX_SUM_SEQ_LEN);
    }
    cout << "v_caches size: " << v_caches[0].size() << endl;

    // Layer Norm weight mmap
    vector<hls::vector<float, T_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_BLOCK_PARALLEL>>> gamma_beta_mmap(
        3 * DECODER_LAYER_NUM * HIDDEN_DIM / T_BLOCK_PARALLEL
    );
    cout << "gamma_beta_mmap size: " << gamma_beta_mmap.size() << endl;

    // 2) Initialize buffers with random data
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> distF(-1.0f/sqrt(HIDDEN_DIM), 1.0f/sqrt(HIDDEN_DIM));
    std::uniform_int_distribution<int>   dist4(-8, 7);
    std::uniform_int_distribution<int>   dist8(-128, 127);

    // Float initializer
    auto initFloatVec = [&](auto &cont, int width) {
        for (auto &vec : cont)
            for (int i = 0; i < width; ++i)
                vec[i] = distF(rng);
    };
    // Int4 initializer
    auto initInt4Vec = [&](auto &cont, int width) {
        for (auto &vec : cont)
            for (int i = 0; i < width; ++i)
                vec[i] = ap_int<4>(dist4(rng));
    };
    // Int8 initializer
    auto initInt8Vec = [&](auto &cont, int width) {
        for (auto &vec : cont)
            for (int i = 0; i < width; ++i)
                vec[i] = ap_int<8>(dist8(rng));
    };

    // initialize all the vocab_lib vectors and weights
    initFloatVec(vocab_lib, T_BLOCK_PARALLEL);

    // for (int i = 0; i < T_QKVO_FFN_BLOCK_PARALLEL; i++) {
    //     initInt4Vec(w_qkvo_FFN_mmaps[i], DEC_QKVO_FFN_W_PARALLEL);
    // }
    for (int i = 0; i < T_QKVO_FFN_BLOCK_PARALLEL; i++) {
        initInt8Vec(w_qkvo_FFN_mmaps[i], DEC_QKVO_FFN_W_PARALLEL/2);
    }


    initFloatVec(w_s_sum_qkvo_FFN_mmap, 2);

    initFloatVec(gamma_beta_mmap, T_BLOCK_PARALLEL);
    

    // 3) run kernel
    size_t io_init_vecs = HIDDEN_DIM/T_BLOCK_PARALLEL;

    const int num_runs = 10;
    int64_t total_time_ns = 0;
    std::cout << "kernel begins running " << num_runs << " times …\n";
    
    for (int run = 0; run < num_runs; ++run) {
    // Call Linear_Layer_tb
    // Apply initialization
        // init rand_seeds
        for (int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
            rand_seeds_mmap[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
        
        // Zero out the io vectors
        for (int idx = 0; idx < (MAX_DEC_SEQ_LEN * (DECODER_LAYER_NUM + 1) + 1) * io_init_vecs; idx++) {
            io_mmap[idx] = 0.0f;
        }

        // Random init first token's embedding
        int first_token_idx = rand() % VOCAB_SIZE;
        for (size_t idx = 0; idx < io_init_vecs; ++idx) {
            io_mmap[idx] = vocab_lib[first_token_idx * io_init_vecs + idx];
        }

        // Random init KV caches
        for (int i = 0; i < MAX_SUM_SEQ_LEN; i++) {
            for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
                for (int h = 0; h < KV_HEAD_NUM; h++) {
                    float scale = float(h+1)/HIDDEN_DIM;
                    for (int j = 0; j < HEAD_DIM; j++) {
                        int qval_k = std::round(distF(rng) / scale);
                        qval_k = std::max(-128, std::min(127, qval_k));
                        int idx = ((layer * KV_HEAD_NUM/DEC_HEAD_PARALLEL + h % (KV_HEAD_NUM/DEC_HEAD_PARALLEL)) * MAX_SUM_SEQ_LEN + i)/DEC_K_PARALLEL *
                                    HEAD_DIM + j;
                        int sub_idx = i % DEC_K_PARALLEL;
                        if(i < MAX_PRE_SEQ_LEN){
                            k_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][idx][sub_idx] = ap_int<8>(qval_k);
                        }
                        else {
                            k_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][idx][sub_idx] = ap_int<8>(0);
                        }

                        int qval_v = std::round(distF(rng) / scale);
                        qval_v = std::max(-128, std::min(127, qval_v));
                        idx = ((layer * (KV_HEAD_NUM/DEC_HEAD_PARALLEL) + h % (KV_HEAD_NUM/DEC_HEAD_PARALLEL)) * HEAD_DIM + j)/DEC_V_PARALLEL * MAX_SUM_SEQ_LEN + i;
                        sub_idx = j % DEC_V_PARALLEL;
                        if(i < MAX_PRE_SEQ_LEN){
                            v_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][idx][sub_idx] = ap_int<8>(qval_v);
                        }
                        else {
                            v_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][idx][sub_idx] = ap_int<8>(0);
                        }
                    }
                }
            }
        }

        // run the kernel
        cout << "kernel begins running!\n";

        
        int64_t kernel_time_ns = tapa::invoke(
            SpinQuant_Decoding, 
            FLAGS_bitstream,
            tapa::read_only_mmap<hls::vector<float, T_BLOCK_PARALLEL>>(vocab_lib),
            tapa::read_write_mmap<hls::vector<float, T_BLOCK_PARALLEL>>(io_mmap),
            // tapa::read_only_mmaps<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_QKVO_FFN_BLOCK_PARALLEL>(w_qkvo_FFN_mmaps),
            tapa::read_only_mmaps<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, T_QKVO_FFN_BLOCK_PARALLEL>(w_qkvo_FFN_mmaps),
            tapa::read_only_mmap<hls::vector<float, 2>>(w_s_sum_qkvo_FFN_mmap),
            tapa::read_write_mmaps<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL>(k_caches),
            tapa::read_write_mmaps<hls::vector<ap_int<8>, DEC_V_PARALLEL>, DEC_HEAD_PARALLEL>(v_caches),
            tapa::read_only_mmap<hls::vector<float, T_BLOCK_PARALLEL>>(gamma_beta_mmap),
            tapa::read_only_mmap<float>(rand_seeds_mmap),
            tapa::write_only_mmap<int>(sampled_token_idx_mmap),
            MAX_PRE_SEQ_LEN,
            MAX_DEC_SEQ_LEN
        );
            
        double t_s = kernel_time_ns * 1e-9;
        std::cout << "  Run " << run << " — kernel time: " << t_s << " s\n";
        total_time_ns += kernel_time_ns;
    }

    
    

    double avg_s = (total_time_ns / double(num_runs)) * 1e-9;
    std::cout << "Average kernel time over " << num_runs << " runs: " << avg_s << " s\n";
}

int main(int argc, char* argv[]) {
    SpinQuant_Decoding_test(argc, argv);
    return 0;
}