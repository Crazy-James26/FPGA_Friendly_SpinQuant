#include <iostream>
#include <vector>
#include <random>
#include <gflags/gflags.h>

DEFINE_string(bitstream, "", ""/*path to bitstream file, run csim if empty*/);

#include "config.h"
#include "SpinQuant_Prefilling.h"

void SpinQuant_test(int argc, char* argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

    cout << PRE_QKVO_W_PARALLEL_READ << " " 
         << PRE_QKVO_W_PARALLEL << " "
         << PRE_K_PARALLEL << " "
         << PRE_V_PARALLEL << " "
         << PRE_FFN_W_PARALLEL_READ << " "
         << PRE_FFN_W_PARALLEL << endl;

    // Input/Output mmap
    vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> io_mmap((DECODER_LAYER_NUM + 1) * MAX_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM);

    // Linear Layer QKVO weight mmap
    vector<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>>> wk_wq_mmap(
        DECODER_LAYER_NUM * ((KV_HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL + (HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL)*HIDDEN_DIM
    );
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> wk_wq_s_sum_mmap(DECODER_LAYER_NUM *  (KV_HIDDEN_DIM + HIDDEN_DIM));

    vector<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>>> wv_wo_mmap(
        DECODER_LAYER_NUM * ((KV_HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL + (HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL)*HIDDEN_DIM
    );
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> wv_wo_s_sum_mmap(DECODER_LAYER_NUM * (KV_HIDDEN_DIM + HIDDEN_DIM));
    
    // MHA
    vector<hls::vector<ap_int<8>, PRE_K_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_K_PARALLEL>>> k_cache(DECODER_LAYER_NUM * KV_HEAD_NUM * MAX_SEQ_LEN/PRE_K_PARALLEL * HEAD_DIM);
    vector<hls::vector<ap_int<8>, PRE_V_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_V_PARALLEL>>> v_cache(DECODER_LAYER_NUM * KV_HIDDEN_DIM/PRE_V_PARALLEL * MAX_SEQ_LEN);

    // FFN
    vector<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>>> w_ffn_gate_mmap(
        DECODER_LAYER_NUM * ((INTER_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * HIDDEN_DIM
    );
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> w_ffn_gate_s_sum_mmap(DECODER_LAYER_NUM * INTER_DIM);
    vector<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>>> w_ffn_up_mmap(
        DECODER_LAYER_NUM * ((INTER_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * HIDDEN_DIM
    );
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> w_ffn_up_s_sum_mmap(DECODER_LAYER_NUM * INTER_DIM);
    vector<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>>> w_ffn_down_mmap(
        DECODER_LAYER_NUM * ((HIDDEN_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * INTER_DIM
    );
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> w_ffn_down_s_sum_mmap(DECODER_LAYER_NUM * HIDDEN_DIM);

    // Layer Norm weight mmap
    vector<float, tapa::aligned_allocator<float>> gamma_beta_mmap_0(DECODER_LAYER_NUM * HIDDEN_DIM);
    vector<float, tapa::aligned_allocator<float>> gamma_beta_mmap_1(DECODER_LAYER_NUM * HIDDEN_DIM);

    // Residual cache
    vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> res0_cache_mmap(MAX_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM);
    vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> res1_cache_mmap(MAX_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM);
    
    // 2) Initialize buffers with random data
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> distF(-1.0f/HIDDEN_DIM, 1.0f/HIDDEN_DIM);
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

    initInt8Vec(wk_wq_mmap, PRE_QKVO_W_PARALLEL_READ/2);
    initFloatVec(wk_wq_s_sum_mmap, 2);
    initInt8Vec(wv_wo_mmap, PRE_QKVO_W_PARALLEL_READ/2);
    initFloatVec(wv_wo_s_sum_mmap, 2);

    initInt8Vec(w_ffn_gate_mmap, PRE_FFN_W_PARALLEL_READ/2);
    initFloatVec(w_ffn_gate_s_sum_mmap, 2);
    initInt8Vec(w_ffn_up_mmap, PRE_FFN_W_PARALLEL_READ/2);
    initFloatVec(w_ffn_up_s_sum_mmap, 2);
    initInt8Vec(w_ffn_down_mmap, PRE_FFN_W_PARALLEL_READ/2);
    initFloatVec(w_ffn_down_s_sum_mmap, 2);

    for (auto &f : gamma_beta_mmap_0) f = distF(rng);
    for (auto &f : gamma_beta_mmap_1) f = distF(rng);

    size_t io_init_vecs = (MAX_SEQ_LEN / TOKEN_PARALLEL) * HIDDEN_DIM;

    const int num_runs = 10;
    int64_t total_time_ns = 0;
    std::cout << "kernel begins running " << num_runs << " times …\n";
    
    for (int run = 0; run < num_runs; ++run) {
    // Call Linear_Layer_tb
    // Apply initialization
        
        // Random init first chunk
        for (size_t idx = 0; idx < io_init_vecs; ++idx) {
            for (int j = 0; j < TOKEN_PARALLEL; ++j) {
                io_mmap[idx][j] = distF(rng);
            }
        }
        // Zero out the remaining vectors
        for (size_t idx = io_init_vecs; idx < io_mmap.size(); ++idx) {
            for (int j = 0; j < TOKEN_PARALLEL; ++j) {
                io_mmap[idx][j] = 0.0f;
            }
        }

        for (auto &vec : k_cache)      for (int i = 0; i < PRE_K_PARALLEL; ++i) vec[i] = 0;
        for (auto &vec : v_cache)      for (int i = 0; i < PRE_V_PARALLEL; ++i) vec[i] = 0;

        for (auto &vec : res0_cache_mmap) for (int i = 0; i < TOKEN_PARALLEL; ++i) vec[i] = 0.0f;
        for (auto &vec : res1_cache_mmap) for (int i = 0; i < TOKEN_PARALLEL; ++i) vec[i] = 0.0f;

        cout << "kernel begins running!\n";
        int64_t kernel_time_ns = tapa::invoke(
            SpinQuant_Prefilling, 
            FLAGS_bitstream,
            tapa::read_write_mmap<hls::vector<float, TOKEN_PARALLEL>>(io_mmap),
            tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>>(wk_wq_mmap),
            tapa::read_only_mmap<hls::vector<float, 2>>(wk_wq_s_sum_mmap),
            tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>>(wv_wo_mmap),
            tapa::read_only_mmap<hls::vector<float, 2>>(wv_wo_s_sum_mmap),
            tapa::read_write_mmap<hls::vector<ap_int<8>, PRE_K_PARALLEL>>(k_cache),
            tapa::read_write_mmap<hls::vector<ap_int<8>, PRE_V_PARALLEL>>(v_cache),
            tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>>(w_ffn_gate_mmap),
            tapa::read_only_mmap<hls::vector<float, 2>>(w_ffn_gate_s_sum_mmap),
            tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>>(w_ffn_up_mmap),
            tapa::read_only_mmap<hls::vector<float, 2>>(w_ffn_up_s_sum_mmap),
            tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>>(w_ffn_down_mmap),
            tapa::read_only_mmap<hls::vector<float, 2>>(w_ffn_down_s_sum_mmap),
            tapa::read_only_mmap<float>(gamma_beta_mmap_0),
            tapa::read_only_mmap<float>(gamma_beta_mmap_1),
            tapa::read_write_mmap<hls::vector<float, TOKEN_PARALLEL>>(res0_cache_mmap),
            tapa::read_write_mmap<hls::vector<float, TOKEN_PARALLEL>>(res1_cache_mmap),
            MAX_SEQ_LEN
        );

        double t_s = kernel_time_ns * 1e-9;
        std::cout << "  Run " << run << " — kernel time: " << t_s << " s\n";
        total_time_ns += kernel_time_ns;

        for(int i = 0; i < 32; i++) std::cout << io_mmap[i][0] << " ";
        std::cout << std::endl;
        for(int i = 0; i < 32; i++) std::cout << io_mmap[io_init_vecs + i][0] << " ";
        std::cout << std::endl;
    }

    double avg_s = (total_time_ns / double(num_runs)) * 1e-9;
    std::cout << "Average kernel time over " << num_runs << " runs: " << avg_s << " s\n";
}

int main(int argc, char* argv[]) {
    SpinQuant_test(argc, argv);
    return 0;
}