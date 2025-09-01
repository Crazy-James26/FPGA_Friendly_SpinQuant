#ifndef _LOGITS_H
#define _LOGITS_H
#include "config.h"

template <typename T, int block_parallel, int max_logits_num, int top_k, bool enable_softmax=true, bool enable_sub_max=true>
void dec_Logits_Max_K_Layer(
    tapa::istream<hls::vector<T, block_parallel>>& logits_stream,
    tapa::ostream<hls::vector<T, 2>>& max_logits_stream,
    int logits_num = max_logits_num
){
    T max_logits_idx[block_parallel][top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits_idx complete
    T max_logits[block_parallel][top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits complete

    T max_logits_idx_final[top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits_idx_final complete
    T max_logits_final[top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits_final complete

    T max_logits_final_exp[top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits_final_exp complete

    init_loop: for (int k = 0; k < top_k; k++) {
    #pragma HLS unroll
        for(int i = 0; i < block_parallel; i++){
        #pragma HLS unroll
            max_logits_idx[i][k] = -1;
            max_logits[i][k] = -1e6;
        }
        max_logits_idx_final[k] = -1;
        max_logits_final[k] = -1e6;
    }

    io_block_loop: for (int k = 0; k < logits_num/block_parallel; k++) {
    #pragma HLS loop_tripcount min=1 max=max_logits_num/block_parallel
        hls::vector<T, block_parallel> temp_pack = logits_stream.read();

        // maintain Top-K in parallel for each lane
        lane_top_k_loop: for (int i = 0; i < block_parallel; i++) {
        #pragma HLS unroll
            T v = temp_pack[i];
            T v_idx = static_cast<T>(int(i * (logits_num/block_parallel) + k));
            // find the position to insert
            int insert_pos = -1;
            for (int k = 0; k < top_k; k++) {
                if (v > max_logits[i][k]) {
                    insert_pos = k;
                    break;
                }
            }
            // insert and shift
            if (insert_pos != -1) {
                for (int k = top_k - 1; k > insert_pos; k--) {
                #pragma HLS unroll
                    max_logits_idx[i][k] = max_logits_idx[i][k - 1];
                    max_logits[i][k] = max_logits[i][k - 1];
                }
                max_logits_idx[i][insert_pos] = v_idx;
                max_logits[i][insert_pos] = v;
            }
        }
    }

    // 4) compute final global Top-K over all candidates
    merge_loop: for (int K = 0; K < top_k; K++) {
        top_k_block_loop: for (int i = 0; i < block_parallel; i++) {
            T val = max_logits[i][K];
            T val_idx = max_logits_idx[i][K];
            int pos = -1;
            for (int k = 0; k < top_k; k++) {
            #pragma HLS unroll
                if (val > max_logits_final[k]) {
                    pos = k;
                    break;
                }
            }
            if (pos != -1) {
                for (int k = top_k - 1; k > pos; k--) {
                #pragma HLS unroll
                    max_logits_idx_final[k] = max_logits_idx_final[k - 1];
                    max_logits_final[k] = max_logits_final[k - 1];
                }
                max_logits_idx_final[pos] = val_idx;
                max_logits_final[pos] = val;
            }
        }
    }

    // 5) apply softmax if enabled
    if (enable_softmax) {
        T sum = 0;
        exp_sum_loop: for (int k = 0; k < top_k; k++) {
        #pragma HLS PIPELINE II = 1
            T temp;
            if(enable_sub_max) 
                temp = exp(max_logits_final[k] - max_logits_final[0]);
            else
                temp = exp(max_logits_final[k]);
            max_logits_final_exp[k] = temp;
            sum += temp;
        }
        exp_scale_loop: for (int k = 0; k < top_k; k++) {
        #pragma HLS PIPELINE II = 1
            max_logits_final_exp[k] /= sum;
        }
    }


    // 5) pack and write out the final Top-K
    hls::vector<T, 2> out_pack;
    write_loop: for (int k = 0; k < top_k; k++) {
    #pragma HLS PIPELINE II = 1
        out_pack[0] = max_logits_idx_final[k];
        out_pack[1] = max_logits_final_exp[k];
        max_logits_stream.write(out_pack);
    }
}


template <typename T, int block_parallel_samp, int max_logits_num, int top_k, int block_parallel_embed, int max_hidden_dim=HIDDEN_DIM, bool enable_sub_max=true>
void dec_Sampling_Embedding_Layer(
    tapa::istream<hls::vector<T, block_parallel_samp>>& logits_stream,
    tapa::mmap<hls::vector<T, block_parallel_embed>> vocab_lib,
    tapa::ostream<hls::vector<T, block_parallel_embed>>& new_embedding_stream,
    T rand_seed, 
    int & sampled_idx,
    int logits_num = max_logits_num,
    int io_hidden_dim = max_hidden_dim
){
    int max_logits_idx[block_parallel_samp][top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits_idx complete
    T max_logits[block_parallel_samp][top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits complete

    int max_logits_idx_final[top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits_idx_final complete
    T max_logits_final[top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits_final complete

    T max_vocab_probs[top_k];
    #pragma HLS ARRAY_PARTITION variable=max_vocab_probs complete

    init_loop: for (int k = 0; k < top_k; k++) {
    #pragma HLS unroll
        for(int i = 0; i < block_parallel_samp; i++){
        #pragma HLS unroll
            max_logits_idx[i][k] = -1;
            max_logits[i][k] = -1e6;
        }
        max_logits_idx_final[k] = -1;
        max_logits_final[k] = -1e6;
    }

    io_block_loop: for (int k = 0; k < logits_num/block_parallel_samp; k++) {
    #pragma HLS loop_tripcount min=1 max=max_logits_num/block_parallel_samp
        hls::vector<T, block_parallel_samp> temp_pack = logits_stream.read();

        // maintain Top-K in parallel for each lane
        lane_top_k_loop: for (int i = 0; i < block_parallel_samp; i++) {
        #pragma HLS unroll
            T v = temp_pack[i];
            int v_idx = i * (logits_num/block_parallel_samp) + k;
            // find the position to insert
            int insert_pos = -1;
            for (int k = 0; k < top_k; k++) {
                if (v > max_logits[i][k]) {
                    insert_pos = k;
                    break;
                }
            }
            // insert and shift
            if (insert_pos != -1) {
                for (int k = top_k - 1; k > insert_pos; k--) {
                #pragma HLS unroll
                    max_logits_idx[i][k] = max_logits_idx[i][k - 1];
                    max_logits[i][k] = max_logits[i][k - 1];
                }
                max_logits_idx[i][insert_pos] = v_idx;
                max_logits[i][insert_pos] = v;
            }
        }
    }

    // 4) compute final global Top-K over all candidates
    merge_loop: for (int K = 0; K < top_k; K++) {
        top_k_block_loop: for (int i = 0; i < block_parallel_samp; i++) {
            T val = max_logits[i][K];
            int val_idx = max_logits_idx[i][K];
            int pos = -1;
            for (int k = 0; k < top_k; k++) {
            #pragma HLS unroll
                if (val > max_logits_final[k]) {
                    pos = k;
                    break;
                }
            }
            if (pos != -1) {
                for (int k = top_k - 1; k > pos; k--) {
                #pragma HLS unroll
                    max_logits_idx_final[k] = max_logits_idx_final[k - 1];
                    max_logits_final[k] = max_logits_final[k - 1];
                }
                max_logits_idx_final[pos] = val_idx;
                max_logits_final[pos] = val;
            }
        }
    }

    for(int i = 0; i < top_k; i++) {
        printf("max_logits_idx_final[%d] = %d, max_logits_final[%d] = %f\n", i, max_logits_idx_final[i], i, max_logits_final[i]);
    }

    // 5) apply softmax
    T sum = 0;
    exp_sum_loop: for (int k = 0; k < top_k; k++) {
    #pragma HLS PIPELINE II = 1
        T temp;
        if(enable_sub_max) 
            temp = exp(max_logits_final[k] - max_logits_final[0]);
        else
            temp = exp(max_logits_final[k]);
        max_vocab_probs[k] = temp;
        sum += temp;
    }
    exp_scale_loop: for (int k = 0; k < top_k; k++) {
    #pragma HLS PIPELINE II = 1
        max_vocab_probs[k] /= sum;
    }
    
    // 5) sample from Top-K according to probabilities
    // Build CDF
    T cdf[top_k];
    #pragma HLS ARRAY_PARTITION variable=cdf complete

    build_cdf: for (int k = 0; k < top_k; ++k) {
    #pragma HLS PIPELINE II = 1
        if(k == 0) 
            cdf[k] = max_vocab_probs[k];
        else 
            cdf[k] = max_vocab_probs[k] + cdf[k - 1];
        printf("k %d, prob %f, cdf %f\n", k, (float)max_vocab_probs[k], cdf[k]);
    }
    printf("rand_seed %f\n", (float)rand_seed);

    int selected_k;
    // Select smallest k with r < cdf[k]
    selected_k = top_k - 1;
    choose_k: for (int k = 0; k < top_k; ++k) {
        // priority-encode the first crossing
        if ((rand_seed < cdf[k]) && (selected_k == top_k - 1)) selected_k = k;
    }
    
    sampled_idx = max_logits_idx_final[selected_k];

    // 6) retrieve embedding from vocab_lib and write to io_mmap
    dec_input_loader<T, block_parallel_embed, max_hidden_dim>(
        vocab_lib, new_embedding_stream, 0, sampled_idx, io_hidden_dim
    );
    printf("selected k %d, sampled token %d\n", selected_k, sampled_idx);
}





#endif