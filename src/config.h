#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <tapa.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_vector.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdint>

using namespace std;


//llama3.2 1B
#define DECODER_LAYER_NUM 16
#define MAX_PRE_SEQ_LEN 1024
#define MAX_DEC_SEQ_LEN 1024
#define MAX_SUM_SEQ_LEN (MAX_PRE_SEQ_LEN + MAX_DEC_SEQ_LEN)
#define log2_MAX_PRE_SEQ_LEN 10
#define log2_MAX_SUM_SEQ_LEN 11

#define HIDDEN_DIM 2048
#define log2_HIDDEN_DIM 11

#define KV_HIDDEN_DIM 512
#define log2_KV_HIDDEN_DIM 9

#define HEAD_DIM 64
#define log2_HEAD_DIM 6
#define sqrt_HEAD_DIM 8

#define Q_HEAD_NUM (HIDDEN_DIM/HEAD_DIM)
#define KV_HEAD_NUM (KV_HIDDEN_DIM/HEAD_DIM)
#define ATTN_GROUP_NUM (HIDDEN_DIM/KV_HIDDEN_DIM)


#define INTER_DIM 8192
#define log2_INTER_DIM 13

#define VOCAB_SIZE 128256

// U280
// Prefilling config
// //small config
// constexpr int  TOKEN_PARALLEL = 8;
// constexpr int  PRE_QKVO_W_PARALLEL = 16;
// constexpr int  PRE_K_PARALLEL = 16;
// constexpr int  PRE_V_PARALLEL = 16;
// constexpr int  PRE_FFN_W_PARALLEL = 64;
// constexpr int  PRE_FFN_W_BLOCK_NUM = 2;

// big config
constexpr int  TOKEN_PARALLEL = 8;
constexpr int  PRE_QKVO_W_PARALLEL = 24;
constexpr int  PRE_K_PARALLEL = 16;
constexpr int  PRE_V_PARALLEL = 16;
constexpr int  PRE_FFN_W_PARALLEL = 96;
constexpr int  PRE_FFN_W_BLOCK_NUM = 3;


// // Decoding config
// constexpr int  T_BLOCK_PARALLEL = 4;
// constexpr int  T_QKVO_FFN_BLOCK_PARALLEL = 8;
// constexpr int  DEC_HEAD_PARALLEL = 4;

// // small config
// constexpr int  DEC_QKVO_FFN_W_PARALLEL = 64;
// constexpr int  DEC_K_PARALLEL = 32;
// constexpr int  DEC_V_PARALLEL = 32;

// // big config
// constexpr int  DEC_QKVO_FFN_W_PARALLEL = 128;
// constexpr int  DEC_K_PARALLEL = 32;
// constexpr int  DEC_V_PARALLEL = 32;

// mem_opt 
constexpr int  T_BLOCK_PARALLEL = 4;
constexpr int  T_QKVO_FFN_BLOCK_PARALLEL = 16;
constexpr int  DEC_HEAD_PARALLEL = (T_QKVO_FFN_BLOCK_PARALLEL/2);

// // small config
// constexpr int  DEC_QKVO_FFN_W_PARALLEL = 32;
// constexpr int  DEC_K_PARALLEL = (DEC_QKVO_FFN_W_PARALLEL/2);
// constexpr int  DEC_V_PARALLEL = (DEC_QKVO_FFN_W_PARALLEL/2);

// big config
constexpr int  DEC_QKVO_FFN_W_PARALLEL = 64;
constexpr int  DEC_K_PARALLEL = (DEC_QKVO_FFN_W_PARALLEL/2);
constexpr int  DEC_V_PARALLEL = (DEC_QKVO_FFN_W_PARALLEL/2);


// // // // V80
// // // Prefilling config
// // constexpr int  TOKEN_PARALLEL = 8;
// // constexpr int  PRE_QKVO_W_PARALLEL = 32;
// // constexpr int  PRE_K_PARALLEL = 32;
// // constexpr int  PRE_V_PARALLEL = 32;
// // constexpr int  PRE_FFN_W_PARALLEL = 128;
// // constexpr int  PRE_FFN_W_BLOCK_NUM = 4;

// // // Decoding config
// // constexpr int  T_BLOCK_PARALLEL = 8;
// // constexpr int  T_QKVO_FFN_BLOCK_PARALLEL = 32;
// // constexpr int  DEC_HEAD_PARALLEL = 8;

// // constexpr int  DEC_QKVO_FFN_W_PARALLEL = 64;
// // constexpr int  DEC_K_PARALLEL = 32;
// // constexpr int  DEC_V_PARALLEL = 32;




// //toy test
// #define DECODER_LAYER_NUM 2
// #define MAX_PRE_SEQ_LEN 64
// #define MAX_DEC_SEQ_LEN 64
// #define MAX_SUM_SEQ_LEN (MAX_PRE_SEQ_LEN + MAX_DEC_SEQ_LEN)
// #define log2_MAX_PRE_SEQ_LEN 6
// #define log2_MAX_SUM_SEQ_LEN 7

// #define HIDDEN_DIM 512
// #define log2_HIDDEN_DIM 9

// #define KV_HIDDEN_DIM 128
// #define log2_KV_HIDDEN_DIM 7

// #define HEAD_DIM 16
// #define log2_HEAD_DIM 4
// #define sqrt_HEAD_DIM 4

// #define Q_HEAD_NUM (HIDDEN_DIM/HEAD_DIM)
// #define KV_HEAD_NUM (KV_HIDDEN_DIM/HEAD_DIM)
// #define ATTN_GROUP_NUM (HIDDEN_DIM/KV_HIDDEN_DIM)


// #define INTER_DIM 2048
// #define log2_INTER_DIM 11

// #define VOCAB_SIZE 3800

// constexpr int  TOKEN_PARALLEL = 8;
// constexpr int  PRE_QKVO_W_PARALLEL = 8;
// constexpr int  PRE_K_PARALLEL = 8;
// constexpr int  PRE_V_PARALLEL = 8;
// constexpr int  PRE_FFN_W_PARALLEL = 32;
// constexpr int  PRE_FFN_W_BLOCK_NUM = 2;

// // constexpr int  TOKEN_PARALLEL = 8;
// // constexpr int  PRE_QKVO_W_PARALLEL = 12;
// // constexpr int  PRE_K_PARALLEL = 8;
// // constexpr int  PRE_V_PARALLEL = 8;
// // constexpr int  PRE_FFN_W_PARALLEL = 48;
// // constexpr int  PRE_FFN_W_BLOCK_NUM = 3;


// constexpr int  T_BLOCK_PARALLEL = 4;
// constexpr int  T_QKVO_FFN_BLOCK_PARALLEL = 8;
// constexpr int  DEC_HEAD_PARALLEL = 4;

// constexpr int  DEC_QKVO_FFN_W_PARALLEL = 32;
// constexpr int  DEC_K_PARALLEL = 16;
// constexpr int  DEC_V_PARALLEL = 16;

// // constexpr int  T_BLOCK_PARALLEL = 4;
// // constexpr int  T_QKVO_FFN_BLOCK_PARALLEL = 16;
// // constexpr int  DEC_HEAD_PARALLEL = 8;

// // constexpr int  DEC_QKVO_FFN_W_PARALLEL = 16;
// // constexpr int  DEC_K_PARALLEL = 8;
// // constexpr int  DEC_V_PARALLEL = 8;


constexpr int  PRE_QKVO_W_PARALLEL_READ = (PRE_QKVO_W_PARALLEL <= 1 ? 1 : PRE_QKVO_W_PARALLEL <= 2 ? 2 :  PRE_QKVO_W_PARALLEL <= 4 ? 4 :
                                            PRE_QKVO_W_PARALLEL <= 8 ? 8 : PRE_QKVO_W_PARALLEL <= 16 ? 16 : PRE_QKVO_W_PARALLEL <= 32 ? 32 : 
                                            PRE_QKVO_W_PARALLEL <= 64 ? 64 : PRE_QKVO_W_PARALLEL <= 128 ? 128 : PRE_QKVO_W_PARALLEL <= 256 ? 256 :
                                            PRE_QKVO_W_PARALLEL <= 512 ? 512 : 1024);

constexpr int  PRE_FFN_W_PARALLEL_READ = (PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM <= 1 ? 1 : PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM <= 2 ? 2 :  PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM <= 4 ? 4 :
                                            PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM <= 8 ? 8 : PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM <= 16 ? 16 : PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM <= 32 ? 32 : 
                                            PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM <= 64 ? 64 : PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM <= 128 ? 128 : PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM <= 256 ? 256 :
                                            PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM <= 512 ? 512 : 1024) * PRE_FFN_W_BLOCK_NUM;

#define LOGITS_MAX_K 5


#endif