#include "turboquant.h"
#include <iostream>
using namespace mlx::core;
int main() {
    turboquant::set_metallib_path(METALLIB_PATH);
    int nh = 8, nkv = 4, N = 4, hd = 512;
    int gqa = nh / nkv;
    printf("D=512 GQA=%d N=%d\n", gqa, N);
    auto Q = random::normal({nh, hd});
    auto K = random::normal({nkv, N, hd});
    auto V = random::normal({nkv, N, hd});
    auto kqv = quantize(K, 64, 4), vqv = quantize(V, 64, 4);
    auto K_deq = dequantize(kqv[0], kqv[1], kqv[2], 64, 4);
    auto V_deq = dequantize(vqv[0], vqv[1], vqv[2], 64, 4);
    auto ref = reshape(fast::scaled_dot_product_attention(
        reshape(Q, {1,nh,1,hd}), reshape(K_deq, {1,nkv,N,hd}),
        reshape(V_deq, {1,nkv,N,hd}), 1.0f, ""), {nh, hd});
    auto fused = turboquant::sdpa_int4(Q, kqv[0], kqv[1], kqv[2], vqv[0], vqv[1], vqv[2], gqa, 1.0f, 0);
    eval(ref, fused);
    
    printf("Head 0, elems 0-7:\n");
    printf("  Ref:   "); for(int i=0;i<8;i++) printf("%.3f ", slice(ref,{0,i},{1,i+1}).item<float>()); printf("\n");
    printf("  Fused: "); for(int i=0;i<8;i++) printf("%.3f ", slice(fused,{0,i},{1,i+1}).item<float>()); printf("\n");
    printf("Head 0, elems 248-256:\n");
    printf("  Ref:   "); for(int i=248;i<256;i++) printf("%.3f ", slice(ref,{0,i},{1,i+1}).item<float>()); printf("\n");
    printf("  Fused: "); for(int i=248;i<256;i++) printf("%.3f ", slice(fused,{0,i},{1,i+1}).item<float>()); printf("\n");
    printf("Head 0, elems 504-512:\n");
    printf("  Ref:   "); for(int i=504;i<512;i++) printf("%.3f ", slice(ref,{0,i},{1,i+1}).item<float>()); printf("\n");
    printf("  Fused: "); for(int i=504;i<512;i++) printf("%.3f ", slice(fused,{0,i},{1,i+1}).item<float>()); printf("\n");
}
