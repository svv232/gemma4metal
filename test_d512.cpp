#include "turboquant.h"
#include <iostream>
using namespace mlx::core;
int main() {
    turboquant::set_metallib_path(METALLIB_PATH);
    int nh = 32, nkv = 4, N = 10, hd = 512;
    int gqa = nh / nkv;
    printf("Test D=512 GQA=%d: nh=%d nkv=%d\n", gqa, nh, nkv);
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
    auto max_err = max(abs(ref - fused)).item<float>();
    printf("Max error: %.6f\n", max_err);
    printf("Ref[0,:4]:   %.4f %.4f %.4f %.4f\n",
        slice(ref,{0,0},{1,1}).item<float>(), slice(ref,{0,1},{1,2}).item<float>(),
        slice(ref,{0,2},{1,3}).item<float>(), slice(ref,{0,3},{1,4}).item<float>());
    printf("Fused[0,:4]: %.4f %.4f %.4f %.4f\n",
        slice(fused,{0,0},{1,1}).item<float>(), slice(fused,{0,1},{1,2}).item<float>(),
        slice(fused,{0,2},{1,3}).item<float>(), slice(fused,{0,3},{1,4}).item<float>());
    printf("%s\n", max_err < 0.1f ? "PASS" : "FAIL");
}
