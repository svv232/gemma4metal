#include "turboquant.h"
#include <iostream>
using namespace mlx::core;
int main() {
    turboquant::set_metallib_path(METALLIB_PATH);
    // Test various configs
    struct TC { int nh, nkv, N, hd; };
    TC tests[] = {
        {32, 4, 10, 512},   // global, small N
        {32, 4, 100, 512},  // global, medium N
        {32, 16, 10, 256},  // sliding, small N
        {16, 16, 10, 256},  // GQA=1
        {8, 4, 10, 512},    // GQA=2, D=512
    };
    for (auto& t : tests) {
        int gqa = t.nh / t.nkv;
        auto Q = random::normal({t.nh, t.hd});
        auto K = random::normal({t.nkv, t.N, t.hd});
        auto V = random::normal({t.nkv, t.N, t.hd});
        auto kqv = quantize(K, 64, 4), vqv = quantize(V, 64, 4);
        auto K_deq = dequantize(kqv[0], kqv[1], kqv[2], 64, 4);
        auto V_deq = dequantize(vqv[0], vqv[1], vqv[2], 64, 4);
        auto ref = reshape(fast::scaled_dot_product_attention(
            reshape(Q, {1,t.nh,1,t.hd}), reshape(K_deq, {1,t.nkv,t.N,t.hd}),
            reshape(V_deq, {1,t.nkv,t.N,t.hd}), 1.0f, ""), {t.nh, t.hd});
        auto fused = turboquant::sdpa_int4(Q, kqv[0], kqv[1], kqv[2], vqv[0], vqv[1], vqv[2], gqa, 1.0f, 0);
        eval(ref, fused);
        auto max_err = max(abs(ref - fused)).item<float>();
        auto ref_norm = sqrt(sum(ref*ref)).item<float>();
        printf("nh=%2d nkv=%d N=%3d hd=%3d gqa=%d  max_err=%.6f ref_norm=%.1f  %s\n",
            t.nh, t.nkv, t.N, t.hd, gqa, max_err, ref_norm,
            max_err < 0.1f ? "PASS" : "FAIL");
    }
}
