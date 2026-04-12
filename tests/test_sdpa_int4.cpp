// Test sdpa_int4 Metal kernel correctness
// Compares fused kernel output vs dequantize + manual SDPA
#include "turboquant.h"
#include <iostream>
#include <cmath>

using namespace mlx::core;

int main() {
    turboquant::set_metallib_path(METALLIB_PATH);

    // Test params matching a sliding attention layer
    int nh = 16, nkv = 16, N = 10, hd = 256;
    int gqa = nh / nkv;
    float scale = 1.0f;

    printf("Testing sdpa_int4: nh=%d nkv=%d N=%d hd=%d gqa=%d\n", nh, nkv, N, hd, gqa);

    // Create test data
    auto Q = random::normal({nh, hd});   // (nh, hd)
    auto K = random::normal({nkv, N, hd});
    auto V = random::normal({nkv, N, hd});

    // Quantize K and V
    auto kqv = quantize(K, 64, 4);
    auto vqv = quantize(V, 64, 4);
    auto Kq = kqv[0], Ks = kqv[1], Kb = kqv[2];
    auto Vq = vqv[0], Vs = vqv[1], Vb = vqv[2];

    // === Reference: dequantize + manual SDPA ===
    auto K_deq = dequantize(Kq, Ks, Kb, 64, 4);  // (nkv, N, hd)
    auto V_deq = dequantize(Vq, Vs, Vb, 64, 4);

    // Compute attention manually per head
    // Q: (nh, hd), K_deq: (nkv, N, hd), V_deq: (nkv, N, hd)
    // For gqa=1: head h uses kv head h
    // scores[h] = Q[h] @ K_deq[h]^T = (1, hd) @ (hd, N) = (1, N)
    auto Q_4d = reshape(Q, {1, nh, 1, hd});
    auto K_4d = reshape(K_deq, {1, nkv, N, hd});
    auto V_4d = reshape(V_deq, {1, nkv, N, hd});
    auto ref_attn = fast::scaled_dot_product_attention(Q_4d, K_4d, V_4d, scale, "");
    auto ref = reshape(ref_attn, {nh, hd});

    // === Fused kernel ===
    auto fused = turboquant::sdpa_int4(Q, Kq, Ks, Kb, Vq, Vs, Vb, gqa, scale, 0);

    eval(ref, fused);

    // Compare
    auto diff = abs(ref - fused);
    auto max_err = max(diff).item<float>();
    auto mean_err = mean(diff).item<float>();
    auto ref_norm = sqrt(sum(ref * ref)).item<float>();
    auto fused_norm = sqrt(sum(fused * fused)).item<float>();

    printf("Reference norm:  %.4f\n", ref_norm);
    printf("Fused norm:      %.4f\n", fused_norm);
    printf("Max error:       %.6f\n", max_err);
    printf("Mean error:      %.6f\n", mean_err);
    printf("Relative error:  %.6f\n", max_err / ref_norm);

    // Print first few values
    eval(ref, fused);
    printf("\nRef  [0, :8]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", slice(ref, {0,i}, {1,i+1}).item<float>());
    printf("\nFused[0, :8]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", slice(fused, {0,i}, {1,i+1}).item<float>());
    printf("\n");

    // Print head 15 too
    printf("Ref  [15, :8]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", slice(ref, {15,i}, {16,i+1}).item<float>());
    printf("\nFused[15, :8]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", slice(fused, {15,i}, {16,i+1}).item<float>());
    printf("\n");

    bool pass = max_err < 0.1f;
    printf("\n%s (max_err=%.6f)\n", pass ? "PASS" : "FAIL", max_err);
    return pass ? 0 : 1;
}
