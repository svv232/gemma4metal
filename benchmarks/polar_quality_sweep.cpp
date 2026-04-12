// PolarQuant Quality Sweep on Real Gemma 4 K Vectors
// Tests different codebook sizes to find the sweet spot
#include "turboquant.h"
#include "mlx/io.h"
#include "mlx/fast.h"
#include <iostream>
#include <chrono>
#include <unordered_map>
#include <cmath>

using namespace mlx::core;

// Compute Lloyd-Max codebook for level 1: uniform on [0, 2pi)
array make_cb_l1(int n) {
    std::vector<float> vals(n);
    for (int i = 0; i < n; i++) vals[i] = (2.0f * M_PI * i / n) + M_PI / n;
    return array(vals.data(), {n}, float32);
}

// Compute codebook for levels 2+: concentrated around pi/4
// Using empirical quantiles of the beta distribution for d=16
array make_cb_higher(int n, int level) {
    // For block_size=16, dimension at each level decreases
    // Level 2: pairs of 8-dim blocks → angles near pi/4
    // Level 3: pairs of 4-dim blocks
    // Level 4: pairs of 2-dim blocks
    // Approximate optimal codebook centroids for different sizes
    std::vector<float> vals;
    if (n == 4) {
        // 2-bit: standard centroids
        if (level == 2) vals = {0.31f, 0.63f, 0.94f, 1.26f};
        else if (level == 3) vals = {0.43f, 0.67f, 0.90f, 1.14f};
        else vals = {0.52f, 0.71f, 0.86f, 1.05f};
    } else if (n == 8) {
        // 3-bit: finer quantization
        if (level == 2) vals = {0.15f, 0.38f, 0.55f, 0.71f, 0.87f, 1.03f, 1.20f, 1.42f};
        else if (level == 3) vals = {0.25f, 0.45f, 0.60f, 0.73f, 0.85f, 0.97f, 1.10f, 1.30f};
        else vals = {0.35f, 0.52f, 0.64f, 0.75f, 0.84f, 0.94f, 1.05f, 1.20f};
    } else if (n == 16) {
        // 4-bit: very fine
        for (int i = 0; i < 16; i++) {
            if (level == 2) vals.push_back(0.08f + i * 1.49f / 15.0f);  // spread [0.08, 1.57]
            else if (level == 3) vals.push_back(0.15f + i * 1.35f / 15.0f);
            else vals.push_back(0.25f + i * 1.10f / 15.0f);
        }
    } else {
        // Uniform fallback
        for (int i = 0; i < n; i++)
            vals.push_back(0.1f + i * 1.47f / (n - 1));
    }
    return array(vals.data(), {n}, float32);
}

float cos_sim(const array& a, const array& b) {
    auto dot = sum(a * b);
    auto na = sqrt(sum(a * a));
    auto nb = sqrt(sum(b * b));
    eval(dot, na, nb);
    return dot.item<float>() / (na.item<float>() * nb.item<float>() + 1e-10f);
}

int main() {
    turboquant::set_metallib_path(METALLIB_PATH);

    std::string model_dir = "/Users/andromeda/.cache/huggingface/hub/"
        "models--mlx-community--gemma-4-31b-it-4bit/snapshots/"
        "535c5606372deb5d5ab7e29280f111ef2a8e084e/";

    std::unordered_map<std::string, array> W;
    for (int s = 1; s <= 4; s++) {
        auto [weights, _] = load_safetensors(
            model_dir + "model-0000" + std::to_string(s) + "-of-00004.safetensors");
        for (auto& kv : weights) W.insert_or_assign(kv.first, kv.second);
    }

    // Get real K vectors from layer 0 (sliding, hd=256)
    auto embed_w = W.at("language_model.model.embed_tokens.weight");
    auto embed_s = W.at("language_model.model.embed_tokens.scales");
    auto embed_b = W.at("language_model.model.embed_tokens.biases");
    auto full_embed = dequantize(embed_w, embed_s, embed_b, 64, 4);
    eval(full_embed);

    // Create input: 32 token embeddings
    std::vector<int> toks = {2, 9259, 236764, 1217, 659, 611, 236881, 818, 5279, 529, 7001, 563, 3477, 496, 236810, 3941,
                             1613, 236743, 236810, 14929, 15579, 532, 910, 1689, 1161, 3636, 6974, 496, 678, 20517, 1003, 506};
    auto idx = array(toks.data(), {32}, int32);
    auto emb = reshape(take(full_embed, idx, 0), {1, 32, 5376});
    emb = emb * array(std::sqrt(5376.0f));

    // Project through layer 0 K
    auto h = fast::rms_norm(emb, W.at("language_model.model.layers.0.input_layernorm.weight"), 1e-6f);
    std::optional<array> kb = W.at("language_model.model.layers.0.self_attn.k_proj.biases");
    auto k_proj = quantized_matmul(h, W.at("language_model.model.layers.0.self_attn.k_proj.weight"),
        W.at("language_model.model.layers.0.self_attn.k_proj.scales"), kb, true, 64, 4);
    // Reshape: (1, 32, 4096) → (1, 16, 32, 256) → flatten to (512, 256)
    k_proj = reshape(k_proj, {-1, 256});
    // K norm
    auto k_normed = fast::rms_norm(reshape(k_proj, {1, 16, 32, 256}),
        W.at("language_model.model.layers.0.self_attn.k_norm.weight"), 1e-6f);
    auto k_flat = reshape(k_normed, {-1, 256});  // (512, 256)
    eval(k_flat);

    printf("K vectors: [%d, %d]\n", (int)k_flat.shape(0), (int)k_flat.shape(1));

    auto P = eye(256);
    eval(P);

    // Sweep configurations
    struct Config {
        int l1_bits, l2_bits, l3_bits, l4_bits;
        float bits_per_dim() const {
            // For block_size=16: 8 L1 angles, 4 L2, 2 L3, 1 L4
            // Total bits = 8*l1 + 4*l2 + 2*l3 + 1*l4 + 16 bits radius (FP16)
            // Per dim = total / 16
            return (8.0f * l1_bits + 4.0f * l2_bits + 2.0f * l3_bits + 1.0f * l4_bits + 16.0f) / 16.0f;
        }
        float compression() const {
            // vs FP32 = 32 bits/dim
            return 32.0f / bits_per_dim();
        }
    };

    std::vector<Config> configs = {
        {4, 2, 2, 2},  // Current: 10 bits angles + 16 bits radius = 1.625 bpd
        {5, 2, 2, 2},  // 18+16 = 2.125 bpd
        {6, 3, 3, 3},  // 36+16 = 3.25 bpd
        {8, 4, 4, 4},  // 64+16 = 5.0 bpd
        {4, 3, 3, 3},  // 17+16 = 2.0625 bpd
        {5, 3, 3, 3},  // 22+16 = 2.375 bpd
        {6, 2, 2, 2},  // 14+16 = 1.875 bpd
        {8, 2, 2, 2},  // 18+16 = 2.125 bpd (same as 5+2+2+2 by coincidence)
    };

    printf("\n%-12s %-8s %-8s %-10s\n", "Config", "BPD", "Compress", "Cos_Sim");
    printf("%-12s %-8s %-8s %-10s\n", "------", "---", "--------", "-------");

    for (auto& cfg : configs) {
        int n1 = 1 << cfg.l1_bits;
        int n2 = 1 << cfg.l2_bits;
        int n3 = 1 << cfg.l3_bits;
        int n4 = 1 << cfg.l4_bits;

        auto cb1 = make_cb_l1(n1);
        auto cb2 = make_cb_higher(n2, 2);
        auto cb3 = make_cb_higher(n3, 3);
        auto cb4 = make_cb_higher(n4, 4);
        eval(cb1, cb2, cb3, cb4);

        // Quantize
        auto result = turboquant::polar_quantize(k_flat, P, cb1, cb2, cb3, cb4);
        // result: [indices_l1, indices_l2, indices_l3, indices_l4, radii]

        // Dequantize
        auto recon = turboquant::polar_dequantize(
            result[0], result[1], result[2], result[3], result[4],
            P, cb1, cb2, cb3, cb4);
        eval(recon);

        // Measure quality
        float cs = cos_sim(k_flat, recon);

        char name[32];
        snprintf(name, sizeof(name), "%d+%d+%d+%d", cfg.l1_bits, cfg.l2_bits, cfg.l3_bits, cfg.l4_bits);
        printf("%-12s %-8.2f %-8.1fx %-10.4f\n", name, cfg.bits_per_dim(), cfg.compression(), cs);
    }

    // Also test int4 and int8 for comparison
    printf("\n--- MLX native quantization ---\n");
    for (int bits : {4, 8}) {
        auto qv = quantize(reshape(k_flat, {1, 1, (int)k_flat.shape(0), 256}), 64, bits);
        auto recon = dequantize(qv[0], qv[1], qv[2], 64, bits);
        recon = reshape(recon, k_flat.shape());
        eval(recon);
        float cs = cos_sim(k_flat, recon);
        float bpd = bits + 2.0f * 32.0f / 64.0f;  // bits + (scale+bias as float32 per group)
        printf("int%-9d %-8.2f %-8.1fx %-10.4f\n", bits, bpd, 32.0f / bpd, cs);
    }

    return 0;
}
