// KV Cache Precision Sweep: find optimal K-bits / V-bits combination
// Tests all combinations on real Gemma4 attention to measure quality
#include "turboquant.h"
#include "mlx/io.h"
#include "mlx/fast.h"
#include <iostream>
#include <unordered_map>
#include <cmath>

using namespace mlx::core;

array bare_rms_norm(const array& x, float eps) {
    return x * rsqrt(mean(x * x, -1, true) + array(eps));
}

int main() {
    turboquant::set_metallib_path(METALLIB_PATH);

    const float rms_eps = 1e-6f;
    const int sliding_hd = 256;

    std::string model_dir = "/Users/andromeda/.cache/huggingface/hub/"
        "models--mlx-community--gemma-4-31b-it-4bit/snapshots/"
        "535c5606372deb5d5ab7e29280f111ef2a8e084e/";

    std::unordered_map<std::string, array> W;
    for (int s = 1; s <= 4; s++) {
        auto [weights, _] = load_safetensors(
            model_dir + "model-0000" + std::to_string(s) + "-of-00004.safetensors");
        for (auto& kv : weights) W.insert_or_assign(kv.first, kv.second);
    }

    auto embed_w = W.at("language_model.model.embed_tokens.weight");
    auto embed_s = W.at("language_model.model.embed_tokens.scales");
    auto embed_b = W.at("language_model.model.embed_tokens.biases");
    auto full_embed = dequantize(embed_w, embed_s, embed_b, 64, 4);
    eval(full_embed);

    // Get real Q, K, V from layer 0 (32 tokens)
    std::vector<int> toks = {2, 9259, 236764, 1217, 659, 611, 236881, 818, 5279, 529, 7001, 563, 3477, 496, 236810, 3941,
                             1613, 236743, 236810, 14929, 15579, 532, 910, 1689, 1161, 3636, 6974, 496, 678, 20517, 1003, 506};
    auto idx = array(toks.data(), {32}, int32);
    auto emb = reshape(take(full_embed, idx, 0), {1, 32, 5376}) * array(std::sqrt(5376.0f));

    std::string L = "language_model.model.layers.0.";
    auto h = fast::rms_norm(emb, W.at(L+"input_layernorm.weight"), rms_eps);

    auto qmm = [&](const array& input, const std::string& key) -> array {
        std::optional<array> biases = W.at(key + ".biases");
        return quantized_matmul(input, W.at(key+".weight"), W.at(key+".scales"), biases, true, 64, 4);
    };

    auto q = qmm(h, L+"self_attn.q_proj");
    auto k = qmm(h, L+"self_attn.k_proj");
    auto v = qmm(h, L+"self_attn.v_proj");

    // Reshape to heads
    q = transpose(reshape(q, {1, 32, 32, 256}), {0,2,1,3});   // (1,32,32,256)
    k = transpose(reshape(k, {1, 32, 16, 256}), {0,2,1,3});   // (1,16,32,256)
    v = transpose(reshape(v, {1, 32, 16, 256}), {0,2,1,3});   // (1,16,32,256)

    // Norms
    q = fast::rms_norm(q, W.at(L+"self_attn.q_norm.weight"), rms_eps);
    k = fast::rms_norm(k, W.at(L+"self_attn.k_norm.weight"), rms_eps);
    v = bare_rms_norm(v, rms_eps);

    // RoPE
    q = fast::rope(q, 256, false, 10000.0f, 1.0f, 0);
    k = fast::rope(k, 256, false, 10000.0f, 1.0f, 0);
    eval(q, k, v);

    // Reference attention (FP32)
    auto ref_attn = fast::scaled_dot_product_attention(q, k, v, 1.0f, "causal");
    eval(ref_attn);

    // Cos similarity between two attention outputs
    auto cos_sim = [](const array& a, const array& b) -> float {
        auto af = reshape(a, {-1});
        auto bf = reshape(b, {-1});
        auto d = sum(af * bf);
        auto na = sqrt(sum(af * af));
        auto nb = sqrt(sum(bf * bf));
        eval(d, na, nb);
        return d.item<float>() / (na.item<float>() * nb.item<float>() + 1e-10f);
    };

    printf("%-12s %-12s %-10s %-10s %-12s\n", "K_bits", "V_bits", "Attn_CS", "BPD", "Compress");
    printf("%-12s %-12s %-10s %-10s %-12s\n", "------", "------", "-------", "---", "--------");

    // Test combinations
    int k_bits_list[] = {4, 8};
    int v_bits_list[] = {4, 8};

    for (int kb : k_bits_list) {
        for (int vb : v_bits_list) {
            // Quantize K
            auto kq = quantize(k, 64, kb);
            auto k_recon = dequantize(kq[0], kq[1], kq[2], 64, kb);

            // Quantize V
            auto vq = quantize(v, 64, vb);
            auto v_recon = dequantize(vq[0], vq[1], vq[2], 64, vb);

            // Attention with quantized KV
            auto attn = fast::scaled_dot_product_attention(q, k_recon, v_recon, 1.0f, "causal");
            eval(attn);

            float cs = cos_sim(ref_attn, attn);
            float k_bpd = kb + 2.0f * 32.0f / 64.0f;  // bits + overhead
            float v_bpd = vb + 2.0f * 32.0f / 64.0f;
            float avg_bpd = (k_bpd + v_bpd) / 2.0f;
            float compress = 64.0f / (k_bpd + v_bpd);  // vs FP32 (32 bits each for K+V)

            printf("int%-9d int%-9d %-10.6f %-10.2f %-12.1fx\n",
                kb, vb, cs, avg_bpd, compress);
        }
    }

    // Also test FP32 K + quantized V only (V is less sensitive)
    printf("\n--- V-only quantization ---\n");
    for (int vb : {4, 8}) {
        auto vq = quantize(v, 64, vb);
        auto v_recon = dequantize(vq[0], vq[1], vq[2], 64, vb);
        auto attn = fast::scaled_dot_product_attention(q, k, v_recon, 1.0f, "causal");
        eval(attn);
        float cs = cos_sim(ref_attn, attn);
        printf("FP32        int%-9d %-10.6f\n", vb, cs);
    }

    // K-only quantization
    printf("\n--- K-only quantization ---\n");
    for (int kb : {4, 8}) {
        auto kq = quantize(k, 64, kb);
        auto k_recon = dequantize(kq[0], kq[1], kq[2], 64, kb);
        auto attn = fast::scaled_dot_product_attention(q, k_recon, v, 1.0f, "causal");
        eval(attn);
        float cs = cos_sim(ref_attn, attn);
        printf("int%-9d FP32        %-10.6f\n", kb, cs);
    }

    return 0;
}
