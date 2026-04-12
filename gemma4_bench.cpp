// Gemma 4 31B KV Cache Compression Benchmark
// Tests FP32 baseline vs int4/int8 quantized KV cache quality
#include "turboquant.h"
#include "mlx/io.h"
#include "mlx/fast.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <functional>

using namespace mlx::core;

struct Timer {
    std::chrono::high_resolution_clock::time_point s;
    Timer() : s(std::chrono::high_resolution_clock::now()) {}
    double ms() { return std::chrono::duration<double,std::milli>(
        std::chrono::high_resolution_clock::now()-s).count(); }
    void reset() { s = std::chrono::high_resolution_clock::now(); }
};

std::vector<std::string> load_vocab(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    uint32_t size; f.read((char*)&size, 4);
    std::vector<std::string> vocab(size);
    for (uint32_t i = 0; i < size; i++) {
        uint16_t len; f.read((char*)&len, 2);
        vocab[i].resize(len);
        f.read(vocab[i].data(), len);
    }
    return vocab;
}

array bare_rms_norm(const array& x, float eps) {
    return x * rsqrt(mean(x * x, -1, true) + array(eps));
}

int main() {
    turboquant::set_metallib_path(METALLIB_PATH);

    const int hidden_size = 5376;
    const int n_layers = 60;
    const float rms_eps = 1e-6f;
    const float final_softcap = 30.0f;
    const float attn_scale = 1.0f;
    const int sliding_hd = 256, global_hd = 512;
    const int sliding_nkv = 16, global_nkv = 4;
    const int sliding_window = 1024;
    auto is_global = [](int l) { return (l % 6) == 5; };

    // Global RoPE frequencies
    std::vector<float> gf(64);
    for (int i = 0; i < 64; i++) gf[i] = 1.0f / std::pow(1000000.0f, 2.0f * i / 512.0f);
    auto global_rope_freqs = array(gf.data(), {64}, float32);
    eval(global_rope_freqs);

    std::string model_dir = "/Users/andromeda/.cache/huggingface/hub/"
        "models--mlx-community--gemma-4-31b-it-4bit/snapshots/"
        "535c5606372deb5d5ab7e29280f111ef2a8e084e/";

    auto vocab = load_vocab("vocab.bin");
    std::unordered_map<std::string, array> W;
    for (int s = 1; s <= 4; s++) {
        auto [weights, _] = load_safetensors(
            model_dir + "model-0000" + std::to_string(s) + "-of-00004.safetensors");
        for (auto& kv : weights) W.insert_or_assign(kv.first, kv.second);
    }
    printf("Loaded %zu tensors\n", W.size());

    auto has = [&](const std::string& k) { return W.find(k) != W.end(); };
    auto qmm = [&](const array& input, const std::string& key) -> array {
        auto wit = W.find(key + ".weight");
        if (wit == W.end()) return input;
        std::optional<array> biases = W.at(key + ".biases");
        return quantized_matmul(input, wit->second, W.at(key+".scales"), biases, true, 64, 4);
    };

    auto embed_w = W.at("language_model.model.embed_tokens.weight");
    auto embed_s = W.at("language_model.model.embed_tokens.scales");
    auto embed_b = W.at("language_model.model.embed_tokens.biases");
    auto final_norm = W.at("language_model.model.norm.weight");
    auto full_embed = dequantize(embed_w, embed_s, embed_b, 64, 4);
    eval(full_embed, final_norm);

    // Test prompts
    struct TestCase {
        const char* name;
        std::vector<int> tokens;
    };
    std::vector<TestCase> tests = {
        {"What is 15*23?", {2, 236820, 3041, 236779, 1340, 236779, 887, 236813, 2364, 107, 3689, 563, 236743, 236770, 236810, 808, 236743, 236778, 236800, 145912, 643, 236779, 1340, 236779, 887, 236813, 107, 236820, 3041, 236779, 1340, 236779, 887, 236813, 4368, 107}},
        {"Haiku about ocean", {2, 236820, 3041, 236779, 1340, 236779, 887, 236813, 2364, 107, 6974, 496, 678, 20517, 1003, 506, 12461, 21603, 643, 236779, 1340, 236779, 887, 236813, 107, 236820, 3041, 236779, 1340, 236779, 887, 236813, 4368, 107}},
        {"5 prog languages", {2, 236820, 3041, 236779, 1340, 236779, 887, 236813, 2364, 107, 1613, 236743, 236810, 14929, 15579, 532, 910, 1689, 1161, 3636, 21603, 643, 236779, 1340, 236779, 887, 236813, 107, 236820, 3041, 236779, 1340, 236779, 887, 236813, 4368, 107}},
    };

    // Mode: 0=FP32, 3=int8, 4=int4
    int modes[] = {0, 4};
    const char* mode_names[] = {"FP32", "", "", "int8", "int4"};

    for (auto& test : tests) {
        printf("\n========================================\n");
        printf("PROMPT: %s\n", test.name);
        printf("========================================\n");

        for (int mode : modes) {
            // KV cache structures
            struct KVC { std::optional<array> k, v; };
            struct QC { std::optional<array> kq, ks, kb, vq, vs, vb; };
            std::vector<KVC> kv(n_layers);
            std::vector<QC> q8(n_layers);
            int offset = 0;

            auto run_layer = [&](const array& x, int l) -> array {
                std::string L = "language_model.model.layers." + std::to_string(l) + ".";
                bool global = is_global(l);
                int hd = W.at(L+"self_attn.k_norm.weight").shape(0);
                int kv_dim = W.at(L+"self_attn.k_proj.weight").shape(0);
                int nkv = kv_dim / hd;
                int q_dim = W.at(L+"self_attn.q_proj.weight").shape(0);
                int nh = q_dim / hd;
                int seq = x.shape(1);

                auto h = fast::rms_norm(x, W.at(L+"input_layernorm.weight"), rms_eps);
                auto q = qmm(h, L+"self_attn.q_proj");
                auto k_new = qmm(h, L+"self_attn.k_proj");
                auto v_new = has(L+"self_attn.v_proj.weight") ? qmm(h, L+"self_attn.v_proj") : k_new;

                q = transpose(reshape(q, {1,seq,nh,hd}), {0,2,1,3});
                k_new = transpose(reshape(k_new, {1,seq,nkv,hd}), {0,2,1,3});
                v_new = transpose(reshape(v_new, {1,seq,nkv,hd}), {0,2,1,3});

                q = fast::rms_norm(q, W.at(L+"self_attn.q_norm.weight"), rms_eps);
                k_new = fast::rms_norm(k_new, W.at(L+"self_attn.k_norm.weight"), rms_eps);
                v_new = bare_rms_norm(v_new, rms_eps);

                if (global) {
                    q = fast::rope(q, 128, false, std::nullopt, 1.0f, offset, global_rope_freqs);
                    k_new = fast::rope(k_new, 128, false, std::nullopt, 1.0f, offset, global_rope_freqs);
                } else {
                    q = fast::rope(q, sliding_hd, false, 10000.0f, 1.0f, offset);
                    k_new = fast::rope(k_new, sliding_hd, false, 10000.0f, 1.0f, offset);
                }

                auto k_full = k_new, v_full = v_new;

                if (mode == 3 || mode == 4) {
                    int qbits = (mode == 4) ? 4 : 8;
                    auto kqv = quantize(k_new, 64, qbits);
                    auto vqv = quantize(v_new, 64, qbits);
                    if (q8[l].kq.has_value()) {
                        q8[l].kq = concatenate({q8[l].kq.value(), kqv[0]}, 2);
                        q8[l].ks = concatenate({q8[l].ks.value(), kqv[1]}, 2);
                        q8[l].kb = concatenate({q8[l].kb.value(), kqv[2]}, 2);
                        q8[l].vq = concatenate({q8[l].vq.value(), vqv[0]}, 2);
                        q8[l].vs = concatenate({q8[l].vs.value(), vqv[1]}, 2);
                        q8[l].vb = concatenate({q8[l].vb.value(), vqv[2]}, 2);
                    } else {
                        q8[l].kq = kqv[0]; q8[l].ks = kqv[1]; q8[l].kb = kqv[2];
                        q8[l].vq = vqv[0]; q8[l].vs = vqv[1]; q8[l].vb = vqv[2];
                    }
                    int total = q8[l].kq.value().shape(2);
                    if (!global && total > sliding_window) {
                        int s = total - sliding_window;
                        auto tr = [&](std::optional<array>& a) { a = slice(a.value(), {0,0,s,0}, a.value().shape()); };
                        tr(q8[l].kq); tr(q8[l].ks); tr(q8[l].kb);
                        tr(q8[l].vq); tr(q8[l].vs); tr(q8[l].vb);
                    }
                    k_full = dequantize(q8[l].kq.value(), q8[l].ks.value(), q8[l].kb.value(), 64, qbits);
                    v_full = dequantize(q8[l].vq.value(), q8[l].vs.value(), q8[l].vb.value(), 64, qbits);
                } else {
                    if (kv[l].k.has_value()) {
                        k_new = concatenate({kv[l].k.value(), k_new}, 2);
                        v_new = concatenate({kv[l].v.value(), v_new}, 2);
                    }
                    if (!global && k_new.shape(2) > sliding_window) {
                        int s = k_new.shape(2) - sliding_window;
                        k_new = slice(k_new, {0,0,s,0}, k_new.shape());
                        v_new = slice(v_new, {0,0,s,0}, v_new.shape());
                    }
                    kv[l].k = k_new; kv[l].v = v_new;
                    k_full = k_new; v_full = v_new;
                }

                auto attn = fast::scaled_dot_product_attention(q, k_full, v_full, attn_scale, "causal");
                attn = reshape(transpose(attn, {0,2,1,3}), {1, seq, nh * hd});
                auto attn_out = qmm(attn, L+"self_attn.o_proj");
                if (has(L+"post_attention_layernorm.weight"))
                    attn_out = fast::rms_norm(attn_out, W.at(L+"post_attention_layernorm.weight"), rms_eps);
                auto hidden = x + attn_out;

                if (has(L+"pre_feedforward_layernorm.weight")) {
                    auto h2 = fast::rms_norm(hidden, W.at(L+"pre_feedforward_layernorm.weight"), rms_eps);
                    auto gate = qmm(h2, L+"mlp.gate_proj");
                    auto up = qmm(h2, L+"mlp.up_proj");
                    auto gelu = gate * array(0.5f) * (array(1.0f) + tanh(
                        array(0.7978845608028654f) * (gate + array(0.044715f) * power(gate, array(3)))));
                    auto ffn = qmm(gelu * up, L+"mlp.down_proj");
                    if (has(L+"post_feedforward_layernorm.weight"))
                        ffn = fast::rms_norm(ffn, W.at(L+"post_feedforward_layernorm.weight"), rms_eps);
                    hidden = hidden + ffn;
                }
                if (has(L+"layer_scalar")) hidden = hidden * W.at(L+"layer_scalar");
                return hidden;
            };

            // Prefill
            Timer t;
            auto pidx = array(test.tokens.data(), {(int)test.tokens.size()}, int32);
            auto emb = reshape(take(full_embed, pidx, 0), {1, (int)test.tokens.size(), hidden_size});
            emb = emb * array(std::sqrt((float)hidden_size));
            auto h = emb;
            for (int l = 0; l < n_layers; l++) h = run_layer(h, l);
            offset += test.tokens.size();

            auto last = slice(h, {0, (int)test.tokens.size()-1, 0}, {1, (int)test.tokens.size(), hidden_size});
            last = fast::rms_norm(last, final_norm, rms_eps);
            auto logits = quantized_matmul(last, embed_w, embed_s, std::optional<array>(embed_b), true, 64, 4);
            logits = array(final_softcap) * tanh(logits / array(final_softcap));
            eval(logits);
            double prefill_ms = t.ms();

            // Generate 50 tokens
            int n_gen = 50;
            std::string output;
            auto flat = reshape(logits, {-1});
            int next_id = argmax(flat, 0).item<int32_t>();

            t.reset();
            for (int step = 0; step < n_gen; step++) {
                if (next_id < (int)vocab.size()) output += vocab[next_id];
                if (next_id == 1 || next_id == 106 || next_id == 50) break;

                auto idx = array({next_id}, {1}, int32);
                auto e = reshape(take(full_embed, idx, 0), {1, 1, hidden_size});
                e = e * array(std::sqrt((float)hidden_size));
                h = e;
                for (int l = 0; l < n_layers; l++) h = run_layer(h, l);
                offset++;

                h = fast::rms_norm(h, final_norm, rms_eps);
                logits = quantized_matmul(h, embed_w, embed_s, std::optional<array>(embed_b), true, 64, 4);
                logits = array(final_softcap) * tanh(logits / array(final_softcap));
                eval(logits);
                flat = reshape(logits, {-1});
                next_id = argmax(flat, 0).item<int32_t>();
            }
            double gen_ms = t.ms();

            // Memory
            size_t bytes = 0;
            for (int l = 0; l < n_layers; l++) {
                if (q8[l].kq.has_value()) {
                    eval(q8[l].kq.value());
                    bytes += q8[l].kq.value().nbytes() + q8[l].ks.value().nbytes() + q8[l].kb.value().nbytes();
                    bytes += q8[l].vq.value().nbytes() + q8[l].vs.value().nbytes() + q8[l].vb.value().nbytes();
                }
                if (kv[l].k.has_value()) {
                    eval(kv[l].k.value());
                    bytes += kv[l].k.value().nbytes() + kv[l].v.value().nbytes();
                }
            }

            printf("\n[%s] prefill=%.0fms, gen=%.0fms (%.1f tok/s), KV=%.1fMB\n",
                mode_names[mode], prefill_ms, gen_ms,
                n_gen / (gen_ms / 1000.0), bytes / 1e6);
            // Trim output for display
            if (output.size() > 200) output = output.substr(0, 200) + "...";
            printf("OUTPUT: %s\n", output.c_str());
        }
    }

    return 0;
}
