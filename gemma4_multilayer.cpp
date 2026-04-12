// Gemma 4 31B Inference with Quantized KV Cache
// Pure C++ on Apple Silicon via MLX + Metal
// Architecture: sliding_attention (hd=256, nkv=16, window=1024) + full_attention (hd=512, nkv=4)
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
    if (!f) { printf("WARNING: vocab.bin not found\n"); return {}; }
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
    std::cout << "Gemma 4 31B — Quantized KV Cache Inference" << std::endl;

    turboquant::set_metallib_path(METALLIB_PATH);

    // Gemma 4 31B config
    const int hidden_size = 5376, n_layers = 60;
    const int sliding_hd = 256, global_hd = 512;
    const int sliding_nkv = 16, global_nkv = 4;
    const int sliding_window = 1024;
    const float rms_eps = 1e-6f, final_softcap = 30.0f, attn_scale = 1.0f;
    auto is_global = [](int l) { return (l % 6) == 5; };

    // Global layers: proportional RoPE (theta=1M, 128/512 dims rotated)
    std::vector<float> gf(64);
    for (int i = 0; i < 64; i++) gf[i] = 1.0f / std::pow(1000000.0f, 2.0f * i / 512.0f);
    auto global_rope_freqs = array(gf.data(), {64}, float32);
    eval(global_rope_freqs);

    std::string model_dir = "/Users/andromeda/.cache/huggingface/hub/"
        "models--mlx-community--gemma-4-31b-it-4bit/snapshots/"
        "535c5606372deb5d5ab7e29280f111ef2a8e084e/";

    auto vocab = load_vocab("vocab.bin");
    printf("Vocab: %zu tokens\n", vocab.size());

    // Load 4-bit weights
    std::unordered_map<std::string, array> W;
    Timer load_t;
    for (int s = 1; s <= 4; s++) {
        auto [weights, _] = load_safetensors(
            model_dir + "model-0000" + std::to_string(s) + "-of-00004.safetensors");
        for (auto& kv : weights) W.insert_or_assign(kv.first, kv.second);
    }
    printf("Loaded %zu tensors in %.1fs\n", W.size(), load_t.ms()/1000);

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

    // KV cache: 0=FP32, 1=FP16 (2x), 4=int4 (6.4x, ≤950), 11=BF16 (2x)
    int cache_mode = 4;  // int4 default, auto-switches to FP16 for long prompts
    const char* mode_name = (cache_mode == 10) ? "FP32-global+int4-sliding" :
                            (cache_mode == 9) ? "int8-global+int4-sliding" :
                            (cache_mode == 4) ? "int4 K+V" :
                            (cache_mode == 8) ? "int4K+int8V" :
                            (cache_mode == 3) ? "int8 K+V" :
                            (cache_mode == 12) ? "FP16-early+int4-late" : (cache_mode == 11) ? "BF16 K+V" :
                            (cache_mode == 1) ? "FP16 K+V" : "FP32";
    printf("KV cache: %s\n", mode_name);

    struct KVCache { std::optional<array> k, v; };
    struct QuantKV {
        std::optional<array> kq, ks, kb, vq, vs, vb;
    };
    std::vector<KVCache> fp_cache(n_layers);
    std::vector<QuantKV> q_cache(n_layers);
    int cache_offset = 0;

    // Save/Load KV cache for prompt caching
    auto save_kv_cache = [&](const std::string& path) {
        std::unordered_map<std::string, array> data;
        data.insert_or_assign("cache_offset", array({cache_offset}, {1}, int32));
        for (int l = 0; l < n_layers; l++) {
            if (q_cache[l].kq.has_value()) {
                std::string p = "l" + std::to_string(l) + ".";
                data.insert_or_assign(p+"kq", q_cache[l].kq.value());
                data.insert_or_assign(p+"ks", q_cache[l].ks.value());
                data.insert_or_assign(p+"kb", q_cache[l].kb.value());
                data.insert_or_assign(p+"vq", q_cache[l].vq.value());
                data.insert_or_assign(p+"vs", q_cache[l].vs.value());
                data.insert_or_assign(p+"vb", q_cache[l].vb.value());
            }
        }
        save_safetensors(path, data);
    };

    auto load_kv_cache = [&](const std::string& path) -> bool {
        std::ifstream check(path);
        if (!check) return false;
        auto [data, _] = load_safetensors(path);
        cache_offset = data.at("cache_offset").item<int32_t>();
        for (int l = 0; l < n_layers; l++) {
            std::string p = "l" + std::to_string(l) + ".";
            if (data.find(p+"kq") != data.end()) {
                q_cache[l].kq = data.at(p+"kq"); q_cache[l].ks = data.at(p+"ks"); q_cache[l].kb = data.at(p+"kb");
                q_cache[l].vq = data.at(p+"vq"); q_cache[l].vs = data.at(p+"vs"); q_cache[l].vb = data.at(p+"vb");
            }
        }
        return true;
    };

    // Forward: one transformer layer
    auto run_layer = [&](const array& x, int l) -> array {
        std::string L = "language_model.model.layers." + std::to_string(l) + ".";
        bool global = is_global(l);
        int hd = W.at(L+"self_attn.k_norm.weight").shape(0);
        int nkv = W.at(L+"self_attn.k_proj.weight").shape(0) / hd;
        int nh = W.at(L+"self_attn.q_proj.weight").shape(0) / hd;
        int seq = x.shape(1);

        // Input norm (Gemma4: weight applied directly, not 1+weight)
        auto h = fast::rms_norm(x, W.at(L+"input_layernorm.weight"), rms_eps);

        // QKV projections
        auto q = qmm(h, L+"self_attn.q_proj");
        auto k_new = qmm(h, L+"self_attn.k_proj");
        auto v_new = has(L+"self_attn.v_proj.weight") ? qmm(h, L+"self_attn.v_proj") : k_new;

        q = transpose(reshape(q, {1,seq,nh,hd}), {0,2,1,3});
        k_new = transpose(reshape(k_new, {1,seq,nkv,hd}), {0,2,1,3});
        v_new = transpose(reshape(v_new, {1,seq,nkv,hd}), {0,2,1,3});

        // Q/K norm + V bare norm (Gemma4: v_norm has no weight)
        q = fast::rms_norm(q, W.at(L+"self_attn.q_norm.weight"), rms_eps);
        k_new = fast::rms_norm(k_new, W.at(L+"self_attn.k_norm.weight"), rms_eps);
        v_new = bare_rms_norm(v_new, rms_eps);

        // RoPE: global=proportional(theta=1M, 128 dims), sliding=default(theta=10K, full)
        if (global) {
            q = fast::rope(q, 128, false, std::nullopt, 1.0f, cache_offset, global_rope_freqs);
            k_new = fast::rope(k_new, 128, false, std::nullopt, 1.0f, cache_offset, global_rope_freqs);
        } else {
            q = fast::rope(q, hd, false, 10000.0f, 1.0f, cache_offset);
            k_new = fast::rope(k_new, hd, false, 10000.0f, 1.0f, cache_offset);
        }

        // KV cache update
        auto k_full = k_new, v_full = v_new;

        if (cache_mode == 10 && global) {
            // Mode 10: FP32 for global layers
            if (fp_cache[l].k.has_value()) {
                k_new = concatenate({fp_cache[l].k.value(), k_new}, 2);
                v_new = concatenate({fp_cache[l].v.value(), v_new}, 2);
            }
            fp_cache[l].k = k_new; fp_cache[l].v = v_new;
            k_full = k_new; v_full = v_new;
        } else if (cache_mode == 12 && l < 30) {
            // Layer-split: FP16 for early layers (0-29), int4 for late (30-59)
            auto k16 = astype(k_new, float16);
            auto v16 = astype(v_new, float16);
            if (fp_cache[l].k.has_value()) {
                k16 = concatenate({fp_cache[l].k.value(), k16}, 2);
                v16 = concatenate({fp_cache[l].v.value(), v16}, 2);
            }
            if (!global && k16.shape(2) > sliding_window) {
                int s = k16.shape(2) - sliding_window;
                k16 = slice(k16, {0,0,s,0}, k16.shape());
                v16 = slice(v16, {0,0,s,0}, v16.shape());
            }
            fp_cache[l].k = k16; fp_cache[l].v = v16;
            k_full = astype(k16, float32); v_full = astype(v16, float32);
        } else if (cache_mode == 3 || cache_mode == 4 || cache_mode == 8 || cache_mode == 9 || cache_mode == 12 || (cache_mode == 10 && !global)) {
            int kb = (cache_mode == 3) ? 8 :
                     (cache_mode == 9) ? (global ? 8 : 4) :
                     (cache_mode == 8) ? 4 : 4;
            int vb = (cache_mode == 3) ? 8 :
                     (cache_mode == 9) ? (global ? 8 : 4) :
                     (cache_mode == 8) ? 8 : 4;
            auto kqv = quantize(k_new, 64, kb);
            auto vqv = quantize(v_new, 64, vb);
            if (q_cache[l].kq.has_value()) {
                q_cache[l].kq = concatenate({q_cache[l].kq.value(), kqv[0]}, 2);
                q_cache[l].ks = concatenate({q_cache[l].ks.value(), kqv[1]}, 2);
                q_cache[l].kb = concatenate({q_cache[l].kb.value(), kqv[2]}, 2);
                q_cache[l].vq = concatenate({q_cache[l].vq.value(), vqv[0]}, 2);
                q_cache[l].vs = concatenate({q_cache[l].vs.value(), vqv[1]}, 2);
                q_cache[l].vb = concatenate({q_cache[l].vb.value(), vqv[2]}, 2);
            } else {
                q_cache[l].kq = kqv[0]; q_cache[l].ks = kqv[1]; q_cache[l].kb = kqv[2];
                q_cache[l].vq = vqv[0]; q_cache[l].vs = vqv[1]; q_cache[l].vb = vqv[2];
            }
            // Cache size limits
            int total = q_cache[l].kq.value().shape(2);
            // int4 safe zone: 950 entries max per layer (compound error beyond this)
            // FP16 mode: use full sliding_window (no compound error issue)
            int max_cache = (cache_mode == 4) ? 950 : (global ? 999999 : sliding_window);
            if (total > max_cache) {
                int s = total - max_cache;
                auto tr = [&](std::optional<array>& a) { a = slice(a.value(), {0,0,s,0}, a.value().shape()); };
                tr(q_cache[l].kq); tr(q_cache[l].ks); tr(q_cache[l].kb);
                tr(q_cache[l].vq); tr(q_cache[l].vs); tr(q_cache[l].vb);
            }
            k_full = dequantize(q_cache[l].kq.value(), q_cache[l].ks.value(), q_cache[l].kb.value(), 64, kb);
            v_full = dequantize(q_cache[l].vq.value(), q_cache[l].vs.value(), q_cache[l].vb.value(), 64, vb);
        } else if (cache_mode == 1 || cache_mode == 11) {
            // FP16/BF16 cache: half the memory of FP32
            auto target_type = (cache_mode == 11) ? bfloat16 : float16;
            auto k16 = astype(k_new, target_type);
            auto v16 = astype(v_new, target_type);
            if (fp_cache[l].k.has_value()) {
                k16 = concatenate({fp_cache[l].k.value(), k16}, 2);
                v16 = concatenate({fp_cache[l].v.value(), v16}, 2);
            }
            if (!global && k16.shape(2) > sliding_window) {
                int s = k16.shape(2) - sliding_window;
                k16 = slice(k16, {0,0,s,0}, k16.shape());
                v16 = slice(v16, {0,0,s,0}, v16.shape());
            }
            fp_cache[l].k = k16; fp_cache[l].v = v16;
            k_full = astype(k16, float32); v_full = astype(v16, float32);
        } else {
            // FP32 cache
            if (fp_cache[l].k.has_value()) {
                k_new = concatenate({fp_cache[l].k.value(), k_new}, 2);
                v_new = concatenate({fp_cache[l].v.value(), v_new}, 2);
            }
            if (!global && k_new.shape(2) > sliding_window) {
                int s = k_new.shape(2) - sliding_window;
                k_new = slice(k_new, {0,0,s,0}, k_new.shape());
                v_new = slice(v_new, {0,0,s,0}, v_new.shape());
            }
            fp_cache[l].k = k_new; fp_cache[l].v = v_new;
            k_full = k_new; v_full = v_new;
        }

        // Attention (scale=1.0, q/k norms handle magnitude)
        // For seq=1 (decode) or seq <= kv_len with no prior cache: standard causal
        // For prefill with prior cache (chunked): custom mask
        int kv_len = k_full.shape(2);
        int prior = kv_len - seq;  // cached entries before this chunk
        array attn = array(0.0f);

        if (seq == 1 || (prior == 0 && seq <= sliding_window)) {
            // Simple case: decode or short prefill
            attn = fast::scaled_dot_product_attention(q, k_full, v_full, attn_scale, "causal");
        } else {
            // Complex case: build proper mask
            // prior entries (from cache) are all BEFORE current Q — accessible
            // current chunk: causal within chunk
            // sliding window: limit distance for sliding layers
            std::vector<float> md(seq * kv_len, 0.0f);
            for (int i = 0; i < seq; i++) {
                int q_pos = prior + i;  // absolute position of this Q token
                for (int j = 0; j < kv_len; j++) {
                    bool is_future = (j > q_pos);  // future token — mask
                    bool outside_window = !global && (q_pos - j) >= sliding_window;
                    if (is_future || outside_window)
                        md[i * kv_len + j] = -1e9f;
                }
            }
            auto mask = reshape(array(md.data(), {seq * kv_len}, float32), {1, 1, seq, kv_len});
            attn = fast::scaled_dot_product_attention(q, k_full, v_full, attn_scale, "", {mask});
        }
        attn = reshape(transpose(attn, {0,2,1,3}), {1, seq, nh * hd});
        auto attn_out = qmm(attn, L+"self_attn.o_proj");
        if (has(L+"post_attention_layernorm.weight"))
            attn_out = fast::rms_norm(attn_out, W.at(L+"post_attention_layernorm.weight"), rms_eps);
        auto hidden = x + attn_out;

        // FFN: GELU(gate) * up → down
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

        // Layer scalar (Gemma4: scales entire hidden state at end)
        if (has(L+"layer_scalar")) hidden = hidden * W.at(L+"layer_scalar");
        return hidden;
    };

    // Load prompt
    std::vector<int> prompt_tokens;
    {
        std::ifstream pf("prompt_491.bin", std::ios::binary);
        if (pf) {
            uint32_t n; pf.read((char*)&n, 4);
            prompt_tokens.resize(n);
            pf.read((char*)prompt_tokens.data(), n * 4);
        } else {
            // Default: "What is the capital of France?"
            prompt_tokens = {2, 236820, 3041, 236779, 1340, 236779, 887, 236813, 2364, 107,
                3689, 563, 506, 5279, 529, 7001, 145912, 643, 236779, 1340,
                236779, 887, 236813, 107, 236820, 3041, 236779, 1340, 236779,
                887, 236813, 4368, 107};
        }
    }

    // Auto-switch to FP16 for long prompts (int4 compounds beyond ~950 tokens)
    if (cache_mode == 4 && (int)prompt_tokens.size() > 950) {
        printf("Auto-switching to FP16 KV (prompt %zu > 950 tokens)\n", prompt_tokens.size());
        cache_mode = 1;
        mode_name = "FP16 K+V";
    }
    if ((int)prompt_tokens.size() > sliding_window) {
        printf("Context: %zu tokens (sliding window=%d, global layers see all)\n",
            prompt_tokens.size(), sliding_window);
    }

    printf("\nPrompt (%zu tokens): ", prompt_tokens.size());
    for (int t : prompt_tokens)
        if (t < (int)vocab.size()) printf("%s", vocab[t].c_str());
    printf("\n");

    // Try loading cached KV
    std::string cache_file = "kv_cache.safetensors";
    Timer prefill_t;
    bool cache_loaded = false;

    if (cache_mode == 4 || cache_mode == 8 || cache_mode == 9) {
        cache_loaded = load_kv_cache(cache_file);
        if (cache_loaded)
            printf("\nLoaded KV cache from %s (offset=%d) in %.0fms\n", cache_file.c_str(), cache_offset, prefill_t.ms());
    }

    array logits = array(0.0f);
    if (!cache_loaded) {
        printf("\nPrefilling...\n");

        // Chunked prefill: process in chunks of sliding_window to handle long prompts
        int chunk_size = sliding_window;
        int total_tokens = prompt_tokens.size();
        int n_chunks = (total_tokens + chunk_size - 1) / chunk_size;

        array h = array(0.0f);
        for (int chunk = 0; chunk < n_chunks; chunk++) {
            int start = chunk * chunk_size;
            int end = std::min(start + chunk_size, total_tokens);
            int len = end - start;

            std::vector<int> chunk_tokens(prompt_tokens.begin() + start, prompt_tokens.begin() + end);
            auto pidx = array(chunk_tokens.data(), {len}, int32);
            auto tok_embed = reshape(take(full_embed, pidx, 0), {1, len, hidden_size});
            tok_embed = tok_embed * array(std::sqrt((float)hidden_size));

            h = tok_embed;
            for (int l = 0; l < n_layers; l++) {
                h = run_layer(h, l);
                if ((l < 3 || l == n_layers - 1) && chunk == n_chunks - 1) {
                    eval(h); printf("  Layer %d: %.0fms\n", l, prefill_t.ms());
                }
            }
            cache_offset += len;
            eval(h);
            if (n_chunks > 1)
                printf("  Chunk %d/%d: %d tokens (%.0fms)\n", chunk+1, n_chunks, len, prefill_t.ms());
        }

        int last_len = h.shape(1);
        auto last_h = slice(h, {0, last_len-1, 0}, {1, last_len, hidden_size});
        last_h = fast::rms_norm(last_h, final_norm, rms_eps);
        logits = quantized_matmul(last_h, embed_w, embed_s, std::optional<array>(embed_b), true, 64, 4);
        logits = array(final_softcap) * tanh(logits / array(final_softcap));
        eval(logits);
        printf("Prefill: %.1fs (%.1f ms/tok)\n", prefill_t.ms()/1000, prefill_t.ms()/prompt_tokens.size());

        // Save KV cache for next time
        if (cache_mode == 4 || cache_mode == 8 || cache_mode == 9) {
            Timer save_t;
            save_kv_cache(cache_file);
            printf("Saved KV cache to %s (%.0fms)\n", cache_file.c_str(), save_t.ms());
        }
    } else {
        // Generate first logits from cached state by running a dummy forward
        // (We need the last hidden state, but it wasn't cached — re-derive from last token)
        int last_tok = prompt_tokens.back();
        auto idx = array({last_tok}, {1}, int32);
        auto emb = reshape(take(full_embed, idx, 0), {1, 1, hidden_size});
        emb = emb * array(std::sqrt((float)hidden_size));
        auto h = emb;
        for (int l = 0; l < n_layers; l++) h = run_layer(h, l);
        cache_offset++;
        h = fast::rms_norm(h, final_norm, rms_eps);
        logits = quantized_matmul(h, embed_w, embed_s, std::optional<array>(embed_b), true, 64, 4);
        logits = array(final_softcap) * tanh(logits / array(final_softcap));
        eval(logits);
    }

    auto flat = reshape(logits, {-1});
    int next_id = argmax(flat, 0).item<int32_t>();
    printf("First token: %d", next_id);
    if (next_id < (int)vocab.size()) printf(" = \"%s\"", vocab[next_id].c_str());
    printf("\n\nGenerating:\n");

    // Generate
    int n_gen = 600;
    std::vector<int> gen_tokens;
    Timer gen_t;

    for (int step = 0; step < n_gen; step++) {
        gen_tokens.push_back(next_id);
        if (next_id == 1 || next_id == 106 || next_id == 50) { printf(" [EOS]\n"); break; }

        auto idx = array({next_id}, {1}, int32);
        auto emb = reshape(take(full_embed, idx, 0), {1, 1, hidden_size});
        emb = emb * array(std::sqrt((float)hidden_size));
        auto hh = emb;
        for (int l = 0; l < n_layers; l++) hh = run_layer(hh, l);
        cache_offset++;

        hh = fast::rms_norm(hh, final_norm, rms_eps);
        logits = quantized_matmul(hh, embed_w, embed_s, std::optional<array>(embed_b), true, 64, 4);
        logits = array(final_softcap) * tanh(logits / array(final_softcap));
        eval(logits);

        flat = reshape(logits, {-1});
        // Temperature sampling (0.7) with basic repetition mitigation
        float temp = 0.7f;
        if (gen_tokens.size() >= 3 && gen_tokens.back() == gen_tokens[gen_tokens.size()-2])
            temp = 1.2f;
        auto probs = softmax(flat / array(temp), 0);
        auto sampled = random::categorical(log(probs + array(1e-10f)));
        eval(sampled);
        next_id = sampled.item<int32_t>();

        if (next_id < (int)vocab.size()) printf("%s", vocab[next_id].c_str());
        fflush(stdout);
    }

    printf("\n\n%zu tokens in %.1fs (%.1f tok/s)\n",
        gen_tokens.size(), gen_t.ms()/1000, gen_tokens.size() / (gen_t.ms()/1000));

    // Memory stats
    size_t cache_bytes = 0, fp32_equiv = 0;
    for (int l = 0; l < n_layers; l++) {
        if (q_cache[l].kq.has_value()) {
            eval(q_cache[l].kq.value());
            cache_bytes += q_cache[l].kq.value().nbytes() + q_cache[l].ks.value().nbytes() + q_cache[l].kb.value().nbytes();
            cache_bytes += q_cache[l].vq.value().nbytes() + q_cache[l].vs.value().nbytes() + q_cache[l].vb.value().nbytes();
        }
        if (fp_cache[l].k.has_value()) {
            eval(fp_cache[l].k.value());
            cache_bytes += fp_cache[l].k.value().nbytes() + fp_cache[l].v.value().nbytes();
        }
        int hd = is_global(l) ? global_hd : sliding_hd;
        int nkv = is_global(l) ? global_nkv : sliding_nkv;
        int seq = std::min(cache_offset, is_global(l) ? cache_offset : sliding_window);
        fp32_equiv += 2 * nkv * seq * hd * 4;
    }
    printf("KV cache (%s): %.1f MB (%.1f MB FP32 equiv, %.1fx)\n",
        mode_name, cache_bytes/1e6, fp32_equiv/1e6, fp32_equiv > 0 ? (float)fp32_equiv/cache_bytes : 0.f);

    return 0;
}
