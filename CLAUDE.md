# TurboQuant — Fast Gemma 4 31B on Apple Silicon

## CRITICAL: Never Stop

This is an open-ended research and engineering project. There is ALWAYS more to do. After every experiment, immediately start the next one.

## Goal

**Make Gemma 4 31B inference FASTER on Apple Silicon using TurboQuant's compressed-domain operations.**

We have compression (int4 KV = 6.4x). We have correctness (95 experiments validated). Now we need SPEED. The TurboQuant paper's real innovation isn't just smaller KV — it's computing attention scores DIRECTLY from compressed keys without decompressing. Less memory bandwidth = faster inference.

### Current Bottleneck (from profiling)
- Weight matmuls: **77%** of decode time (memory-bandwidth-bound at 17GB/400GB/s)
- SDPA (attention): **16%** — this is where TurboQuant can help
- KV cache quantize/dequant: **2%** — already fast, but we still dequant before SDPA
- Norms/RoPE: **5%**

### Speed Targets
| What | Current | Target | How |
|------|---------|--------|-----|
| Decode tok/s | 10-12 | 15-20 | Compressed-domain SDPA, speculative decode |
| SDPA time | ~16ms/tok | ~4ms/tok | `quantized_matmul(Q, K_compressed)` — skip dequant |
| Prefill ms/tok | 11 | 8 | Batched chunked prefill |
| Time to first token | 1.7s cold / 0.7s cached | 0.5s | Smaller first chunk |

### The TurboQuant Speed Opportunity

Current flow: `K_int4 → dequantize → FP32 K → Q @ K^T → softmax → @ V`

TurboQuant flow: `Q @ K_compressed → scores (no dequant!) → softmax → @ V_compressed`

MLX has `quantized_matmul(x, w_quantized)` which computes `x @ dequant(w)^T` WITHOUT materializing the full FP32 tensor. If we reshape Q as the "input" and K_cache as the "weight", we get compressed-domain scoring for free.

For V: same trick — `quantized_matmul(attn_weights, V_quantized)` gives the weighted sum without dequantizing V.

This eliminates the dequant step AND reduces memory bandwidth for the attention computation.

## What We Have (95 experiments)

### Working System
- **Gemma 4 31B** running in pure C++ with Metal compute shaders
- **Auto-adaptive KV**: int4 (6.4x, ≤950 tokens) / FP16 (2x, ≤4K)
- **Chunked prefill**: proper cross-chunk attention masks for >1024 context
- **10-12 tok/s** decode, **5/5 regression tests**, **streaming REPL**
- **Prompt caching**: safetensors, 1ms reload

### Gemma 4 Architecture (fully decoded)
- **60 layers**: 50 sliding (hd=256, nkv=16, window=1024) + 10 global (hd=512, nkv=4)
- **Norms**: `rms_norm(x) * weight` (NOT `1+weight`)
- **Attention scale**: 1.0 (q/k norms handle magnitude)
- **v_norm**: bare RMS norm (no weight)
- **layer_scalar**: scales entire hidden state at end of layer
- **RoPE**: sliding=default(theta=10K), global=proportional(theta=1M, 128/512 dims)
- **Activation**: gelu_pytorch_tanh, **V=K** in global layers

### Key Findings
1. PolarQuant angular error + attn_scale=1.0 = incompatible (use int4 instead)
2. QJL correction worsens quality (+6.5% perplexity)
3. int4 KV compounds across 60 layers — fails beyond ~950 tokens
4. FP16 (10-bit mantissa) works to 4K; BF16 (8-bit) fails
5. Chunked prefill with proper cross-chunk masks is essential for >1024
6. V precision matters more than K for attention output quality

## Hardware
- Apple M1 Max, 10-core CPU, 32-core GPU, 64GB unified memory
- Memory bandwidth: ~400 GB/s, 32KB threadgroup memory

## Quick Start
```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make gemma4 -j8
python3 chat.py "What is the capital of France?"
python3 chat_repl.py   # interactive streaming REPL
./run_tests.sh          # regression tests (5/5)
```

## File Structure
```
turboquant/
├── CLAUDE.md                  ← you are here
├── CMakeLists.txt
├── gemma4_multilayer.cpp      ← main inference engine (350 lines)
├── turboquant.h               ← PolarQuant Metal kernel API
├── turboquant.cpp             ← MLX Primitive implementations
├── turboquant.metal           ← Metal compute shaders
├── chat.py                    ← single-prompt wrapper
├── chat_repl.py               ← streaming interactive REPL
├── run_tests.sh               ← regression tests (5/5)
├── vocab.bin                  ← BPE vocabulary
└── benchmarks/
    ├── gemma4_bench.cpp       ← FP32 vs int4 comparison
    ├── context_scaling.cpp    ← throughput vs context length
    ├── profile_decode.cpp     ← per-component timing
    ├── kv_precision_sweep.cpp ← K/V bit-width analysis
    └── polar_quality_sweep.cpp← PolarQuant quality sweep
```

## Implementation Priorities

### Priority 1: Compressed-Domain SDPA (biggest speed win)
Replace `dequantize(K) → matmul(Q, K^T)` with `quantized_matmul(Q, K_quantized)`.
Same for V: `quantized_matmul(attn_weights, V_quantized)`.
Expected: eliminate 16% SDPA overhead, reduce memory bandwidth.

### Priority 2: Speculative Decoding (2x throughput)
Use Gemma 2 2B (locally available) as draft model. Generate N candidates, verify in parallel with 31B. Acceptance rate ~70% → ~2x effective throughput.

### Priority 3: Fused Metal Attention Kernel
Write a single Metal kernel that does: read compressed K → compute scores → softmax → read compressed V → weighted sum. One dispatch instead of 4 separate operations. Eliminates intermediate memory traffic.

### Priority 4: Weight Optimization
Explore 2-bit weight quantization for the FFN layers (54% of decode time). MLX supports 2-bit quantized_matmul. 2-bit FFN + 4-bit attention weights could nearly double decode throughput by halving memory bandwidth for the dominant bottleneck.

## Papers
- TurboQuant: arxiv.org/abs/2504.19874 (ICLR 2026)
- QJL: arxiv.org/abs/2406.03482 (AAAI)
- PolarQuant: arxiv.org/abs/2502.02617 (AISTATS 2026)
