# TurboQuant — Fused int4 SDPA for Gemma 4 on Apple Silicon

> Custom Metal kernel that computes attention directly from int4 quantized KV cache — 37% faster than dequantize + Flash Attention. 177 experiments.

## Results

### Fused sdpa_int4 Metal Kernel

The fused kernel dequantizes K/V in GPU registers and computes attention in a single Metal dispatch. No intermediate matrices materialized.

| Context | TurboQuant | Baseline | Speedup | Memory Saved |
|:--------|:----------:|:--------:|:-------:|:------------:|
| 33 tokens | 10.4 tok/s | 9.6 tok/s | +8% | ~80 MB |
| 786 tokens | **10.0 tok/s** | **7.3 tok/s** | **+37%** | **~640 MB** |

The fused kernel maintains near-constant ~10 tok/s regardless of context length, while the baseline degrades as context grows.

### Python Extension

TurboQuant also ships as a Python extension (nanobind) callable from any MLX program:

```python
import turboquant_ext as tq
tq.set_metallib_path("build/turboquant.metallib")

# 1.6x faster than mx.dequantize + mx.fast.scaled_dot_product_attention
result = tq.sdpa_int4(queries, k_quant, k_scales, k_biases,
                       v_quant, v_scales, v_biases, gqa_factor)
```

| Context | TurboQuant | MLX quantized_matmul | Speedup |
|:--------|:----------:|:--------------------:|:-------:|
| 50 tok  | 0.35ms | 0.43ms | **1.22x** |
| 500 tok | 0.42ms | 0.52ms | **1.23x** |
| 950 tok | 0.45ms | 0.59ms | **1.33x** |

### All Configurations Tested

| Configuration | tok/s | Notes |
|:---|:---:|:---|
| MoE 26B-A4B (mlx-lm) | 59 | Different model, 4B active params |
| 31B + TurboQuant fused | **10.0** | Custom Metal kernel |
| 31B baseline C++ | 7.3-9.6 | Degrades with context |
| 31B mlx-lm (Python) | 15 | mx.compile graph fusion |

## How It Works

Standard attention with int4 KV cache:
```
dequantize(K_int4) → K_float32 [640MB temporary] → Flash Attention → output
```

TurboQuant fused attention:
```
sdpa_int4(Q, K_int4, V_int4) → output [zero temporaries, dequantize in registers]
```

The kernel uses:
- **Vectorized 4-bit unpacking** (MLX qdot pattern: pre-scaled query × uint16 packed masks)
- **Online softmax** (single pass, no intermediate score matrix)
- **SIMD-parallel** architecture matching MLX's sdpa_vector design
- **Two configurations**: D=256 (BN=32, transpose reduction) and D=512 (BN=16, sequential reduction)

## Quick Start

```bash
# Build C++ engine
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make gemma4 -j8

# Run (needs Gemma 4 31B 4-bit weights)
./gemma4

# Run tests
./test_sdpa_int4

# Python extension
cd python && cmake -B /tmp/tq_build -DPython_EXECUTABLE=$(which python3.12) && cmake --build /tmp/tq_build -j8
PYTHONPATH=/tmp/tq_build python3.12 -c "import turboquant_ext; print('OK')"
```

## File Structure

```
turboquant/
├── gemma4_multilayer.cpp   # C++ inference engine with fused SDPA
├── turboquant.h            # API: sdpa_int4, PolarQuant primitives
├── turboquant.cpp          # MLX Primitive implementations + Metal dispatch
├── turboquant.metal        # Metal kernels: sdpa_int4_256/512 + PolarQuant
├── CMakeLists.txt          # Build system
├── test_sdpa_int4.cpp      # Kernel correctness test
├── chat.py                 # Python wrapper for C++ engine
├── chat_repl.py            # Interactive REPL
├── run_tests.sh            # 5/5 regression tests
├── mlx_moe_chat.py         # MoE 59 tok/s (mlx-lm, for comparison)
├── mlx_chat.py             # 31B 15 tok/s (mlx-lm, for comparison)
├── python/                 # Python extension (nanobind)
│   ├── CMakeLists.txt
│   ├── tq_bindings.cpp
│   ├── setup.py
│   └── turboquant.py       # Pure-Python fallback
└── assets/                 # Benchmark charts
```

## Key Findings (177 experiments)

1. **Fused int4 SDPA beats dequantize+SDPA** — 37% faster at 786 tokens
2. **PolarQuant fails on Gemma 4** — attn_scale=1.0 amplifies angular error
3. **QJL correction makes quality worse** — +6.5% perplexity
4. **int4 KV compounds across 60 layers** — fails beyond ~950 tokens
5. **int8 KV also fails** beyond ~1000 tokens (attn_scale=1.0 fundamental)
6. **MoE is 4x faster than dense** on Apple Silicon (bandwidth-bound)
7. **Speculative decode slower** — 25% E2B acceptance rate
8. **FP16 intermediates don't help** — quantized_matmul has cast overhead

## Papers

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — QJL + PolarQuant
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI) — 1-bit quantized JL transform
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — recursive polar quantization
