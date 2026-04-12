# Fused int4 Attention for Gemma 4 on Apple Silicon

> I set out to implement TurboQuant (PolarQuant + QJL) for Gemma 4's KV cache. It doesn't work on this model. What I built instead is faster.

## Background: Why TurboQuant Fails on Gemma 4

The [TurboQuant paper](https://arxiv.org/abs/2504.19874) (ICLR 2026) proposes PolarQuant + QJL for KV cache compression. I implemented both as Metal compute shaders and tested them on Gemma 4 31B across 177 experiments.

**PolarQuant** encodes key vectors as recursive polar coordinates (radius + angles). On most models this achieves ~7x compression with high quality. On Gemma 4, it produces gibberish. The reason: Gemma 4 uses `attn_scale=1.0` (the Q/K norms handle magnitude instead), which means attention scores are not dampened — a 4% directional error from PolarQuant gets amplified through softmax and compounds across 60 layers.

**QJL** (1-bit quantized Johnson-Lindenstrauss) is supposed to correct PolarQuant's residual error. On Gemma 4, it makes quality *worse* — +6.5% perplexity. The 1-bit variance adds noise to an already sensitive attention mechanism.

**MLX's native int4 quantization** (per-group scale + bias) works fine at 6.4x compression — close to PolarQuant's 7.1x — because it preserves the exact direction of key vectors better than angular encoding.

![Paper vs Implementation](assets/paper_comparison.png)

## What I Built Instead

Since int4 KV cache works but the standard attention path wastes bandwidth dequantizing keys every token, I wrote a **fused Metal kernel** (`sdpa_int4`) that computes attention scores directly from the int4 packed data. No dequantization. No intermediate float32 matrices. Everything happens in GPU registers.

```
Standard:  dequantize(K_int4) → K_float32 [780MB temporary] → SDPA → output
Ours:      sdpa_int4(Q, K_int4, V_int4) → output [0 bytes temporary]
```

The kernel uses MLX's own `qdot` vectorization pattern: pre-divide query values by `{1, 16, 256, 4096}`, multiply against `uint16` masks `{0xF, 0xF0, 0xF00, 0xF000}`, accumulate with online softmax. One Metal dispatch per attention head group, with SIMD-parallel reduction across simdgroups.

## Performance

### Decode speed stays constant as context grows

The baseline slows down because dequantizing longer key sequences costs more bandwidth. The fused kernel reads int4 data directly — its cost barely changes with context length.

![Throughput vs Context](assets/throughput_vs_context.png)

| Context | Fused Kernel | Baseline (deq + SDPA) | Speedup |
|:--------|:------------:|:---------------------:|:-------:|
| 33 tokens | 10.4 tok/s | 9.6 tok/s | +8% |
| 423 tokens | 10.0 tok/s | 7.4 tok/s | +34% |
| 786 tokens | **10.0 tok/s** | **7.3 tok/s** | **+37%** |

### Peak memory: 780MB less at 950 tokens

Every attention layer in the baseline allocates a temporary float32 matrix for the dequantized keys. Across 50 sliding-attention layers at 950 tokens, that's 780MB of temporaries. The fused kernel allocates nothing.

![Memory Savings](assets/memory_savings.png)

### Python extension: 1.2-1.3x faster than MLX's quantized_matmul

The kernel ships as a nanobind Python extension. Benchmarked against MLX's own `quantized_matmul`-based attention (which is what mlx-lm uses internally):

![Python Kernel Benchmark](assets/python_kernel_benchmark.png)

## Hardware

Gemma 4 31B at 4-bit weights needs 17.4 GB. Whether 256K context fits depends on KV cache format:

![Hardware Requirements](assets/hardware_requirements.png)

## What I Tried (177 experiments)

| Approach | Result | Why |
|:---|:---|:---|
| **Fused sdpa_int4 kernel** | **+37% decode** | Dequantize in registers, zero temporaries |
| PolarQuant (Metal shader) | Broken output | attn_scale=1.0 amplifies angular error |
| QJL residual correction | +6.5% perplexity | 1-bit variance hurts softmax |
| int4 KV > 950 tokens | Fails | Compound error across 60 layers |
| int8 KV > 1000 tokens | Fails | Same fundamental limit (attn_scale=1.0) |
| Speculative decode (E2B→31B) | 12 tok/s (slower) | 25% draft acceptance rate |
| FP16 intermediates | Slower | quantized_matmul has cast overhead |
| async_eval pipelining | Neutral | Mutable KV cache prevents overlap |
| Chunked layer eval | Slower | Sync overhead exceeds graph overhead |

## Getting Started

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`)
- Python 3.12 (`brew install python@3.12`)
- CMake 3.27+ (`brew install cmake`)
- ~20 GB free disk space (for model weights)

### Step 1: Clone and set up

```bash
git clone https://github.com/your-org/turboquant.git
cd turboquant
```

### Step 2: Build MLX from source

TurboQuant links against MLX's C++ library. Build it once:

```bash
git clone --depth 1 https://github.com/ml-explore/mlx.git /tmp/mlx-source
mkdir /tmp/mlx-source/build && cd /tmp/mlx-source/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF
make -j8
cd -
```

### Step 3: Build TurboQuant

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
cd ..
```

This produces two binaries in `build/`:
- `gemma4` — the inference engine
- `test_sdpa_int4` — kernel correctness test

### Step 4: Download model weights

```bash
pip3.12 install huggingface_hub
python3.12 -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/gemma-4-31b-it-4bit', local_dir_links=True)
"
```

The engine expects weights at `~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/`. If your path differs, update the `model_dir` string in `engine/gemma4_multilayer.cpp`.

### Step 5: Run

```bash
# Run inference
./build/gemma4

# Run kernel correctness test (should print PASS)
./build/test_sdpa_int4

# Run regression tests (5/5 should pass)
bash engine/run_tests.sh
```

### Step 6 (optional): Python extension

To use the fused kernel from Python:

```bash
# Set up venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install mlx>=0.31.0

# Build extension (pins nanobind v2.10.2 to match MLX's ABI)
cmake -S python -B /tmp/tq_build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DPython_EXECUTABLE=$(which python3.12)
cmake --build /tmp/tq_build -j8

# Verify
PYTHONPATH=/tmp/tq_build python3.12 -c "
import turboquant_ext as tq
tq.set_metallib_path('/tmp/tq_build/turboquant.metallib')
print('TurboQuant Python extension loaded')
"
```

Usage:

```python
import turboquant_ext as tq
import mlx.core as mx

tq.set_metallib_path("/tmp/tq_build/turboquant.metallib")

# queries: (num_heads, head_dim) float32
# k_quant: (num_kv_heads, N, head_dim/8) uint32  (from mx.quantize)
# k_scales, k_biases: (num_kv_heads, N, head_dim/64) float32
result = tq.sdpa_int4(
    queries, k_quant, k_scales, k_biases,
    v_quant, v_scales, v_biases,
    gqa_factor=2,   # num_heads // num_kv_heads
    scale=1.0,      # attention scale
    sliding_window=0
)
```

## File Structure

```
turboquant/
├── lib/                        # Fused int4 SDPA library
│   ├── turboquant.h            #   C++ API
│   ├── turboquant.cpp          #   Metal kernel dispatch (MLX Primitive)
│   └── turboquant.metal        #   Metal compute shaders (sdpa_int4 + PolarQuant)
├── engine/                     # Gemma 4 31B inference engine
│   ├── gemma4_multilayer.cpp   #   60-layer forward pass
│   ├── chat.py / chat_repl.py  #   Python wrappers
│   └── run_tests.sh            #   Regression tests (5/5)
├── python/                     # Python extension (nanobind → pre-compiled Metal)
│   ├── CMakeLists.txt          #   Pinned nanobind v2.10.2 (MLX 0.31.1 ABI)
│   ├── tq_bindings.cpp         #   Bindings exposing lib/turboquant.metal kernels
│   ├── setup.py                #   pip-installable
│   └── requirements.txt        #   mlx>=0.31.0
├── tests/
│   └── test_sdpa_int4.cpp      #   Kernel correctness (max error < 0.00001)
└── assets/                     #   Benchmark charts
```

## References

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — the paper I attempted to implement
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI) — 1-bit quantized JL transform
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — recursive polar quantization
