# QJL (Quantized Johnson-Lindenstrauss) Specification

Source: arxiv.org/abs/2406.03482 (AAAI 2025)
Reference CUDA: github.com/amirzandieh/QJL

## Core Idea

Apply JL random projection to compress high-dimensional vectors, then quantize to sign bits (1-bit). Use an asymmetric estimator: quantized keys × full-precision query sketch. This is unbiased with variance O(1/d).

## Algorithm

### Setup
```
S ∈ R^(m × d), S_{ij} ~ N(0,1)  // random projection matrix
// m = sketch_dim, typically m = d
// Generated via QR decomposition of random Gaussian, scaled by √d
```

### Key Quantization
```
QJL_QUANT(k):
  1. outlier_idx ← top-k dimensions of k by magnitude
  2. k_inlier ← k with outlier dims zeroed
  3. k_outlier ← k with inlier dims zeroed
  4. sketch ← S × k_inlier                     // R^m
  5. quant ← sign(sketch)                       // {-1, +1}^m
  6. Pack: 8 sign bits → 1 uint8
  7. outlier_sketch ← S_outlier × k_outlier     // separate projection for outliers
  8. outlier_quant ← sign(outlier_sketch)
  9. key_norm ← ||k||₂
  10. outlier_norm ← ||k_outlier||₂
  11. Return (quant_packed, outlier_quant_packed, key_norm, outlier_norm, outlier_idx)
```

### Score Computation (Attention Logits)
```
QJL_SCORE(q, quantized_key):
  // q is full precision query vector
  1. query_sketch ← q @ S^T                     // R^m, NOT quantized
  2. norm_k ← √(key_norm² - outlier_norm²)      // inlier norm
  3. scl ← √(π/2) / sketch_dim
  4. For each quantized key token:
     a. Unpack sign bits from uint8
     b. inner_prod ← Σ_i (bit_i ? +query_sketch_i : -query_sketch_i)
     c. inner_prod_outlier ← similar for outlier bits
     d. score ← scl × norm_k × inner_prod
              + scl_outlier × outlier_norm × inner_prod_outlier
  5. Return score
```

### Mathematical Guarantee
```
E[QJL_SCORE(q, QJL_QUANT(k))] = <q, k>    // UNBIASED
Var[...] ≤ (π / (2d)) · ||q||² · ||k||²    // variance → 0 as d grows
```

## CUDA Kernel Architecture (to port to Metal)

### qjl_quant_kernel
```
Grid:  (batch × head × n, blocksPerGroup, numProjBlocks)
Block: (WARP_SIZE=32, WARPS_PER_BLOCK=32)

Shared memory:
  shared_mask[EMB_DIM]           // outlier flags (uint8)
  shared_keys[EMB_DIM][WARP_SIZE] // transposed key tile
  shared_key_quant[WARP_SIZE][WARPS_PER_BLOCK] // bit staging

Algorithm:
  1. Load outlier mask into shared memory
  2. For each projection block:
     a. Load key tile into shared (transpose for coalescing)
     b. Each warp handles one projection row
     c. Dot product: accumulate S[row][col] × key[col] for inlier cols
     d. Take sign of accumulator
     e. Pack 8 sign bits → uint8 via shift-and-OR
  3. Write packed bits to output
```

### qjl_score_kernel (calc_score_kernel)
```
Grid:  (batch × head, n_tokens, blocksPerGroup)
Block: (32, 8) — 32 threads/warp, 8 warps/block

Shared memory:
  shared_query[128]
  shared_outlier_ind[32]
  shared_innprod[32]
  shared_q_sketch[32][8]

Algorithm:
  1. Load query into shared memory
  2. Compute query_sketch = q @ S^T (tiled across warps)
  3. For each quantized key token:
     a. Load packed uint8 bits
     b. Unpack: (byte >> shift) & 1
     c. Accumulate: bit ? +query_sketch_i : -query_sketch_i
     d. Warp reduce sum
     e. Apply scaling: score = scl × norm × sum
  4. Write attention logit
```

## Memory Layout

Per token in KV cache:
```
key_quant:         (B, H, N, sketch_dim/8) uint8    // packed sign bits
key_outlier_quant: (B, H, N, outlier_sketch_dim/8) uint8
key_norm:          (B, H, N) float32
outlier_norm:      (B, H, N) float32
outlier_indices:   (B, H, N, k) uint8

Effective: ~3 bits per coordinate (1 sign + amortized norms/outliers)
```

## Metal Porting Notes

| CUDA | Metal |
|------|-------|
| `__shared__` | `threadgroup` memory (32KB limit on M1) |
| `atomicAdd` | `atomic_fetch_add_explicit(..., memory_order_relaxed)` |
| Warp shuffle `__shfl_down_sync` | `simd_shuffle_down` |
| Warp size 32 | SIMD width 32 on M1 GPU |
| `cudaAccessPropertyPersisting` (L2 hint) | No equivalent; rely on automatic caching |
| Thread block (blockDim) | Threadgroup (threads_per_threadgroup) |
| Grid (gridDim) | Threadgroups per grid |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |

Key M1 GPU specs:
- 32 execution units, 32-wide SIMD
- 32KB threadgroup memory
- 64KB tile memory (for tile shading, not compute)
- Unified memory — no explicit host↔device transfers needed
