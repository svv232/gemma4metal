# TurboQuant Algorithm Specification

Source: arxiv.org/abs/2504.19874 (ICLR 2026)

## Overview

TurboQuant combines PolarQuant (b-1 bits, MSE-optimal) with QJL (1 bit, unbiased error correction) to achieve b-bit quantization with zero memory overhead from quantization constants.

## Algorithm 1: TurboQuant_mse (MSE-Optimized)

### Setup (once)
```
1. Generate Pi ∈ R^(d×d) via QR decomposition of random N(0,1) matrix
2. Precompute codebook: centroids c_1,...,c_{2^b} ∈ [-1,1]
   minimizing C(f_X, b) = min Σ_{i=1}^{2^b} ∫ |x - c_i|² f_X(x) dx
   where f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^((d-3)/2)
```

### Quantize
```
QUANT_MSE(x):
  1. y ← Pi × x                              // random rotation
  2. For each j in [d]:
       idx_j ← argmin_{k ∈ [2^b]} |y_j - c_k|  // nearest centroid
  3. Return idx
```

### Dequantize
```
DEQUANT_MSE(idx):
  1. For each j in [d]: ŷ_j ← c_{idx_j}
  2. x̃ ← Pi^T × ŷ                           // inverse rotation
  3. Return x̃
```

## Algorithm 2: TurboQuant_prod (Unbiased Inner-Product)

### Setup (once)
```
1. Instantiate TurboQuant_mse with bit-width (b-1)
2. Generate S ∈ R^(d×d), S_{i,j} ~ N(0,1)
```

### Quantize
```
QUANT_PROD(x):
  1. idx ← Quant_mse(x)                      // (b-1)-bit MSE quantization
  2. r ← x - DeQuant_mse(idx)                // residual
  3. qjl ← sign(S × r)                       // 1-bit QJL on residual
  4. γ ← ||r||₂                              // residual norm (full precision)
  5. Return (idx, qjl, γ)
```

### Dequantize / Inner Product
```
DEQUANT_PROD(idx, qjl, γ):
  1. x̃_mse ← DeQuant_mse(idx)
  2. x̃_qjl ← (√(π/2) / d) · γ · S^T × qjl
  3. Return x̃_mse + x̃_qjl
```

For inner product estimation (attention scores), use asymmetric approach:
```
INNER_PROD(query, quantized_key):
  1. x̃_mse ← DeQuant_mse(idx)
  2. score_mse ← <query, x̃_mse>
  3. query_sketch ← S × query                // project query (NOT quantized)
  4. score_qjl ← (√(π/2) / d) · γ · <query_sketch, qjl>  // qjl is ±1 signs
  5. Return score_mse + score_qjl
```

## Codebook Values

Distribution on unit sphere after random rotation:
```
f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^((d-3)/2)
```
This is Beta((d-1)/2, (d-1)/2) on [-1,1], converges to N(0, 1/d) for large d.

Known centroids (for unit-norm vectors):
- **1-bit**: ±√(2/(πd))
- **2-bit**: ±0.453/√d, ±1.51/√d

3-bit and 4-bit must be computed via Lloyd-Max iteration:
```python
def lloyd_max(pdf, n_centroids, x_range=(-1, 1), iters=1000):
    # Initialize centroids uniformly
    centroids = np.linspace(x_range[0], x_range[1], n_centroids)
    for _ in range(iters):
        # Boundaries = midpoints between adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        boundaries = np.concatenate([[x_range[0]], boundaries, [x_range[1]]])
        # Update centroids = conditional expectation in each region
        for i in range(n_centroids):
            num = integrate(lambda x: x * pdf(x), boundaries[i], boundaries[i+1])
            den = integrate(lambda x: pdf(x), boundaries[i], boundaries[i+1])
            centroids[i] = num / den
    return centroids
```

## Bit Allocation for KV Cache

- **2.5-bit**: 32 outlier channels × 3 bits + 96 regular × 2 bits
- **3-bit**: full 3-bit TurboQuant_prod (2-bit PolarQuant + 1-bit QJL residual)
- **3.5-bit**: mixed outlier/regular allocation
- **4-bit**: 3-bit PolarQuant + 1-bit QJL residual

Outlier channels identified by magnitude (top-k dimensions with largest variance across tokens).

## Distortion Bounds

MSE: D_mse ≤ (√3 · π / 2) · (1/4^b) ≈ 2.72 / 4^b
Inner product: D_prod ≤ (√3 · π² · ||y||²) / (d · 4^b)
Lower bound: D_mse ≥ 1/4^b (gap to optimal ≈ 2.7×)

## Computational Complexity

- Random rotation Pi×x: O(d²) — or O(d log d) with randomized Hadamard
- Nearest centroid: O(d · 2^b) per vector
- QJL stage S×r: O(d²) — or O(d log d) with structured projection
- Total per token: O(d²) dominated by matrix-vector products
