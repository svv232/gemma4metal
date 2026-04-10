# PolarQuant Algorithm Specification

Source: arxiv.org/abs/2502.02617 (AISTATS 2026)

## Core Idea

Convert vectors from Cartesian to polar coordinates recursively. After random rotation (preconditioning), the angle distributions become data-independent and analytically known, eliminating the need to store per-block quantization constants. This removes the 1-2 bit overhead of traditional quantizers.

## Algorithm

### Setup (once)
```
1. Generate preconditioning matrix P:
   Option A: QR decomposition of random N(0,1) matrix → O(d²)
   Option B: Randomized Hadamard: P = H · diag(s) where H is normalized
             Hadamard matrix, s_i ∈ {-1, +1} random → O(d log d)

2. Precompute codebooks per level:
   For l = 1: Lloyd-Max on Uniform[0, 2π), allocate b₁ bits
   For l = 2,...,log₂(d): Lloyd-Max on f_{Ψ^(l)}(ψ), allocate b_l bits
```

### Forward Polar Transform
```
POLAR(y):  // y = P × x (preconditioned vector)
  r^(0) ← y
  For l = 1, ..., log₂(d):
    For j = 1, ..., d/2^l:
      ψ_j^(l) ← arctan2(r_{2j}^(l-1), r_{2j-1}^(l-1))  // level 1: [0,2π)
      // or arctan(r_{2j}^(l-1) / r_{2j-1}^(l-1))         // level ≥2: [0,π/2]
      r_j^(l) ← √(r_{2j-1}^(l-1)² + r_{2j}^(l-1)²)

  Output: angles {ψ_j^(l)} for all levels, radius r₁^(log₂d) = ||y||₂
```

### Angle Quantization
```
QUANT_ANGLES(angles, codebooks):
  For each level l, for each angle ψ_j^(l):
    idx_j^(l) ← argmin_{k ∈ [2^{b_l}]} |codebook_l[k] - ψ_j^(l)|
  Return all indices
```

### Inverse Polar Transform (Dequantization)
```
INVERSE_POLAR(indices, radius, codebooks):
  r₁^(log₂d) ← radius
  For l = log₂(d), ..., 1:   // REVERSE order
    For j = 1, ..., d/2^l:
      θ ← codebook_l[idx_j^(l)]
      r_{2j-1}^(l-1) ← r_j^(l) × cos(θ)
      r_{2j}^(l-1)   ← r_j^(l) × sin(θ)

  x̃ ← P^T × r^(0)           // inverse preconditioning
  Return x̃
```

## Angle Distribution PDFs

After preconditioning with random orthogonal P, the angle distributions are data-independent:

**Level 1** (range [0, 2π)):
```
f_{Ψ^(1)}(ψ) = 1/(2π)    // Uniform
```

**Level l ≥ 2** (range [0, π/2]):
```
f_{Ψ^(l)}(ψ) = [Γ(2^(l-1)) / (2^(2^(l-1)-2) · Γ(2^(l-2))²)] · sin^(2^(l-1)-1)(2ψ)
```

Properties:
- Mean = π/4 for all l ≥ 2
- Variance = O(1/2^l) — concentrates tighter at higher levels
- For l=2: sin¹(2ψ) = sin(2ψ), fairly spread
- For l=3: sin³(2ψ), more concentrated around π/4
- For l=4: sin⁷(2ψ), very concentrated
- For l≥5: extremely concentrated, 1-2 bits suffice

## Bit Allocation (block size 16, d=16, L=4 levels)

| Level | # Angles | Range      | Bits/angle | Total bits |
|-------|----------|------------|------------|------------|
| 1     | 8        | [0, 2π)    | 4          | 32         |
| 2     | 4        | [0, π/2]   | 2          | 8          |
| 3     | 2        | [0, π/2]   | 2          | 4          |
| 4     | 1        | [0, π/2]   | 2          | 2          |
| Radius| 1        | R⁺         | 16 (FP16)  | 16         |
| **Total** | **16** |            |            | **62 bits** |

**Effective: 62/16 = 3.875 bits per coordinate**

For different target bit-widths, adjust level-1 bits:
- 3-bit target: level-1 gets 2 bits → 16+8+4+2+16 = 46 bits → 2.875 bpc
- 4-bit target: level-1 gets 4 bits → 32+8+4+2+16 = 62 bits → 3.875 bpc
- 2-bit target: level-1 gets 1 bit → 8+8+4+2+16 = 38 bits → 2.375 bpc

## Codebook Construction

### Offline (data-independent, recommended)
```python
import numpy as np
from scipy import integrate, special

def level_pdf(psi, level):
    """Analytical PDF for angle at given level (l >= 2)."""
    n = 2**(level - 1)
    log_norm = (special.gammaln(n) - (n-2)*np.log(2) - 2*special.gammaln(n/2))
    return np.exp(log_norm) * np.sin(2*psi)**(n-1)

def compute_codebook(level, n_bits, n_iters=1000):
    """Lloyd-Max codebook for a given level."""
    if level == 1:
        # Uniform on [0, 2π)
        n_centroids = 2**n_bits
        return np.linspace(0, 2*np.pi, n_centroids, endpoint=False) + np.pi/n_centroids

    n_centroids = 2**n_bits
    a, b = 0, np.pi/2
    pdf = lambda psi: level_pdf(psi, level)

    # Initialize uniformly
    centroids = np.linspace(a, b, n_centroids + 2)[1:-1]

    for _ in range(n_iters):
        boundaries = np.concatenate([[a], (centroids[:-1] + centroids[1:])/2, [b]])
        for i in range(n_centroids):
            num, _ = integrate.quad(lambda x: x * pdf(x), boundaries[i], boundaries[i+1])
            den, _ = integrate.quad(pdf, boundaries[i], boundaries[i+1])
            if den > 0:
                centroids[i] = num / den
    return centroids
```

### Online (dataset-specific, optional)
Collect angle samples from actual key vectors during prefill, run 1-D k-means++ per level. Slightly better quality but requires extra computation.

## Reconstruction Formula (Exact)

For coordinate i of a d-dimensional vector:
```
x_i = ||x||₂ × ∏_{l=1}^{log₂d} [cos(ψ_{⌊i/2^l⌋}^(l))]^{1{(i mod 2^l) ≤ 2^(l-1)}}
                                 × [sin(ψ_{⌊i/2^l⌋}^(l))]^{1{(i mod 2^l) > 2^(l-1)}}
```

## Error Bound

```
E[||x - x̃||₂²] = ε · ||x||₂²
```
achieved with O(log(1/ε)) bits per coordinate.

## Computational Complexity

- Preconditioning P×x: O(d²) full, O(d log d) Hadamard
- Polar transform: O(d log d) — log₂(d) levels, d/2 ops per level
- Codebook lookup: O(d · 2^b) per vector
- Inverse polar: O(d log d)
- Inverse preconditioning: O(d²) full, O(d log d) Hadamard

## Metal Implementation Notes

- Polar transform: level l has d/2^l independent (arctan, norm) pairs — parallel within level, sequential across levels
- Inverse polar: same structure, d/2^l independent (cos, sin) multiplies per level
- Metal fast-math intrinsics: `atan2`, `cos`, `sin`, `sqrt`, `rsqrt`
- Block size 16 = 4 levels, fits entirely in registers/threadgroup memory
- Codebooks are tiny (max 16 centroids per level) → Metal constant memory or threadgroup memory
- Bit packing variable-width indices needs careful implementation but straightforward with shifts
- For Hadamard preconditioning: use butterfly structure, maps well to SIMD operations
