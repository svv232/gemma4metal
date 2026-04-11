"""
Fused Metal kernel for TurboQuant attention scoring.

Combines polar inverse reconstruction + inner product + QJL residual scoring
into a single Metal kernel dispatch, eliminating intermediate memory traffic.

For each (query, key_token) pair, computes:
  score = <q, reconstruct_polar(key)> + sqrt(pi/2)/d * gamma * <S*q, signs>

Each thread handles one (query, key_token) pair.
"""

import mlx.core as mx
import numpy as np

# Precomputed sqrt(pi/2)
_SQRT_PI_OVER_2 = np.sqrt(np.pi / 2)

_fused_score_kernel = mx.fast.metal_kernel(
    name="turboquant_fused_score",
    input_names=[
        "queries",           # (Nq, D) float32
        "precondition",      # (D, D) float32 — orthogonal preconditioning matrix
        "codebook_l1", "codebook_l2", "codebook_l3", "codebook_l4",
        "indices_l1",        # (Nk * 8,) uint32
        "indices_l2",        # (Nk * 4,) uint32
        "indices_l3",        # (Nk * 2,) uint32
        "indices_l4",        # (Nk,) uint32
        "radii",             # (Nk * n_blocks,) float32
        "projection",        # (sketch_dim, D) float32 — QJL projection
        "sign_bits",         # (Nk, sketch_dim) uint8
        "residual_norms",    # (Nk,) float32
    ],
    output_names=["scores"],
    header="""
// TurboQuant fused scoring constants
constant constexpr float SQRT_PI_OVER_2 = 1.2533141f;
constant constexpr uint BLOCK_SIZE = 16;
constant constexpr uint N_BLOCKS_PER_VEC = 8;  // 128 / 16
constant constexpr uint DIM = 128;
constant constexpr uint SKETCH_DIM = 128;
""",
    source="""
        uint qi = thread_position_in_grid.x;  // query index
        uint ki = thread_position_in_grid.y;  // key index

        // --- Step 1: Reconstruct key from PolarQuant indices ---
        // Each key has N_BLOCKS_PER_VEC=8 blocks of 16 coords
        float key_recon[DIM];

        for (uint blk = 0; blk < N_BLOCKS_PER_VEC; blk++) {
            uint block_idx = ki * N_BLOCKS_PER_VEC + blk;
            float r[16];
            r[0] = radii[block_idx];

            // Level 4: 1 → 2
            {
                uint idx = indices_l4[block_idx];
                float theta = codebook_l4[idx];
                float r0 = r[0];
                r[0] = r0 * metal::cos(theta);
                r[1] = r0 * metal::sin(theta);
            }
            // Level 3: 2 → 4
            {
                float tmp[4];
                for (uint j = 0; j < 2; j++) {
                    uint idx = indices_l3[block_idx * 2 + j];
                    float theta = codebook_l3[idx];
                    tmp[2*j]   = r[j] * metal::cos(theta);
                    tmp[2*j+1] = r[j] * metal::sin(theta);
                }
                for (uint j = 0; j < 4; j++) r[j] = tmp[j];
            }
            // Level 2: 4 → 8
            {
                float tmp[8];
                for (uint j = 0; j < 4; j++) {
                    uint idx = indices_l2[block_idx * 4 + j];
                    float theta = codebook_l2[idx];
                    tmp[2*j]   = r[j] * metal::cos(theta);
                    tmp[2*j+1] = r[j] * metal::sin(theta);
                }
                for (uint j = 0; j < 8; j++) r[j] = tmp[j];
            }
            // Level 1: 8 → 16
            {
                float tmp[16];
                for (uint j = 0; j < 8; j++) {
                    uint idx = indices_l1[block_idx * 8 + j];
                    float theta = codebook_l1[idx];
                    tmp[2*j]   = r[j] * metal::cos(theta);
                    tmp[2*j+1] = r[j] * metal::sin(theta);
                }
                for (uint j = 0; j < 16; j++) r[j] = tmp[j];
            }

            // Store reconstructed block (in preconditioned space)
            for (uint j = 0; j < 16; j++) {
                key_recon[blk * 16 + j] = r[j];
            }
        }

        // --- Step 1b: Inverse precondition: x = P^T @ y ---
        // P is orthogonal so P^T = P^(-1). x[i] = sum_j P[j][i] * y[j]
        // Since precondition is (D, D) stored row-major: P[row][col]
        // x[i] = sum_j P[j][i] * key_recon[j]  (column i of P)
        float key_orig[DIM];
        for (uint i = 0; i < DIM; i++) {
            float acc = 0.0f;
            for (uint j = 0; j < DIM; j++) {
                // P^T[i][j] = P[j][i]
                acc += precondition[j * DIM + i] * key_recon[j];
            }
            key_orig[i] = acc;
        }

        // --- Step 2: PolarQuant score = <q, key_orig> ---
        float score_polar = 0.0f;
        for (uint d = 0; d < DIM; d++) {
            score_polar += queries[qi * DIM + d] * key_orig[d];
        }

        // --- Step 3: QJL residual score ---
        // query_sketch = q @ projection^T
        // score_qjl = sqrt(pi/2)/sketch_dim * residual_norm * sum(sign_i * query_sketch_i)
        float score_qjl = 0.0f;
        for (uint s = 0; s < SKETCH_DIM; s++) {
            // Compute query_sketch[s] = sum_d q[d] * projection[s][d]
            float qs = 0.0f;
            for (uint d = 0; d < DIM; d++) {
                qs += queries[qi * DIM + d] * projection[s * DIM + d];
            }
            // sign bit
            uint8_t bit = sign_bits[ki * SKETCH_DIM + s];
            score_qjl += bit ? qs : -qs;
        }
        float scale_qjl = SQRT_PI_OVER_2 / float(SKETCH_DIM);
        score_qjl *= scale_qjl * residual_norms[ki];

        // --- Final score ---
        uint Nk = residual_norms_shape[0];
        scores[qi * Nk + ki] = score_polar + score_qjl;
    """,
)


def turboquant_fused_score(
    queries: mx.array,           # (Nq, D)
    precondition: mx.array,      # (D, D)
    codebooks: list,             # [l1, l2, l3, l4] codebook arrays
    angle_indices: list,         # [l1, l2, l3, l4] flat index arrays
    radii: mx.array,             # (Nk * n_blocks,) float32
    projection: mx.array,        # (sketch_dim, D)
    sign_bits: mx.array,         # (Nk, sketch_dim) uint8
    residual_norms: mx.array,    # (Nk,) float32
) -> mx.array:
    """
    Fused TurboQuant scoring kernel.

    Combines polar inverse + inner product + QJL residual in single dispatch.

    Returns:
        scores: (Nq, Nk) attention logits (unscaled)
    """
    Nq = queries.shape[0]
    Nk = residual_norms.shape[0]

    outputs = _fused_score_kernel(
        inputs=[
            queries.astype(mx.float32),
            precondition.astype(mx.float32),
            codebooks[0], codebooks[1], codebooks[2], codebooks[3],
            angle_indices[0].astype(mx.uint32),
            angle_indices[1].astype(mx.uint32),
            angle_indices[2].astype(mx.uint32),
            angle_indices[3].astype(mx.uint32),
            radii.astype(mx.float32),
            projection.astype(mx.float32),
            sign_bits,
            residual_norms.astype(mx.float32),
        ],
        output_shapes=[(Nq, Nk)],
        output_dtypes=[mx.float32],
        grid=(Nq, Nk, 1),
        threadgroup=(1, 1, 1),
    )
    return outputs[0]
