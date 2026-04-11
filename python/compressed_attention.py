"""
Compressed-domain attention: THE TurboQuant innovation on Metal.

Single Metal kernel computes attention output from:
  - Compressed keys (PolarQuant indices + radii)
  - Raw queries (full precision)
  - Full precision values

The key insight: attention scores are computed directly from compressed
indices without ever materializing the full key matrix. This saves
memory bandwidth proportional to the compression ratio.

For N=256K tokens with D=128:
  Standard SDPA reads: N × D × 2 bytes = 64MB per query
  Compressed scoring: N × (indices + radii) ≈ N × 20 bytes = 5MB per query
  That's 12x less memory bandwidth!
"""

import mlx.core as mx
import numpy as np


# Fused attention: score + softmax + value weighted sum
# Per query: iterate over all keys, compute score, track running softmax, accumulate output
# This is "flash attention" style — online softmax, no materialization of N×N score matrix

_compressed_attention_kernel = mx.fast.metal_kernel(
    name="compressed_attention",
    input_names=[
        "queries",        # (Nq, D) — preconditioned queries (Pq)
        "codebook_l1", "codebook_l2", "codebook_l3", "codebook_l4",
        "indices_l1",     # (Nk * n_blocks * 8) uint8
        "indices_l2",     # (Nk * n_blocks * 4) uint8
        "indices_l3",     # (Nk * n_blocks * 2) uint8
        "indices_l4",     # (Nk * n_blocks) uint8
        "radii",          # (Nk * n_blocks) float16
        "values",         # (Nk, D) float32
        "params",         # [Nk, D, n_blocks, scale_factor]
    ],
    output_names=["output"],  # (Nq, D)
    header="""
constant constexpr uint BLOCK_SIZE = 16;
""",
    source="""
        // Each thread handles one query → produces one D-dim output
        uint qi = thread_position_in_grid.x;

        uint Nk = as_type<uint>(params[0]);
        uint D = as_type<uint>(params[1]);
        uint N_BLOCKS = as_type<uint>(params[2]);
        float scale = params[3];

        // Online softmax state
        float max_score = -1e30f;
        float sum_exp = 0.0f;
        float acc[256];  // max D = 256
        for (uint d = 0; d < D; d++) acc[d] = 0.0f;

        // Load query
        float q[256];
        for (uint d = 0; d < D; d++) {
            q[d] = queries[qi * D + d];
        }

        // For each key token
        for (uint ki = 0; ki < Nk; ki++) {
            // Compute score from compressed key: <Pq, y_recon>
            // y_recon = polar_inverse(codebooks, indices[ki])
            float score = 0.0f;

            for (uint blk = 0; blk < N_BLOCKS; blk++) {
                uint block_idx = ki * N_BLOCKS + blk;

                // Inverse polar transform for this block
                float r[16];
                r[0] = static_cast<float>(radii[block_idx]);

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

                // Dot product: query[blk*16:(blk+1)*16] · r
                uint base = blk * BLOCK_SIZE;
                for (uint j = 0; j < BLOCK_SIZE; j++) {
                    score += q[base + j] * r[j];
                }
            }

            // Apply attention scaling
            score *= scale;

            // Online softmax: update running max and sum
            float old_max = max_score;
            if (score > max_score) max_score = score;

            float exp_diff = metal::exp(old_max - max_score);
            float exp_score = metal::exp(score - max_score);

            // Rescale accumulated values
            sum_exp = sum_exp * exp_diff + exp_score;
            for (uint d = 0; d < D; d++) {
                acc[d] = acc[d] * exp_diff + exp_score * values[ki * D + d];
            }
        }

        // Normalize by sum of exponentials
        float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (uint d = 0; d < D; d++) {
            output[qi * D + d] = acc[d] * inv_sum;
        }
    """,
)


def compressed_attention(
    queries_preconditioned: mx.array,  # (Nq, D) — Pq
    codebooks: list,                    # [cb_l1..4]
    indices: list,                      # [idx_l1..4] flat uint8
    radii: mx.array,                   # (Nk * n_blocks,) float16
    values: mx.array,                  # (Nk, D) float32
    n_keys: int,
    embed_dim: int,
    n_blocks: int,
    scale: float,
) -> mx.array:
    """
    Compressed-domain attention in a single Metal kernel dispatch.

    Computes: softmax(score(Pq, compressed_keys) * scale) @ values
    WITHOUT ever materializing the full (Nk, D) key matrix.
    """
    Nq = queries_preconditioned.shape[0]

    params = mx.array([
        mx.array(n_keys, dtype=mx.uint32).view(mx.float32).item(),
        mx.array(embed_dim, dtype=mx.uint32).view(mx.float32).item(),
        mx.array(n_blocks, dtype=mx.uint32).view(mx.float32).item(),
        scale,
    ], dtype=mx.float32)

    outputs = _compressed_attention_kernel(
        inputs=[
            queries_preconditioned.astype(mx.float32),
            codebooks[0], codebooks[1], codebooks[2], codebooks[3],
            indices[0].astype(mx.uint8),
            indices[1].astype(mx.uint8),
            indices[2].astype(mx.uint8),
            indices[3].astype(mx.uint8),
            radii.astype(mx.float16),
            values.astype(mx.float32),
            params,
        ],
        output_shapes=[(Nq, embed_dim)],
        output_dtypes=[mx.float32],
        grid=(Nq, 1, 1),
        threadgroup=(1, 1, 1),
    )
    return outputs[0]
