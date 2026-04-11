"""
Fused Metal kernels for TurboQuant quantize/dequantize.

Eliminates Python orchestration overhead by doing the entire pipeline
in a single Metal kernel dispatch:

Quantize:  precondition -> forward_polar -> codebook_nearest -> pack
Dequantize: unpack -> inverse_polar -> inverse_precondition

Target: match mx.quantize speed (~3ms) vs current Python pipeline (~430ms).
"""

import mlx.core as mx
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Fused Quantize: one kernel does everything per vector
#
# Input: (N, D) float32 vectors, (D, D) precondition matrix, codebooks
# Output: packed angle indices (uint8), radius (float16)
#
# Each thread processes one D-dimensional vector:
#   1. y = P @ x (precondition)
#   2. For each block of 16: forward polar -> angles
#   3. For each angle: nearest codebook entry -> index
#   4. Store indices + radius
# ──────────────────────────────────────────────────────────────────────

_fused_quantize_kernel = mx.fast.metal_kernel(
    name="turboquant_fused_quantize",
    input_names=[
        "vectors",       # (N, D) float32
        "precondition",  # (D, D) float32
        "cb_l1",         # level 1 codebook
        "cb_l2",         # level 2 codebook
        "cb_l3",         # level 3 codebook
    ],
    output_names=[
        "indices_l1",    # (N, n_blocks * 4) uint8 — level 1 indices
        "indices_l2",    # (N, n_blocks * 2) uint8
        "indices_l3",    # (N, n_blocks) uint8
        "radii",         # (N, n_blocks) float16
    ],
    header="""
constant constexpr uint BLOCK_SIZE = 16;
constant constexpr uint DIM = 128;
constant constexpr uint N_BLOCKS = 8;  // DIM / BLOCK_SIZE
""",
    source="""
        uint vec_idx = thread_position_in_grid.x;
        uint D_actual = vectors_shape[1];

        // Step 1: Precondition y = P @ x
        float y[DIM];
        for (uint i = 0; i < D_actual; i++) {
            float acc = 0.0f;
            for (uint j = 0; j < D_actual; j++) {
                acc += precondition[i * D_actual + j] * vectors[vec_idx * D_actual + j];
            }
            y[i] = acc;
        }

        uint n_blocks = D_actual / BLOCK_SIZE;

        // Step 2-3: For each block, do forward polar + codebook lookup
        for (uint blk = 0; blk < n_blocks; blk++) {
            float r[16];
            for (uint j = 0; j < BLOCK_SIZE; j++) {
                r[j] = y[blk * BLOCK_SIZE + j];
            }

            // Forward polar level 1: 16 -> 8 angles + 8 radii
            uint8_t idx_l1[8];
            {
                float tmp[8];
                uint n_cb1 = cb_l1_shape[0];
                for (uint j = 0; j < 8; j++) {
                    float a = r[2*j], b = r[2*j+1];
                    float angle = metal::atan2(b, a);
                    if (angle < 0.0f) angle += 2.0f * M_PI_F;

                    // Nearest codebook entry
                    float best_dist = 1e10f;
                    uint8_t best_idx = 0;
                    for (uint c = 0; c < n_cb1; c++) {
                        float d = metal::abs(angle - cb_l1[c]);
                        if (d < best_dist) { best_dist = d; best_idx = c; }
                    }
                    idx_l1[j] = best_idx;
                    tmp[j] = metal::sqrt(a*a + b*b);
                }
                for (uint j = 0; j < 8; j++) r[j] = tmp[j];
            }

            // Forward polar level 2: 8 -> 4
            uint8_t idx_l2[4];
            {
                float tmp[4];
                uint n_cb2 = cb_l2_shape[0];
                for (uint j = 0; j < 4; j++) {
                    float a = r[2*j], b = r[2*j+1];
                    float angle = metal::atan2(b, a);
                    float best_dist = 1e10f;
                    uint8_t best_idx = 0;
                    for (uint c = 0; c < n_cb2; c++) {
                        float d = metal::abs(angle - cb_l2[c]);
                        if (d < best_dist) { best_dist = d; best_idx = c; }
                    }
                    idx_l2[j] = best_idx;
                    tmp[j] = metal::sqrt(a*a + b*b);
                }
                for (uint j = 0; j < 4; j++) r[j] = tmp[j];
            }

            // Forward polar level 3: 4 -> 2
            uint8_t idx_l3[2];
            {
                float tmp[2];
                uint n_cb3 = cb_l3_shape[0];
                for (uint j = 0; j < 2; j++) {
                    float a = r[2*j], b = r[2*j+1];
                    float angle = metal::atan2(b, a);
                    float best_dist = 1e10f;
                    uint8_t best_idx = 0;
                    for (uint c = 0; c < n_cb3; c++) {
                        float d = metal::abs(angle - cb_l3[c]);
                        if (d < best_dist) { best_dist = d; best_idx = c; }
                    }
                    idx_l3[j] = best_idx;
                    tmp[j] = metal::sqrt(a*a + b*b);
                }
                r[0] = tmp[0]; r[1] = tmp[1];
            }

            // Level 4: 2 -> 1 (radius only, stored as the block's final value)
            // For block_size=16 with 4 levels, level 4 gives final radius
            {
                float a = r[0], b = r[1];
                // We could quantize this angle too, but for now just store radius
                radii[vec_idx * n_blocks + blk] = static_cast<half>(metal::sqrt(a*a + b*b));
            }

            // Store indices
            // Level 1: 8 indices per block
            for (uint j = 0; j < 8; j++) {
                indices_l1[vec_idx * n_blocks * 8 + blk * 8 + j] = idx_l1[j];
            }
            // Level 2: 4 per block
            for (uint j = 0; j < 4; j++) {
                indices_l2[vec_idx * n_blocks * 4 + blk * 4 + j] = idx_l2[j];
            }
            // Level 3: 2 per block (pack into 1 byte: 4 bits each)
            indices_l3[vec_idx * n_blocks * 2 + blk * 2] = idx_l3[0];
            indices_l3[vec_idx * n_blocks * 2 + blk * 2 + 1] = idx_l3[1];
        }
    """,
)


def fused_quantize(
    vectors: mx.array,          # (N, D) float32
    precondition: mx.array,     # (D, D) float32
    codebooks: list,            # [cb_l1, cb_l2, cb_l3] for 3 higher levels
) -> dict:
    """
    Fused TurboQuant quantize: precondition + polar_forward + codebook_lookup
    in a single Metal kernel dispatch.

    Returns dict with packed indices and radii.
    """
    N, D = vectors.shape
    n_blocks = D // 16

    outputs = _fused_quantize_kernel(
        inputs=[
            vectors.astype(mx.float32),
            precondition.astype(mx.float32),
            codebooks[0].astype(mx.float32),  # level 1
            codebooks[1].astype(mx.float32),  # level 2
            codebooks[2].astype(mx.float32),  # level 3
        ],
        output_shapes=[
            (N, n_blocks * 8),     # indices_l1
            (N, n_blocks * 4),     # indices_l2
            (N, n_blocks * 2),     # indices_l3
            (N, n_blocks),         # radii
        ],
        output_dtypes=[mx.uint8, mx.uint8, mx.uint8, mx.float16],
        grid=(N, 1, 1),
        threadgroup=(1, 1, 1),
    )

    return {
        "indices_l1": outputs[0],
        "indices_l2": outputs[1],
        "indices_l3": outputs[2],
        "radii": outputs[3],
    }
