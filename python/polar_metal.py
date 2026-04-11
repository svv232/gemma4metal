"""
Metal-accelerated PolarQuant kernels via MLX custom Metal kernel API.

Forward: vector → angles + radius (for quantization during prefill)
Inverse: angle indices + radius → reconstructed vector (for scoring during decode)
"""

import mlx.core as mx

# ──────────────────────────────────────────────────────────────────────
# PolarQuant Inverse: angle indices + radius → reconstructed vector
#
# For block_size=16, n_levels=4:
#   Level 4: 1 angle → radius splits into 2 values
#   Level 3: 2 angles → 2 values split into 4
#   Level 2: 4 angles → 4 values split into 8
#   Level 1: 8 angles → 8 values split into 16
#
# Each thread reconstructs one block of 16 coordinates.
# ──────────────────────────────────────────────────────────────────────

_inverse_kernel = mx.fast.metal_kernel(
    name="polar_inverse",
    input_names=["codebook_l1", "codebook_l2", "codebook_l3", "codebook_l4",
                 "indices_l1", "indices_l2", "indices_l3", "indices_l4",
                 "radii"],
    output_names=["output"],
    source="""
        // Each thread reconstructs one block of BLOCK_SIZE=16 coordinates
        uint block_idx = thread_position_in_grid.x;

        // Read radius for this block
        float r[16];  // max block size
        r[0] = static_cast<float>(radii[block_idx]);

        // Level 4: 1 angle, radius → 2 values
        {
            uint idx = indices_l4[block_idx];
            float theta = codebook_l4[idx];
            float c = metal::cos(theta);
            float s = metal::sin(theta);
            float r0 = r[0];
            r[0] = r0 * c;
            r[1] = r0 * s;
        }

        // Level 3: 2 angles, 2 values → 4 values
        {
            float tmp[4];
            for (uint j = 0; j < 2; j++) {
                uint idx = indices_l3[block_idx * 2 + j];
                float theta = codebook_l3[idx];
                float c = metal::cos(theta);
                float s = metal::sin(theta);
                tmp[2*j]   = r[j] * c;
                tmp[2*j+1] = r[j] * s;
            }
            for (uint j = 0; j < 4; j++) r[j] = tmp[j];
        }

        // Level 2: 4 angles, 4 values → 8 values
        {
            float tmp[8];
            for (uint j = 0; j < 4; j++) {
                uint idx = indices_l2[block_idx * 4 + j];
                float theta = codebook_l2[idx];
                float c = metal::cos(theta);
                float s = metal::sin(theta);
                tmp[2*j]   = r[j] * c;
                tmp[2*j+1] = r[j] * s;
            }
            for (uint j = 0; j < 8; j++) r[j] = tmp[j];
        }

        // Level 1: 8 angles, 8 values → 16 values
        {
            float tmp[16];
            for (uint j = 0; j < 8; j++) {
                uint idx = indices_l1[block_idx * 8 + j];
                float theta = codebook_l1[idx];
                float c = metal::cos(theta);
                float s = metal::sin(theta);
                tmp[2*j]   = r[j] * c;
                tmp[2*j+1] = r[j] * s;
            }
            for (uint j = 0; j < 16; j++) r[j] = tmp[j];
        }

        // Write output block
        for (uint j = 0; j < 16; j++) {
            output[block_idx * 16 + j] = r[j];
        }
    """,
)


def polar_inverse_metal(
    codebooks: list,       # [cb_l1, cb_l2, cb_l3, cb_l4] float32 arrays
    angle_indices: list,   # [idx_l1, idx_l2, idx_l3, idx_l4] — flattened
    radii: mx.array,       # (N_blocks,) float16 or float32
    block_size: int = 16,
) -> mx.array:
    """
    Metal-accelerated PolarQuant inverse transform.

    Args:
        codebooks: list of 4 codebook arrays (one per level)
        angle_indices: list of 4 index arrays, each flattened to (N_blocks * angles_per_level,)
        radii: (N_blocks,) final radius per block

    Returns:
        output: (N_blocks * block_size,) reconstructed coordinates
    """
    N_blocks = radii.shape[0]
    radii_f32 = radii.astype(mx.float32) if radii.dtype != mx.float32 else radii

    outputs = _inverse_kernel(
        inputs=[
            codebooks[0], codebooks[1], codebooks[2], codebooks[3],
            angle_indices[0], angle_indices[1], angle_indices[2], angle_indices[3],
            radii_f32,
        ],
        output_shapes=[(N_blocks * block_size,)],
        output_dtypes=[mx.float32],
        grid=(N_blocks, 1, 1),
        threadgroup=(1, 1, 1),
    )
    return outputs[0]


# ──────────────────────────────────────────────────────────────────────
# PolarQuant Forward: vector → angles + radius
#
# Each thread processes one block of 16 coordinates.
# Level 1: 16 coords → 8 (angle, radius) pairs
# Level 2: 8 radii → 4 (angle, radius) pairs
# Level 3: 4 radii → 2 (angle, radius) pairs
# Level 4: 2 radii → 1 (angle, radius) pair
# ──────────────────────────────────────────────────────────────────────

_forward_kernel = mx.fast.metal_kernel(
    name="polar_forward",
    input_names=["input_data"],
    output_names=["angles_l1", "angles_l2", "angles_l3", "angles_l4", "radii"],
    source="""
        uint block_idx = thread_position_in_grid.x;

        // Load block into registers
        float r[16];
        for (uint j = 0; j < 16; j++) {
            r[j] = input_data[block_idx * 16 + j];
        }

        // Level 1: 16 → 8 (angles in [0, 2pi), wrap negative atan2)
        {
            float tmp[8];
            for (uint j = 0; j < 8; j++) {
                float a = r[2*j];
                float b = r[2*j+1];
                float angle = metal::atan2(b, a);
                // Wrap to [0, 2pi)
                if (angle < 0.0f) angle += 2.0f * M_PI_F;
                angles_l1[block_idx * 8 + j] = angle;
                tmp[j] = metal::sqrt(a*a + b*b);
            }
            for (uint j = 0; j < 8; j++) r[j] = tmp[j];
        }

        // Level 2: 8 → 4 (angles in [0, pi/2] since inputs are norms)
        {
            float tmp[4];
            for (uint j = 0; j < 4; j++) {
                float a = r[2*j];
                float b = r[2*j+1];
                angles_l2[block_idx * 4 + j] = metal::atan2(b, a);
                tmp[j] = metal::sqrt(a*a + b*b);
            }
            for (uint j = 0; j < 4; j++) r[j] = tmp[j];
        }

        // Level 3: 4 → 2
        {
            float tmp[2];
            for (uint j = 0; j < 2; j++) {
                float a = r[2*j];
                float b = r[2*j+1];
                angles_l3[block_idx * 2 + j] = metal::atan2(b, a);
                tmp[j] = metal::sqrt(a*a + b*b);
            }
            r[0] = tmp[0];
            r[1] = tmp[1];
        }

        // Level 4: 2 → 1
        {
            float a = r[0];
            float b = r[1];
            angles_l4[block_idx] = metal::atan2(b, a);
            radii[block_idx] = metal::sqrt(a*a + b*b);
        }
    """,
)


def polar_forward_metal(input_data: mx.array) -> tuple:
    """
    Metal-accelerated PolarQuant forward transform.

    Args:
        input_data: (N_blocks * 16,) flat input coordinates

    Returns:
        (angles_l1, angles_l2, angles_l3, angles_l4, radii)
    """
    N_blocks = input_data.shape[0] // 16

    outputs = _forward_kernel(
        inputs=[input_data.astype(mx.float32)],
        output_shapes=[
            (N_blocks * 8,),   # angles_l1
            (N_blocks * 4,),   # angles_l2
            (N_blocks * 2,),   # angles_l3
            (N_blocks,),       # angles_l4
            (N_blocks,),       # radii
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32, mx.float32],
        grid=(N_blocks, 1, 1),
        threadgroup=(1, 1, 1),
    )
    return outputs
