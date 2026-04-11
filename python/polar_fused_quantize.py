"""
Fused Metal kernel: polar_forward + codebook_nearest in one dispatch.

Eliminates the Python argmin bottleneck (5.1ms) by doing codebook lookup
inside the forward polar transform kernel.
"""

import mlx.core as mx

_fused_polar_quantize_kernel = mx.fast.metal_kernel(
    name="polar_fused_quantize",
    input_names=["input_data", "cb_l1", "cb_l2", "cb_l3", "cb_l4"],
    output_names=["indices_l1", "indices_l2", "indices_l3", "indices_l4", "radii"],
    source="""
        uint block_idx = thread_position_in_grid.x;

        float r[16];
        for (uint j = 0; j < 16; j++) {
            r[j] = input_data[block_idx * 16 + j];
        }

        uint n_cb1 = cb_l1_shape[0];
        uint n_cb2 = cb_l2_shape[0];
        uint n_cb3 = cb_l3_shape[0];
        uint n_cb4 = cb_l4_shape[0];

        // Level 1: 16 → 8, angles in [0, 2π), nearest codebook
        {
            float tmp[8];
            for (uint j = 0; j < 8; j++) {
                float a = r[2*j], b = r[2*j+1];
                float angle = metal::atan2(b, a);
                if (angle < 0.0f) angle += 2.0f * M_PI_F;

                // Nearest codebook entry
                float best = 1e10f;
                uint8_t idx = 0;
                for (uint c = 0; c < n_cb1; c++) {
                    float d = metal::abs(angle - cb_l1[c]);
                    if (d < best) { best = d; idx = c; }
                }
                indices_l1[block_idx * 8 + j] = idx;
                tmp[j] = metal::sqrt(a*a + b*b);
            }
            for (uint j = 0; j < 8; j++) r[j] = tmp[j];
        }

        // Level 2: 8 → 4
        {
            float tmp[4];
            for (uint j = 0; j < 4; j++) {
                float a = r[2*j], b = r[2*j+1];
                float angle = metal::atan2(b, a);
                float best = 1e10f;
                uint8_t idx = 0;
                for (uint c = 0; c < n_cb2; c++) {
                    float d = metal::abs(angle - cb_l2[c]);
                    if (d < best) { best = d; idx = c; }
                }
                indices_l2[block_idx * 4 + j] = idx;
                tmp[j] = metal::sqrt(a*a + b*b);
            }
            for (uint j = 0; j < 4; j++) r[j] = tmp[j];
        }

        // Level 3: 4 → 2
        {
            float tmp[2];
            for (uint j = 0; j < 2; j++) {
                float a = r[2*j], b = r[2*j+1];
                float angle = metal::atan2(b, a);
                float best = 1e10f;
                uint8_t idx = 0;
                for (uint c = 0; c < n_cb3; c++) {
                    float d = metal::abs(angle - cb_l3[c]);
                    if (d < best) { best = d; idx = c; }
                }
                indices_l3[block_idx * 2 + j] = idx;
                tmp[j] = metal::sqrt(a*a + b*b);
            }
            r[0] = tmp[0]; r[1] = tmp[1];
        }

        // Level 4: 2 → 1
        {
            float a = r[0], b = r[1];
            float angle = metal::atan2(b, a);
            float best = 1e10f;
            uint8_t idx = 0;
            for (uint c = 0; c < n_cb4; c++) {
                float d = metal::abs(angle - cb_l4[c]);
                if (d < best) { best = d; idx = c; }
            }
            indices_l4[block_idx] = idx;
            radii[block_idx] = static_cast<half>(metal::sqrt(a*a + b*b));
        }
    """,
)


def fused_polar_quantize(input_data, codebooks):
    """Fused polar forward + codebook quantize in one Metal dispatch."""
    N_blocks = input_data.shape[0] // 16

    outputs = _fused_polar_quantize_kernel(
        inputs=[
            input_data.astype(mx.float32),
            codebooks[0], codebooks[1], codebooks[2], codebooks[3],
        ],
        output_shapes=[
            (N_blocks * 8,),   # indices_l1
            (N_blocks * 4,),   # indices_l2
            (N_blocks * 2,),   # indices_l3
            (N_blocks,),       # indices_l4
            (N_blocks,),       # radii
        ],
        output_dtypes=[mx.uint8, mx.uint8, mx.uint8, mx.uint8, mx.float16],
        grid=(N_blocks, 1, 1),
        threadgroup=(1, 1, 1),
    )
    return outputs
