"""
Fused multi-head compressed SDPA via mx.fast.metal_kernel.

Single dispatch handles ALL heads. Each thread computes one head's
attention output by streaming through compressed keys with online softmax.

This is the compressed-domain equivalent of MLX's sdpa_vector kernel.
"""

import mlx.core as mx

_sdpa_compressed_kernel = mx.fast.metal_kernel(
    name="sdpa_compressed_multihead",
    input_names=[
        "queries",    # (BH, D) preconditioned queries
        "cb_l1", "cb_l2", "cb_l3", "cb_l4",
        "indices_l1", # (BH * N * n_blocks * 8) uint8
        "indices_l2", # (BH * N * n_blocks * 4) uint8
        "indices_l3", # (BH * N * n_blocks * 2) uint8
        "indices_l4", # (BH * N * n_blocks) uint8
        "radii",      # (BH * N * n_blocks) float16
        "values",     # (BH, N, D) float32
        "params",     # [N, D, n_blocks, scale] float32
    ],
    output_names=["output"],  # (BH, D)
    header="""
constant constexpr uint BS = 16;
""",
    source="""
        // Each thread = one head's full attention computation
        uint head_idx = thread_position_in_grid.x;

        uint N = as_type<uint>(params[0]);
        uint D = as_type<uint>(params[1]);
        uint NB = as_type<uint>(params[2]);
        float scale = params[3];

        // Online softmax state
        float max_s = -1e30f;
        float sum_e = 0.0f;

        // Output accumulator (in registers)
        float acc[128]; // max D
        for (uint d = 0; d < D; d++) acc[d] = 0.0f;

        // Load query (pre-scaled)
        float q[128];
        for (uint d = 0; d < D; d++) {
            q[d] = scale * queries[head_idx * D + d];
        }

        // Stream through all keys for this head
        uint head_key_base = head_idx * N;

        for (uint ki = 0; ki < N; ki++) {
            // Reconstruct key from compressed indices
            float score = 0.0f;

            for (uint blk = 0; blk < NB; blk++) {
                uint block_idx = (head_key_base + ki) * NB + blk;

                // Polar inverse for this block
                float r[16];
                r[0] = static_cast<float>(radii[block_idx]);

                // Level 4
                { uint idx = indices_l4[block_idx];
                  float th = cb_l4[idx]; float r0 = r[0];
                  r[0] = r0 * metal::cos(th); r[1] = r0 * metal::sin(th); }
                // Level 3
                { float t[4];
                  for (uint j=0;j<2;j++) { uint idx=indices_l3[block_idx*2+j]; float th=cb_l3[idx];
                    t[2*j]=r[j]*metal::cos(th); t[2*j+1]=r[j]*metal::sin(th); }
                  for (uint j=0;j<4;j++) r[j]=t[j]; }
                // Level 2
                { float t[8];
                  for (uint j=0;j<4;j++) { uint idx=indices_l2[block_idx*4+j]; float th=cb_l2[idx];
                    t[2*j]=r[j]*metal::cos(th); t[2*j+1]=r[j]*metal::sin(th); }
                  for (uint j=0;j<8;j++) r[j]=t[j]; }
                // Level 1
                { float t[16];
                  for (uint j=0;j<8;j++) { uint idx=indices_l1[block_idx*8+j]; float th=cb_l1[idx];
                    t[2*j]=r[j]*metal::cos(th); t[2*j+1]=r[j]*metal::sin(th); }
                  for (uint j=0;j<16;j++) r[j]=t[j]; }

                // Dot product with query
                uint base = blk * BS;
                for (uint j = 0; j < BS; j++) score += q[base+j] * r[j];
            }

            // Online softmax
            float old_max = max_s;
            if (score > max_s) max_s = score;
            float factor = metal::exp(old_max - max_s);
            float exp_s = metal::exp(score - max_s);
            sum_e = sum_e * factor + exp_s;

            // Accumulate weighted values
            uint v_off = (head_key_base + ki) * D;
            for (uint d = 0; d < D; d++) {
                acc[d] = acc[d] * factor + exp_s * values[v_off + d];
            }
        }

        // Normalize
        float inv = 1.0f / (sum_e + 1e-8f);
        for (uint d = 0; d < D; d++) {
            output[head_idx * D + d] = acc[d] * inv;
        }
    """,
)


def sdpa_compressed(
    queries_preconditioned,  # (BH, Nq, D)
    codebooks,
    indices,                 # [l1, l2, l3, l4] flat
    radii,
    values,                  # (BH, N, D)
    n_keys, embed_dim, n_blocks, scale, n_heads,
):
    """Multi-head compressed SDPA in one dispatch."""
    params = mx.array([
        mx.array(n_keys, dtype=mx.uint32).view(mx.float32).item(),
        mx.array(embed_dim, dtype=mx.uint32).view(mx.float32).item(),
        mx.array(n_blocks, dtype=mx.uint32).view(mx.float32).item(),
        scale,
    ], dtype=mx.float32)

    BH = n_heads
    outputs = _sdpa_compressed_kernel(
        inputs=[
            queries_preconditioned.reshape(BH, embed_dim).astype(mx.float32),
            codebooks[0], codebooks[1], codebooks[2], codebooks[3],
            indices[0].astype(mx.uint8),
            indices[1].astype(mx.uint8),
            indices[2].astype(mx.uint8),
            indices[3].astype(mx.uint8),
            radii.astype(mx.float16),
            values.reshape(BH * n_keys, embed_dim).astype(mx.float32),
            params,
        ],
        output_shapes=[(BH * embed_dim,)],
        output_dtypes=[mx.float32],
        grid=(BH, 1, 1),
        threadgroup=(1, 1, 1),
    )
    return outputs[0].reshape(BH, 1, embed_dim)
