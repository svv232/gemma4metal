"""
Block-level fused PolarQuant scoring kernel.

Instead of reconstructing all key vectors then computing matmul,
this kernel fuses polar_inverse + inner_product at the block level:

  <Pq, y_recon> = Σ_b <Pq[b*16:(b+1)*16], polar_inverse(block_b)>

Each thread handles one (query, key) pair, iterating over blocks.
Avoids materializing the full (Nk, D) reconstruction matrix.
"""

import mlx.core as mx
import numpy as np

_block_score_kernel = mx.fast.metal_kernel(
    name="polar_block_score",
    input_names=[
        "pq",                # (Nq, D) preconditioned query (P @ q)
        "codebook_l1", "codebook_l2", "codebook_l3", "codebook_l4",
        "indices_l1",        # (Nk * n_blocks * 8) uint32
        "indices_l2",        # (Nk * n_blocks * 4) uint32
        "indices_l3",        # (Nk * n_blocks * 2) uint32
        "indices_l4",        # (Nk * n_blocks) uint32
        "radii",             # (Nk * n_blocks) float32
    ],
    output_names=["scores"],
    header="""
constant constexpr uint BLOCK_SIZE = 16;
constant constexpr uint N_BLOCKS = 8;  // D/BLOCK_SIZE = 128/16
constant constexpr uint DIM = 128;
""",
    source="""
        uint qi = thread_position_in_grid.x;
        uint ki = thread_position_in_grid.y;

        float total = 0.0f;

        for (uint blk = 0; blk < N_BLOCKS; blk++) {
            uint block_idx = ki * N_BLOCKS + blk;

            // Polar inverse for this block
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

            // Dot product with query slice
            uint base = blk * BLOCK_SIZE;
            for (uint j = 0; j < BLOCK_SIZE; j++) {
                total += r[j] * pq[qi * DIM + base + j];
            }
        }

        uint Nk = radii_shape[0] / N_BLOCKS;
        scores[qi * Nk + ki] = total;
    """,
)


def polar_block_score(
    pq: mx.array,              # (Nq, D) preconditioned query
    codebooks: list,           # [l1, l2, l3, l4]
    angle_indices: list,       # [l1, l2, l3, l4] flat
    radii: mx.array,           # (Nk * n_blocks,) float32
    n_keys: int,
) -> mx.array:
    """
    Block-fused PolarQuant scoring.

    Returns:
        scores: (Nq, Nk) — polar component of attention scores
    """
    Nq = pq.shape[0]

    outputs = _block_score_kernel(
        inputs=[
            pq.astype(mx.float32),
            codebooks[0], codebooks[1], codebooks[2], codebooks[3],
            angle_indices[0].astype(mx.uint32),
            angle_indices[1].astype(mx.uint32),
            angle_indices[2].astype(mx.uint32),
            angle_indices[3].astype(mx.uint32),
            radii.astype(mx.float32),
        ],
        output_shapes=[(Nq, n_keys)],
        output_dtypes=[mx.float32],
        grid=(Nq, n_keys, 1),
        threadgroup=(1, 1, 1),
    )
    return outputs[0]


# ──────────────────────────────────────────────────────────────────────
# Batched version: all heads in one dispatch
# Grid: (Nq, Nk, BH) — each thread handles one (query, key, head) triple
# ──────────────────────────────────────────────────────────────────────

_batched_block_score_kernel = mx.fast.metal_kernel(
    name="polar_batched_block_score",
    input_names=[
        "pq",                # (BH * Nq, D) all preconditioned queries
        "codebook_l1", "codebook_l2", "codebook_l3", "codebook_l4",
        "indices_l1",        # (BH * Nk * n_blocks * 8) uint32
        "indices_l2",        # (BH * Nk * n_blocks * 4) uint32
        "indices_l3",        # (BH * Nk * n_blocks * 2) uint32
        "indices_l4",        # (BH * Nk * n_blocks) uint32
        "radii",             # (BH * Nk * n_blocks) float32
        "params",            # (3,) uint32: [Nq, Nk, n_blocks]
    ],
    output_names=["scores"],
    header="""
constant constexpr uint BLOCK_SIZE = 16;
constant constexpr uint DIM = 128;
""",
    source="""
        uint qi_local = thread_position_in_grid.x;
        uint ki = thread_position_in_grid.y;
        uint bh = thread_position_in_grid.z;

        uint Nq = params[0];
        uint Nk = params[1];
        uint N_BLOCKS = params[2];

        if (qi_local >= Nq || ki >= Nk) return;

        // Global query index for this head
        uint qi_global = bh * Nq + qi_local;
        // Global key index for this head
        uint ki_global = bh * Nk + ki;

        float total = 0.0f;

        for (uint blk = 0; blk < N_BLOCKS; blk++) {
            uint block_idx = ki_global * N_BLOCKS + blk;

            float r[16];
            r[0] = radii[block_idx];

            // Level 4
            {
                uint idx = indices_l4[block_idx];
                float theta = codebook_l4[idx];
                float r0 = r[0];
                r[0] = r0 * metal::cos(theta);
                r[1] = r0 * metal::sin(theta);
            }
            // Level 3
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
            // Level 2
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
            // Level 1
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

            uint base = blk * BLOCK_SIZE;
            for (uint j = 0; j < BLOCK_SIZE; j++) {
                total += r[j] * pq[qi_global * DIM + base + j];
            }
        }

        scores[bh * Nq * Nk + qi_local * Nk + ki] = total;
    """,
)


# ──────────────────────────────────────────────────────────────────────
# SIMD-parallel version: 32 threads cooperate on each (query, key) pair
# Each lane handles a different subset of the 128 coordinates
# Lane i processes coords [i*4 : (i+1)*4] across all blocks
# ──────────────────────────────────────────────────────────────────────

_simd_block_score_kernel = mx.fast.metal_kernel(
    name="polar_simd_block_score",
    input_names=[
        "pq",
        "codebook_l1", "codebook_l2", "codebook_l3", "codebook_l4",
        "indices_l1", "indices_l2", "indices_l3", "indices_l4",
        "radii",
        "params",
    ],
    output_names=["scores"],
    header="""
constant constexpr uint BLOCK_SIZE = 16;
constant constexpr uint DIM = 128;
constant constexpr uint N_BLOCKS = 8;
""",
    source="""
        // Grid: (Nq * 32, Nk, BH)
        uint global_x = thread_position_in_grid.x;
        uint qi_local = global_x / 32;
        uint lane = global_x % 32;
        uint ki = thread_position_in_grid.y;
        uint bh = thread_position_in_grid.z;

        uint Nq = params[0];
        uint Nk = params[1];

        if (qi_local >= Nq || ki >= Nk) return;

        uint qi_global = bh * Nq + qi_local;
        uint ki_global = bh * Nk + ki;

        // Each lane handles a subset of the 128 total coordinates
        // Lane l handles coordinates: for each block, coords assigned to this lane
        // With 32 lanes and 128 coords: lane l handles coords l*4..(l+1)*4-1
        // But coords are in blocks of 16, so lane l within block b handles:
        //   block b = l / 2, local coords (l%2)*8..(l%2+1)*8-1 ... no this is complex

        // Simpler: each lane handles one block (8 blocks, 32 lanes = 4 lanes/block)
        // Lane l handles block l/4, and within that block handles coords (l%4)*4..(l%4+1)*4
        // BUT: polar inverse must be computed for the whole block first

        // Best approach: lane handles one block at a time, SIMD reduces across blocks
        // With 8 blocks and 32 lanes: lanes 0-7 each do one block, lanes 8-31 idle
        // Then simd_sum accumulates

        float partial = 0.0f;

        if (lane < N_BLOCKS) {
            uint blk = lane;
            uint block_idx = ki_global * N_BLOCKS + blk;

            float r[16];
            r[0] = radii[block_idx];

            // Level 4
            {
                uint idx = indices_l4[block_idx];
                float theta = codebook_l4[idx];
                float r0 = r[0];
                r[0] = r0 * metal::cos(theta);
                r[1] = r0 * metal::sin(theta);
            }
            // Level 3
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
            // Level 2
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
            // Level 1
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

            uint base = blk * BLOCK_SIZE;
            for (uint j = 0; j < BLOCK_SIZE; j++) {
                partial += r[j] * pq[qi_global * DIM + base + j];
            }
        }

        // SIMD reduce across 32 lanes (8 active + 24 zeros)
        float total = simd_sum(partial);

        if (lane == 0) {
            scores[bh * Nq * Nk + qi_local * Nk + ki] = total;
        }
    """,
)


def polar_simd_block_score(
    pq: mx.array,
    codebooks: list,
    angle_indices: list,
    radii: mx.array,
    n_queries: int,
    n_keys: int,
    n_heads: int,
) -> mx.array:
    """SIMD-parallel batched block-fused scoring."""
    n_blocks = 128 // 16
    params = mx.array([n_queries, n_keys, n_blocks], dtype=mx.uint32)

    outputs = _simd_block_score_kernel(
        inputs=[
            pq.reshape(n_heads * n_queries, 128).astype(mx.float32),
            codebooks[0], codebooks[1], codebooks[2], codebooks[3],
            angle_indices[0].astype(mx.uint32),
            angle_indices[1].astype(mx.uint32),
            angle_indices[2].astype(mx.uint32),
            angle_indices[3].astype(mx.uint32),
            radii.astype(mx.float32),
            params,
        ],
        output_shapes=[(n_heads * n_queries * n_keys,)],
        output_dtypes=[mx.float32],
        grid=(n_queries * 32, n_keys, n_heads),
        threadgroup=(32, 1, 1),
    )
    return outputs[0].reshape(n_heads, n_queries, n_keys)


# ──────────────────────────────────────────────────────────────────────
# Threadgroup-optimized version: multiple keys per threadgroup,
# query vector cached in threadgroup memory
# Grid: (Nq, ceil(Nk/KEYS_PER_TG), BH)
# Threadgroup: (KEYS_PER_TG, 1, 1)
# ──────────────────────────────────────────────────────────────────────

KEYS_PER_TG = 64  # keys processed per threadgroup

_tg_block_score_kernel = mx.fast.metal_kernel(
    name="polar_tg_block_score",
    input_names=[
        "pq",
        "codebook_l1", "codebook_l2", "codebook_l3", "codebook_l4",
        "indices_l1", "indices_l2", "indices_l3", "indices_l4",
        "radii",
        "params",
    ],
    output_names=["scores"],
    header="""
constant constexpr uint BLOCK_SIZE = 16;
constant constexpr uint DIM = 128;
constant constexpr uint N_BLOCKS = 8;
constant constexpr uint KEYS_PER_TG = 64;
""",
    source="""
        uint global_x = thread_position_in_grid.x;
        uint qi = global_x / KEYS_PER_TG;
        uint lane = global_x % KEYS_PER_TG;
        uint ki_tile = thread_position_in_grid.y;
        uint bh = thread_position_in_grid.z;

        uint Nq = params[0];
        uint Nk = params[1];

        // Cache query vector in threadgroup memory
        threadgroup float shared_pq[DIM];
        uint qi_global = bh * Nq + qi;
        for (uint i = lane; i < DIM; i += KEYS_PER_TG) {
            shared_pq[i] = pq[qi_global * DIM + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each thread processes one key
        uint ki = ki_tile * KEYS_PER_TG + lane;
        if (ki >= Nk) return;

        uint ki_global = bh * Nk + ki;

        float total = 0.0f;

        for (uint blk = 0; blk < N_BLOCKS; blk++) {
            uint block_idx = ki_global * N_BLOCKS + blk;
            float r[16];
            r[0] = radii[block_idx];

            // Level 4
            {
                uint idx = indices_l4[block_idx];
                float theta = codebook_l4[idx];
                float r0 = r[0];
                r[0] = r0 * metal::cos(theta);
                r[1] = r0 * metal::sin(theta);
            }
            // Level 3
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
            // Level 2
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
            // Level 1
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

            uint base = blk * BLOCK_SIZE;
            for (uint j = 0; j < BLOCK_SIZE; j++) {
                total += r[j] * shared_pq[base + j];
            }
        }

        scores[bh * Nq * Nk + qi * Nk + ki] = total;
    """,
)


def polar_tg_block_score(
    pq: mx.array,
    codebooks: list,
    angle_indices: list,
    radii: mx.array,
    n_queries: int,
    n_keys: int,
    n_heads: int,
) -> mx.array:
    """Threadgroup-optimized batched block-fused scoring."""
    n_blocks = 128 // 16
    params = mx.array([n_queries, n_keys, n_blocks], dtype=mx.uint32)
    keys_per_tg = KEYS_PER_TG
    n_key_tiles = (n_keys + keys_per_tg - 1) // keys_per_tg

    outputs = _tg_block_score_kernel(
        inputs=[
            pq.reshape(n_heads * n_queries, 128).astype(mx.float32),
            codebooks[0], codebooks[1], codebooks[2], codebooks[3],
            angle_indices[0].astype(mx.uint32),
            angle_indices[1].astype(mx.uint32),
            angle_indices[2].astype(mx.uint32),
            angle_indices[3].astype(mx.uint32),
            radii.astype(mx.float32),
            params,
        ],
        output_shapes=[(n_heads * n_queries * n_keys,)],
        output_dtypes=[mx.float32],
        grid=(n_queries * keys_per_tg, n_key_tiles, n_heads),
        threadgroup=(keys_per_tg, 1, 1),
    )
    return outputs[0].reshape(n_heads, n_queries, n_keys)


# ──────────────────────────────────────────────────────────────────────
# Multi-key version: each thread processes KEYS_PER_THREAD keys
# Amortizes query vector load across multiple keys
# ──────────────────────────────────────────────────────────────────────

KEYS_PER_THREAD = 4

_multikey_block_score_kernel = mx.fast.metal_kernel(
    name="polar_multikey_block_score",
    input_names=[
        "pq",
        "codebook_l1", "codebook_l2", "codebook_l3", "codebook_l4",
        "indices_l1", "indices_l2", "indices_l3", "indices_l4",
        "radii",
        "params",
    ],
    output_names=["scores"],
    header="""
constant constexpr uint BLOCK_SIZE = 16;
constant constexpr uint DIM = 128;
constant constexpr uint N_BLOCKS = 8;
constant constexpr uint KPT = 4;  // keys per thread
""",
    source="""
        uint qi = thread_position_in_grid.x;
        uint ki_base = thread_position_in_grid.y * KPT;
        uint bh = thread_position_in_grid.z;

        uint Nq = params[0];
        uint Nk = params[1];

        if (qi >= Nq) return;

        uint qi_global = bh * Nq + qi;

        // Load query once, process KPT keys
        float pq_local[DIM];
        for (uint d = 0; d < DIM; d++) {
            pq_local[d] = pq[qi_global * DIM + d];
        }

        for (uint kk = 0; kk < KPT; kk++) {
            uint ki = ki_base + kk;
            if (ki >= Nk) return;

            uint ki_global = bh * Nk + ki;
            float total = 0.0f;

            for (uint blk = 0; blk < N_BLOCKS; blk++) {
                uint block_idx = ki_global * N_BLOCKS + blk;
                float r[16];
                r[0] = radii[block_idx];

                // Level 4
                {
                    uint idx = indices_l4[block_idx];
                    float theta = codebook_l4[idx];
                    float r0 = r[0];
                    r[0] = r0 * metal::cos(theta);
                    r[1] = r0 * metal::sin(theta);
                }
                // Level 3
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
                // Level 2
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
                // Level 1
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

                uint base = blk * BLOCK_SIZE;
                for (uint j = 0; j < BLOCK_SIZE; j++) {
                    total += r[j] * pq_local[base + j];
                }
            }

            scores[bh * Nq * Nk + qi * Nk + ki] = total;
        }
    """,
)


def polar_multikey_block_score(
    pq, codebooks, angle_indices, radii, n_queries, n_keys, n_heads,
):
    """Multi-key batched block-fused scoring."""
    n_blocks = 128 // 16
    params = mx.array([n_queries, n_keys, n_blocks], dtype=mx.uint32)
    n_key_groups = (n_keys + KEYS_PER_THREAD - 1) // KEYS_PER_THREAD

    outputs = _multikey_block_score_kernel(
        inputs=[
            pq.reshape(n_heads * n_queries, 128).astype(mx.float32),
            codebooks[0], codebooks[1], codebooks[2], codebooks[3],
            angle_indices[0].astype(mx.uint32),
            angle_indices[1].astype(mx.uint32),
            angle_indices[2].astype(mx.uint32),
            angle_indices[3].astype(mx.uint32),
            radii.astype(mx.float32),
            params,
        ],
        output_shapes=[(n_heads * n_queries * n_keys,)],
        output_dtypes=[mx.float32],
        grid=(n_queries, n_key_groups, n_heads),
        threadgroup=(1, 1, 1),
    )
    return outputs[0].reshape(n_heads, n_queries, n_keys)


def polar_batched_block_score(
    pq: mx.array,              # (BH, Nq, D) or (BH * Nq, D)
    codebooks: list,
    angle_indices: list,       # flat for all heads
    radii: mx.array,           # flat for all heads
    n_queries: int,
    n_keys: int,
    n_heads: int,              # B * H
) -> mx.array:
    """
    Batched block-fused PolarQuant scoring — all heads in one dispatch.

    Returns:
        scores: (BH, Nq, Nk)
    """
    n_blocks = 128 // 16  # D // BLOCK_SIZE
    params = mx.array([n_queries, n_keys, n_blocks], dtype=mx.uint32)

    outputs = _batched_block_score_kernel(
        inputs=[
            pq.reshape(n_heads * n_queries, 128).astype(mx.float32),
            codebooks[0], codebooks[1], codebooks[2], codebooks[3],
            angle_indices[0].astype(mx.uint32),
            angle_indices[1].astype(mx.uint32),
            angle_indices[2].astype(mx.uint32),
            angle_indices[3].astype(mx.uint32),
            radii.astype(mx.float32),
            params,
        ],
        output_shapes=[(n_heads * n_queries * n_keys,)],
        output_dtypes=[mx.float32],
        grid=(n_queries, n_keys, n_heads),
        threadgroup=(1, 1, 1),
    )
    return outputs[0].reshape(n_heads, n_queries, n_keys)
