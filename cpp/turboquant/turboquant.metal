// TurboQuant fused Metal kernel: forward_polar + codebook_nearest
//
// Compiled into metallib at build time, loaded by eval_gpu.

#include <metal_stdlib>
using namespace metal;

kernel void polar_fused_quantize(
    device const float* input        [[buffer(0)]],   // (N_blocks * block_size)
    device const float* cb_l1        [[buffer(1)]],   // level 1 codebook
    device const float* cb_l2        [[buffer(2)]],   // level 2 codebook
    device const float* cb_l3        [[buffer(3)]],   // level 3 codebook
    device const float* cb_l4        [[buffer(4)]],   // level 4 codebook
    device const uint& n_cb1         [[buffer(5)]],   // codebook sizes
    device const uint& n_cb2         [[buffer(6)]],
    device const uint& n_cb3         [[buffer(7)]],
    device const uint& n_cb4         [[buffer(8)]],
    device uint8_t* indices_l1       [[buffer(9)]],
    device uint8_t* indices_l2       [[buffer(10)]],
    device uint8_t* indices_l3       [[buffer(11)]],
    device uint8_t* indices_l4       [[buffer(12)]],
    device half* radii               [[buffer(13)]],
    uint block_idx                   [[thread_position_in_grid]]
) {
    float r[16];
    for (uint j = 0; j < 16; j++) {
        r[j] = input[block_idx * 16 + j];
    }

    // Level 1: 16 → 8
    {
        float tmp[8];
        for (uint j = 0; j < 8; j++) {
            float a = r[2*j], b = r[2*j+1];
            float angle = metal::atan2(b, a);
            if (angle < 0.0f) angle += 2.0f * M_PI_F;
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
}

kernel void polar_inverse_dequantize(
    device const float* cb_l1        [[buffer(0)]],
    device const float* cb_l2        [[buffer(1)]],
    device const float* cb_l3        [[buffer(2)]],
    device const float* cb_l4        [[buffer(3)]],
    device const uint8_t* indices_l1 [[buffer(4)]],
    device const uint8_t* indices_l2 [[buffer(5)]],
    device const uint8_t* indices_l3 [[buffer(6)]],
    device const uint8_t* indices_l4 [[buffer(7)]],
    device const half* radii         [[buffer(8)]],
    device float* output             [[buffer(9)]],
    uint block_idx                   [[thread_position_in_grid]]
) {
    float r[16];
    r[0] = static_cast<float>(radii[block_idx]);

    // Level 4: 1 → 2
    {
        uint idx = indices_l4[block_idx];
        float theta = cb_l4[idx];
        float r0 = r[0];
        r[0] = r0 * metal::cos(theta);
        r[1] = r0 * metal::sin(theta);
    }

    // Level 3: 2 → 4
    {
        float tmp[4];
        for (uint j = 0; j < 2; j++) {
            uint idx = indices_l3[block_idx * 2 + j];
            float theta = cb_l3[idx];
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
            float theta = cb_l2[idx];
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
            float theta = cb_l1[idx];
            tmp[2*j]   = r[j] * metal::cos(theta);
            tmp[2*j+1] = r[j] * metal::sin(theta);
        }
        for (uint j = 0; j < 16; j++) r[j] = tmp[j];
    }

    for (uint j = 0; j < 16; j++) {
        output[block_idx * 16 + j] = r[j];
    }
}

// Simple sequential compressed SDPA: one thread per head
kernel void sdpa_tq_sequential(
    device const float* queries      [[buffer(0)]],   // (BH, D)
    device const uint8_t* idx_l1     [[buffer(1)]],
    device const uint8_t* idx_l2     [[buffer(2)]],
    device const uint8_t* idx_l3     [[buffer(3)]],
    device const uint8_t* idx_l4     [[buffer(4)]],
    device const half* radii         [[buffer(5)]],
    device const float* cb1          [[buffer(6)]],
    device const float* cb2          [[buffer(7)]],
    device const float* cb3          [[buffer(8)]],
    device const float* cb4          [[buffer(9)]],
    device const float* values       [[buffer(10)]],
    device float* out                [[buffer(11)]],
    device const uint* params_buf    [[buffer(12)]],  // [N, D, NB] + scale as float
    uint head_idx [[thread_position_in_grid]]
) {
    uint N = params_buf[0];
    uint D = params_buf[1];
    uint NB = params_buf[2];
    float scale = as_type<float>(params_buf[3]);

    float q[256];
    for (uint d = 0; d < D; d++) q[d] = scale * queries[head_idx * D + d];

    float max_s = -1e30f;
    float sum_e = 0.0f;
    float acc[256];
    for (uint d = 0; d < D; d++) acc[d] = 0.0f;

    for (uint ki = 0; ki < N; ki++) {
        float score = 0.0f;
        for (uint blk = 0; blk < NB; blk++) {
            uint bi = (head_idx * N + ki) * NB + blk;
            float r[16];
            r[0] = float(radii[bi]);

            { float th=cb4[idx_l4[bi]]; float r0=r[0]; r[0]=r0*cos(th); r[1]=r0*sin(th); }
            { float t[4]; for(uint j=0;j<2;j++){float th=cb3[idx_l3[bi*2+j]]; t[2*j]=r[j]*cos(th); t[2*j+1]=r[j]*sin(th);} for(uint j=0;j<4;j++) r[j]=t[j]; }
            { float t[8]; for(uint j=0;j<4;j++){float th=cb2[idx_l2[bi*4+j]]; t[2*j]=r[j]*cos(th); t[2*j+1]=r[j]*sin(th);} for(uint j=0;j<8;j++) r[j]=t[j]; }
            { float t[16]; for(uint j=0;j<8;j++){float th=cb1[idx_l1[bi*8+j]]; t[2*j]=r[j]*cos(th); t[2*j+1]=r[j]*sin(th);} for(uint j=0;j<16;j++) r[j]=t[j]; }

            uint base = blk * 16;
            for (uint j = 0; j < 16; j++) score += q[base+j] * r[j];
        }

        float old_max = max_s;
        if (score > max_s) max_s = score;
        float factor = metal::exp(old_max - max_s);
        float exp_s = metal::exp(score - max_s);
        sum_e = sum_e * factor + exp_s;

        uint v_off = (head_idx * N + ki) * D;
        for (uint d = 0; d < D; d++) acc[d] = acc[d] * factor + exp_s * values[v_off + d];
    }

    float inv = 1.0f / (sum_e + 1e-8f);
    for (uint d = 0; d < D; d++) out[head_idx * D + d] = acc[d] * inv;
}

// SIMD-parallel compressed SDPA
// Adapted from MLX sdpa_vector with polar inverse key loading
//
// Architecture:
// - BN=24 simdgroups, each processes a different key token
// - BD=32 SIMD lanes, each handles D/32=4 elements of the key
// - Online softmax with running max + sum_exp across simdgroups
// - Threadgroup reduction for final output
//
// Key loading: instead of reading raw floats, each lane reconstructs
// its 4 elements from compressed polar indices. Since 4 elements fit
// within one 16-element block, each lane reconstructs its block independently.

kernel void sdpa_tq_parallel(
    device const float* queries      [[buffer(0)]],   // (BH, D)
    device const uint8_t* idx_l1     [[buffer(1)]],   // (BH*N*NB*8)
    device const uint8_t* idx_l2     [[buffer(2)]],   // (BH*N*NB*4)
    device const uint8_t* idx_l3     [[buffer(3)]],   // (BH*N*NB*2)
    device const uint8_t* idx_l4     [[buffer(4)]],   // (BH*N*NB)
    device const half* radii         [[buffer(5)]],   // (BH*N*NB)
    device const float* cb1          [[buffer(6)]],
    device const float* cb2          [[buffer(7)]],
    device const float* cb3          [[buffer(8)]],
    device const float* cb4          [[buffer(9)]],
    device const float* values       [[buffer(10)]],  // (BH*N, D)
    device float* out                [[buffer(11)]],  // (BH, D)
    device const uint* params_buf    [[buffer(12)]],  // [N, D, NB, scale_bits]
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    uint N = params_buf[0];
    uint D = params_buf[1];
    uint NB = params_buf[2];
    float scale = as_type<float>(params_buf[3]);

    constexpr uint BN = 24;  // simdgroups per threadgroup
    constexpr uint BD = 32;
    uint qk_pt = D / BD;  // elements per lane (4 for D=128)
    uint v_pt = D / BD;

    uint head_idx = tid.x;

    // Load query slice (pre-scaled)
    float q[8];
    for (uint i = 0; i < qk_pt; i++) {
        q[i] = scale * queries[head_idx * D + simd_lid * qk_pt + i];
    }

    float o[8];
    for (uint i = 0; i < v_pt; i++) o[i] = 0.0f;

    float max_score = -1e30f;
    float sum_exp = 0.0f;

    uint key_base = head_idx * N;

    // Main loop: each simdgroup handles a different key, stride by BN
    for (uint ki = simd_gid; ki < N; ki += BN) {
        // Reconstruct this lane's elements of key[ki]
        // Lane handles elements [simd_lid * qk_pt, simd_lid * qk_pt + qk_pt)
        // These all fall within block = (simd_lid * qk_pt) / 16
        float k_local[8];

        uint elem_start = simd_lid * qk_pt;
        uint blk = elem_start / 16;
        uint blk_off = elem_start % 16;

        // Reconstruct the full block for this lane's elements
        uint bi = (key_base + ki) * NB + blk;
        float r[16];
        r[0] = float(radii[bi]);

        { float th=cb4[idx_l4[bi]]; float r0=r[0]; r[0]=r0*cos(th); r[1]=r0*sin(th); }
        { float t[4]; for(uint j=0;j<2;j++){float th=cb3[idx_l3[bi*2+j]]; t[2*j]=r[j]*cos(th); t[2*j+1]=r[j]*sin(th);} for(uint j=0;j<4;j++) r[j]=t[j]; }
        { float t[8]; for(uint j=0;j<4;j++){float th=cb2[idx_l2[bi*4+j]]; t[2*j]=r[j]*cos(th); t[2*j+1]=r[j]*sin(th);} for(uint j=0;j<8;j++) r[j]=t[j]; }
        { float t[16]; for(uint j=0;j<8;j++){float th=cb1[idx_l1[bi*8+j]]; t[2*j]=r[j]*cos(th); t[2*j+1]=r[j]*sin(th);} for(uint j=0;j<16;j++) r[j]=t[j]; }

        for (uint i = 0; i < qk_pt; i++) {
            k_local[i] = r[blk_off + i];
        }

        // Score: q · k via SIMD reduction
        float score = 0.0f;
        for (uint i = 0; i < qk_pt; i++) score += q[i] * k_local[i];
        score = simd_sum(score);

        // Online softmax
        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_s = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_s;

        // Value accumulation
        uint v_off = (key_base + ki) * D + simd_lid * v_pt;
        for (uint i = 0; i < v_pt; i++) {
            o[i] = o[i] * factor + exp_s * values[v_off + i];
        }
    }

    // Threadgroup reduction (identical to MLX sdpa_vector)
    threadgroup float tg_max[BN];
    threadgroup float tg_sum[BN];
    threadgroup float tg_out[BN * BD];

    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sg_max = (simd_lid < BN) ? tg_max[simd_lid] : -1e30f;
    float global_max = simd_max(sg_max);
    float sg_factor = fast::exp(sg_max - global_max);
    float sg_sum = (simd_lid < BN) ? tg_sum[simd_lid] : 0.0f;
    float global_sum = simd_sum(sg_sum * sg_factor);

    for (uint i = 0; i < v_pt; i++) {
        tg_out[simd_lid * BN + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Read with guard: only BN simdgroups have valid data
        float val = (simd_lid < BN) ?
            tg_out[simd_gid * BN + simd_lid] * fast::exp(tg_max[simd_lid] - global_max) : 0.0f;
        o[i] = simd_sum(val);
        o[i] = global_sum == 0 ? o[i] : (o[i] / global_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // After reduction, simdgroup 0 has the final aggregated output
    // Each lane in simdgroup 0 writes its v_pt elements
    if (simd_gid == 0) {
        for (uint i = 0; i < v_pt; i++) {
            out[head_idx * D + simd_lid * v_pt + i] = o[i];
        }
    }
}

// Compressed scoring kernel: outputs raw scores, not full attention
// This kernel ONLY computes q·k for all keys from compressed data
// Softmax + value matmul done separately by MLX
kernel void tq_score_parallel(
    device const float* queries      [[buffer(0)]],   // (BH, D) preconditioned
    device const uint8_t* idx_l1     [[buffer(1)]],
    device const uint8_t* idx_l2     [[buffer(2)]],
    device const uint8_t* idx_l3     [[buffer(3)]],
    device const uint8_t* idx_l4     [[buffer(4)]],
    device const half* radii         [[buffer(5)]],
    device const float* cb1          [[buffer(6)]],
    device const float* cb2          [[buffer(7)]],
    device const float* cb3          [[buffer(8)]],
    device const float* cb4          [[buffer(9)]],
    device float* scores             [[buffer(10)]],  // (BH, N) output scores
    device const uint* params_buf    [[buffer(11)]],  // [N, D, NB, scale_bits]
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    uint N = params_buf[0];
    uint D = params_buf[1];
    uint NB = params_buf[2];
    float scale = as_type<float>(params_buf[3]);

    constexpr uint BN = 24;
    uint qk_pt = D / 32;
    uint head_idx = tid.x;
    uint key_base = head_idx * N;

    // Load query slice
    float q[8];
    for (uint i = 0; i < qk_pt; i++)
        q[i] = scale * queries[head_idx * D + simd_lid * qk_pt + i];

    // Each simdgroup processes different keys
    for (uint ki = simd_gid; ki < N; ki += BN) {
        uint elem_start = simd_lid * qk_pt;
        uint blk = elem_start / 16;
        uint blk_off = elem_start % 16;
        uint bi = (key_base + ki) * NB + blk;

        float r[16];
        r[0] = float(radii[bi]);
        { float th=cb4[idx_l4[bi]]; float r0=r[0]; r[0]=r0*cos(th); r[1]=r0*sin(th); }
        { float t[4]; for(uint j=0;j<2;j++){float th=cb3[idx_l3[bi*2+j]]; t[2*j]=r[j]*cos(th); t[2*j+1]=r[j]*sin(th);} for(uint j=0;j<4;j++) r[j]=t[j]; }
        { float t[8]; for(uint j=0;j<4;j++){float th=cb2[idx_l2[bi*4+j]]; t[2*j]=r[j]*cos(th); t[2*j+1]=r[j]*sin(th);} for(uint j=0;j<8;j++) r[j]=t[j]; }
        { float t[16]; for(uint j=0;j<8;j++){float th=cb1[idx_l1[bi*8+j]]; t[2*j]=r[j]*cos(th); t[2*j+1]=r[j]*sin(th);} for(uint j=0;j<16;j++) r[j]=t[j]; }

        float score = 0.0f;
        for (uint i = 0; i < qk_pt; i++) score += q[i] * r[blk_off + i];
        score = simd_sum(score);

        if (simd_lid == 0) {
            scores[head_idx * N + ki] = score;
        }
    }
}
