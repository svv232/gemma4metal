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

// Optimized parallel scoring with shared memory block reconstruction
// One lane per block reconstructs into threadgroup memory
// All lanes read their elements from shared memory
// 4x fewer trig ops than tq_score_parallel
kernel void tq_score_shared(
    device const float* queries      [[buffer(0)]],
    device const uint8_t* idx_l1     [[buffer(1)]],
    device const uint8_t* idx_l2     [[buffer(2)]],
    device const uint8_t* idx_l3     [[buffer(3)]],
    device const uint8_t* idx_l4     [[buffer(4)]],
    device const half* radii         [[buffer(5)]],
    device const float* cb1          [[buffer(6)]],
    device const float* cb2          [[buffer(7)]],
    device const float* cb3          [[buffer(8)]],
    device const float* cb4          [[buffer(9)]],
    device float* scores             [[buffer(10)]],
    device const uint* params_buf    [[buffer(11)]],
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

    // Shared memory: one reconstructed key per simdgroup (D floats each)
    // BN simdgroups × D floats = 24 × 128 = 3072 floats = 12KB (fits in 32KB)
    threadgroup float shared_keys[24 * 128];

    float q[8];
    for (uint i = 0; i < qk_pt; i++)
        q[i] = scale * queries[head_idx * D + simd_lid * qk_pt + i];

    for (uint ki = simd_gid; ki < N; ki += BN) {
        // Phase 1: Reconstruct key into shared memory
        // Each lane reconstructs one block (lanes 0-7 for NB=8 blocks)
        // Lanes 8-31 idle during reconstruction
        if (simd_lid < NB) {
            uint blk = simd_lid;
            uint bi = (key_base + ki) * NB + blk;
            float r[16];
            r[0] = float(radii[bi]);

            { float th=cb4[idx_l4[bi]]; float r0=r[0]; r[0]=r0*cos(th); r[1]=r0*sin(th); }
            { float t[4]; for(uint j=0;j<2;j++){float th=cb3[idx_l3[bi*2+j]]; t[2*j]=r[j]*cos(th); t[2*j+1]=r[j]*sin(th);} for(uint j=0;j<4;j++) r[j]=t[j]; }
            { float t[8]; for(uint j=0;j<4;j++){float th=cb2[idx_l2[bi*4+j]]; t[2*j]=r[j]*cos(th); t[2*j+1]=r[j]*sin(th);} for(uint j=0;j<8;j++) r[j]=t[j]; }
            { float t[16]; for(uint j=0;j<8;j++){float th=cb1[idx_l1[bi*8+j]]; t[2*j]=r[j]*cos(th); t[2*j+1]=r[j]*sin(th);} for(uint j=0;j<16;j++) r[j]=t[j]; }

            // Write to shared memory
            for (uint j = 0; j < 16; j++)
                shared_keys[simd_gid * D + blk * 16 + j] = r[j];
        }

        // Barrier: wait for all blocks to be reconstructed
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: All lanes read their elements from shared memory and compute score
        float score = 0.0f;
        for (uint i = 0; i < qk_pt; i++) {
            score += q[i] * shared_keys[simd_gid * D + simd_lid * qk_pt + i];
        }
        score = simd_sum(score);

        if (simd_lid == 0) {
            scores[head_idx * N + ki] = score;
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Bit-packed quantize: outputs packed indices instead of uint8
// Level 1: 8 × 4-bit → 1 uint32 per block
// Level 2: 4 × 2-bit → 1 uint8 per block  
// Level 3: 2 × 2-bit → 1 uint8 per block (4 bits used)
// Level 4: 1 × 2-bit → 1 uint8 per block (2 bits used)
kernel void polar_quantize_packed(
    device const float* input        [[buffer(0)]],
    device const float* cb_l1        [[buffer(1)]],
    device const float* cb_l2        [[buffer(2)]],
    device const float* cb_l3        [[buffer(3)]],
    device const float* cb_l4        [[buffer(4)]],
    device const uint& n_cb1         [[buffer(5)]],
    device const uint& n_cb2         [[buffer(6)]],
    device const uint& n_cb3         [[buffer(7)]],
    device const uint& n_cb4         [[buffer(8)]],
    device uint* packed_l1           [[buffer(9)]],  // 1 uint32 per block (8×4bit)
    device uint8_t* packed_l2        [[buffer(10)]],  // 1 uint8 per block (4×2bit)
    device uint8_t* packed_l3        [[buffer(11)]],  // 1 uint8 per block (2×2bit)
    device uint8_t* packed_l4        [[buffer(12)]],  // 1 uint8 per block (1×2bit)
    device half* radii               [[buffer(13)]],
    uint block_idx                   [[thread_position_in_grid]]
) {
    float r[16];
    for (uint j = 0; j < 16; j++) r[j] = input[block_idx * 16 + j];

    // Level 1: pack 8 × 4-bit into one uint32
    uint packed1 = 0;
    {
        float tmp[8];
        for (uint j = 0; j < 8; j++) {
            float a = r[2*j], b = r[2*j+1];
            float angle = metal::atan2(b, a);
            if (angle < 0.0f) angle += 2.0f * M_PI_F;
            uint8_t idx = 0;
            float best = 1e10f;
            for (uint c = 0; c < n_cb1; c++) {
                float d = metal::abs(angle - cb_l1[c]);
                if (d < best) { best = d; idx = c; }
            }
            packed1 |= (uint(idx) & 0xF) << (j * 4);
            tmp[j] = metal::sqrt(a*a + b*b);
        }
        for (uint j = 0; j < 8; j++) r[j] = tmp[j];
    }
    packed_l1[block_idx] = packed1;

    // Level 2: pack 4 × 2-bit into one uint8
    uint8_t packed2 = 0;
    {
        float tmp[4];
        for (uint j = 0; j < 4; j++) {
            float a = r[2*j], b = r[2*j+1];
            float angle = metal::atan2(b, a);
            uint8_t idx = 0;
            float best = 1e10f;
            for (uint c = 0; c < n_cb2; c++) {
                float d = metal::abs(angle - cb_l2[c]);
                if (d < best) { best = d; idx = c; }
            }
            packed2 |= (idx & 0x3) << (j * 2);
            tmp[j] = metal::sqrt(a*a + b*b);
        }
        for (uint j = 0; j < 4; j++) r[j] = tmp[j];
    }
    packed_l2[block_idx] = packed2;

    // Level 3: pack 2 × 2-bit
    uint8_t packed3 = 0;
    {
        float tmp[2];
        for (uint j = 0; j < 2; j++) {
            float a = r[2*j], b = r[2*j+1];
            float angle = metal::atan2(b, a);
            uint8_t idx = 0;
            float best = 1e10f;
            for (uint c = 0; c < n_cb3; c++) {
                float d = metal::abs(angle - cb_l3[c]);
                if (d < best) { best = d; idx = c; }
            }
            packed3 |= (idx & 0x3) << (j * 2);
            tmp[j] = metal::sqrt(a*a + b*b);
        }
        r[0] = tmp[0]; r[1] = tmp[1];
    }
    packed_l3[block_idx] = packed3;

    // Level 4: 1 × 2-bit
    {
        float a = r[0], b = r[1];
        float angle = metal::atan2(b, a);
        uint8_t idx = 0;
        float best = 1e10f;
        for (uint c = 0; c < n_cb4; c++) {
            float d = metal::abs(angle - cb_l4[c]);
            if (d < best) { best = d; idx = c; }
        }
        packed_l4[block_idx] = idx;
        radii[block_idx] = static_cast<half>(metal::sqrt(a*a + b*b));
    }
}

// Bit-packed inverse polar: reads packed indices, reconstructs vector
kernel void polar_inverse_packed(
    device const float* cb1          [[buffer(0)]],
    device const float* cb2          [[buffer(1)]],
    device const float* cb3          [[buffer(2)]],
    device const float* cb4          [[buffer(3)]],
    device const uint* packed_l1     [[buffer(4)]],  // 1 uint32 per block
    device const uint8_t* packed_l2  [[buffer(5)]],  // 1 uint8 per block
    device const uint8_t* packed_l3  [[buffer(6)]],  // 1 uint8 per block
    device const uint8_t* packed_l4  [[buffer(7)]],  // 1 uint8 per block
    device const half* radii         [[buffer(8)]],
    device float* output             [[buffer(9)]],
    uint block_idx                   [[thread_position_in_grid]]
) {
    float r[16];
    r[0] = float(radii[block_idx]);

    // Unpack L4: 1 × 2-bit
    { uint idx = packed_l4[block_idx] & 0x3;
      float th = cb4[idx]; float r0 = r[0];
      r[0] = r0*cos(th); r[1] = r0*sin(th); }

    // Unpack L3: 2 × 2-bit from uint8
    { float t[4]; uint8_t p = packed_l3[block_idx];
      for (uint j=0;j<2;j++) {
          uint idx = (p >> (j*2)) & 0x3;
          float th = cb3[idx];
          t[2*j]=r[j]*cos(th); t[2*j+1]=r[j]*sin(th);
      }
      for (uint j=0;j<4;j++) r[j]=t[j]; }

    // Unpack L2: 4 × 2-bit from uint8
    { float t[8]; uint8_t p = packed_l2[block_idx];
      for (uint j=0;j<4;j++) {
          uint idx = (p >> (j*2)) & 0x3;
          float th = cb2[idx];
          t[2*j]=r[j]*cos(th); t[2*j+1]=r[j]*sin(th);
      }
      for (uint j=0;j<8;j++) r[j]=t[j]; }

    // Unpack L1: 8 × 4-bit from uint32
    { float t[16]; uint p = packed_l1[block_idx];
      for (uint j=0;j<8;j++) {
          uint idx = (p >> (j*4)) & 0xF;
          float th = cb1[idx];
          t[2*j]=r[j]*cos(th); t[2*j+1]=r[j]*sin(th);
      }
      for (uint j=0;j<16;j++) r[j]=t[j]; }

    for (uint j = 0; j < 16; j++)
        output[block_idx * 16 + j] = r[j];
}

// Zero-trig inverse polar: uses precomputed cos/sin tables instead of runtime trig
// Input: codebook cos/sin pairs (precomputed on CPU)
kernel void polar_inverse_packed_fast(
    device const float2* cs_l1       [[buffer(0)]],  // {cos, sin} pairs for L1 (16 entries)
    device const float2* cs_l2       [[buffer(1)]],  // {cos, sin} for L2 (4 entries)
    device const float2* cs_l3       [[buffer(2)]],  // {cos, sin} for L3
    device const float2* cs_l4       [[buffer(3)]],  // {cos, sin} for L4
    device const uint* packed_l1     [[buffer(4)]],
    device const uint8_t* packed_l2  [[buffer(5)]],
    device const uint8_t* packed_l3  [[buffer(6)]],
    device const uint8_t* packed_l4  [[buffer(7)]],
    device const half* radii         [[buffer(8)]],
    device float* output             [[buffer(9)]],
    uint block_idx                   [[thread_position_in_grid]]
) {
    float r[16];
    r[0] = float(radii[block_idx]);

    // L4: table lookup (zero trig!)
    { uint idx = packed_l4[block_idx] & 0x3;
      float2 cs = cs_l4[idx];
      float r0 = r[0]; r[0] = r0 * cs.x; r[1] = r0 * cs.y; }

    // L3
    { float t[4]; uint8_t p = packed_l3[block_idx];
      for (uint j=0;j<2;j++) {
          float2 cs = cs_l3[(p >> (j*2)) & 0x3];
          t[2*j]=r[j]*cs.x; t[2*j+1]=r[j]*cs.y;
      }
      for (uint j=0;j<4;j++) r[j]=t[j]; }

    // L2
    { float t[8]; uint8_t p = packed_l2[block_idx];
      for (uint j=0;j<4;j++) {
          float2 cs = cs_l2[(p >> (j*2)) & 0x3];
          t[2*j]=r[j]*cs.x; t[2*j+1]=r[j]*cs.y;
      }
      for (uint j=0;j<8;j++) r[j]=t[j]; }

    // L1
    { float t[16]; uint p = packed_l1[block_idx];
      for (uint j=0;j<8;j++) {
          float2 cs = cs_l1[(p >> (j*4)) & 0xF];
          t[2*j]=r[j]*cs.x; t[2*j+1]=r[j]*cs.y;
      }
      for (uint j=0;j<16;j++) r[j]=t[j]; }

    for (uint j = 0; j < 16; j++)
        output[block_idx * 16 + j] = r[j];
}

// ============================================================================
// Fused int4 SDPA: Q @ dequantize(K_int4)^T with online softmax + V accumulation
// All in one kernel dispatch — no intermediate score matrix materialized.
//
// Based on MLX sdpa_vector architecture:
// - BN simdgroups, each processes a different key token (stride BN)
// - BD=32 SIMD lanes, each handles D/BD elements
// - Online softmax with running max + sum_exp across simdgroups
//
// Key innovation: dequantize int4 K in registers, never materializing full K.
// At long contexts this saves ~50% memory bandwidth vs dequantize-then-SDPA.
//
// int4 format (MLX native): uint32 packs 8 x 4-bit values
//   dequantize: value = scale * ((packed >> (4*i)) & 0xF) + bias
//   group_size=64: 64 elements share one scale+bias pair
// ============================================================================

template <int D, int BN = 32>
[[kernel]] void sdpa_int4_vector(
    const device float* queries        [[buffer(0)]],
    const device uint32_t* k_quant     [[buffer(1)]],
    const device float* k_scales       [[buffer(2)]],
    const device float* k_biases       [[buffer(3)]],
    const device uint32_t* v_quant     [[buffer(4)]],
    const device float* v_scales       [[buffer(5)]],
    const device float* v_biases       [[buffer(6)]],
    device float* out                  [[buffer(7)]],
    const constant int& gqa_factor     [[buffer(8)]],
    const constant int& N              [[buffer(9)]],
    const constant float& scale        [[buffer(10)]],
    const constant int& sliding_window [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr int BD = 32;
    constexpr int elems_per_lane = D / BD;

    const int head_idx = tid.x;
    const int kv_head_idx = head_idx / gqa_factor;

    const int k_packed_dim = D / 8;
    const int k_scale_dim = D / 64;

    const device uint32_t* k_q_base = k_quant + kv_head_idx * N * k_packed_dim;
    const device float* k_s_base = k_scales + kv_head_idx * N * k_scale_dim;
    const device float* k_b_base = k_biases + kv_head_idx * N * k_scale_dim;
    const device uint32_t* v_q_base = v_quant + kv_head_idx * N * k_packed_dim;
    const device float* v_s_base = v_scales + kv_head_idx * N * k_scale_dim;
    const device float* v_b_base = v_biases + kv_head_idx * N * k_scale_dim;

    int lane_start = simd_lid * elems_per_lane;

    // Pre-scaled query values for vectorized 4-bit dot product (MLX qdot pattern)
    thread float q_scaled[D / BD];
    for (int i = 0; i < elems_per_lane; i += 4) {
        float q0 = scale * queries[head_idx * D + lane_start + i];
        float q1 = scale * queries[head_idx * D + lane_start + i + 1];
        float q2 = scale * queries[head_idx * D + lane_start + i + 2];
        float q3 = scale * queries[head_idx * D + lane_start + i + 3];
        q_scaled[i]     = q0;
        q_scaled[i + 1] = q1 / 16.0f;
        q_scaled[i + 2] = q2 / 256.0f;
        q_scaled[i + 3] = q3 / 4096.0f;
    }

    thread float o[D / BD];
    for (int i = 0; i < elems_per_lane; i++) o[i] = 0.0f;

    float max_score = -1e30f;
    float sum_exp = 0.0f;

    for (int ki = simd_gid; ki < N; ki += BN) {
        if (sliding_window > 0 && (N - 1 - ki) >= sliding_window) continue;

        // Vectorized K scoring (MLX qdot pattern for 4-bit)
        // Read K as uint16 (4 nibbles each), use pre-scaled query to avoid per-nibble shifts
        // score = sum_groups(scale * dot(q_prescaled, k_packed_nibbles) + bias * sum(q_raw))
        float score = 0.0f;
        // k_q_base is uint32* with k_packed_dim uint32s per token
        // Reinterpret as uint16: 2x as many entries, each holding 4 nibbles
        const device uint16_t* kw = (const device uint16_t*)(k_q_base + ki * k_packed_dim)
                                    + lane_start / 4;
        for (int i = 0; i < elems_per_lane; i += 4) {
            int group_idx = (lane_start + i) / 64;
            float s = k_s_base[ki * k_scale_dim + group_idx];
            float b = k_b_base[ki * k_scale_dim + group_idx];
            uint16_t w = kw[i / 4];
            // Packed dot: q_prescaled already divided by {1, 16, 256, 4096}
            // so q_prescaled[j] * (w & mask_j) = q_raw[j] * nibble_j
            float accum = q_scaled[i]     * float(w & 0x000Fu)
                        + q_scaled[i + 1] * float(w & 0x00F0u)
                        + q_scaled[i + 2] * float(w & 0x0F00u)
                        + q_scaled[i + 3] * float(w & 0xF000u);
            // Bias: b * sum(q_raw) where q_raw = q_scaled * {1, 16, 256, 4096}
            float qsum = q_scaled[i] + q_scaled[i+1]*16.0f
                       + q_scaled[i+2]*256.0f + q_scaled[i+3]*4096.0f;
            score += s * accum + b * qsum;
        }
        score = simd_sum(score);

        // Online softmax
        float new_max = max(max_score, score);
        float factor = metal::fast::exp(max_score - new_max);
        float exp_score = metal::fast::exp(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;

        // Vectorized V dequantize + weighted accumulation
        const device uint16_t* vw = (const device uint16_t*)(v_q_base + ki * k_packed_dim)
                                    + lane_start / 4;
        for (int i = 0; i < elems_per_lane; i += 4) {
            int group_idx = (lane_start + i) / 64;
            float vs = v_s_base[ki * k_scale_dim + group_idx];
            float vb = v_b_base[ki * k_scale_dim + group_idx];
            uint16_t w = vw[i / 4];
            // Extract 4 values: vs * nibble + vb
            float v0 = vs * float(w & 0x000Fu) + vb;
            float v1 = vs * float((w >> 4) & 0xFu) + vb;
            float v2 = vs * float((w >> 8) & 0xFu) + vb;
            float v3 = vs * float((w >> 12) & 0xFu) + vb;
            o[i]     = o[i]     * factor + exp_score * v0;
            o[i + 1] = o[i + 1] * factor + exp_score * v1;
            o[i + 2] = o[i + 2] * factor + exp_score * v2;
            o[i + 3] = o[i + 3] * factor + exp_score * v3;
        }
    }

    // Reduction: combine partial results across simdgroups
    // Use threadgroup memory: each simdgroup writes its rescaled partials,
    // then simdgroup 0 reads and sums.

    // Step 1: compute global max and sum across simdgroups
    threadgroup float tg_max_arr[BN];
    threadgroup float tg_sum_arr[BN];
    if (simd_lid == 0) {
        tg_max_arr[simd_gid] = max_score;
        tg_sum_arr[simd_gid] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sg_max = (simd_lid < uint(BN)) ? tg_max_arr[simd_lid] : -1e30f;
    float new_max = simd_max(sg_max);
    float sg_factor = metal::fast::exp(sg_max - new_max);
    float sg_sum_val = (simd_lid < uint(BN)) ? tg_sum_arr[simd_lid] : 0.0f;
    float global_sum = simd_sum(sg_sum_val * sg_factor);

    // Step 2: each simdgroup rescales its partial output
    float my_factor = metal::fast::exp(max_score - new_max);
    for (int i = 0; i < elems_per_lane; i++) {
        o[i] *= my_factor;
    }

    // Step 3: reduce across simdgroups using BN passes through threadgroup memory
    // Each pass: one simdgroup writes its D values, others accumulate
    // Use a D-sized buffer (not BN*D which exceeds 32KB for D=512)
    threadgroup float tg_accum[D];

    // Initialize accumulator with simdgroup 0's values
    if (simd_gid == 0) {
        for (int i = 0; i < elems_per_lane; i++) {
            tg_accum[simd_lid * elems_per_lane + i] = o[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Add remaining simdgroups one at a time
    for (int sg = 1; sg < BN; sg++) {
        if (int(simd_gid) == sg) {
            for (int i = 0; i < elems_per_lane; i++) {
                tg_accum[simd_lid * elems_per_lane + i] += o[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 4: write output — simdgroup 0 reads from accumulator and normalizes
    if (simd_gid == 0) {
        for (int i = 0; i < elems_per_lane; i++) {
            int elem = simd_lid * elems_per_lane + i;
            float val = tg_accum[elem];
            out[head_idx * D + elem] = global_sum == 0 ? 0.0f : (val / global_sum);
        }
    }
}

template [[host_name("sdpa_int4_256")]] [[kernel]]
void sdpa_int4_vector<256, 32>(
    const device float*, const device uint32_t*, const device float*, const device float*,
    const device uint32_t*, const device float*, const device float*,
    device float*, const constant int&, const constant int&, const constant float&,
    const constant int&, uint3, uint, uint);

template [[host_name("sdpa_int4_512")]] [[kernel]]
void sdpa_int4_vector<512, 16>(
    const device float*, const device uint32_t*, const device float*, const device float*,
    const device uint32_t*, const device float*, const device float*,
    device float*, const constant int&, const constant int&, const constant float&,
    const constant int&, uint3, uint, uint);
