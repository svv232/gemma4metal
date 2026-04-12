#pragma once

#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace turboquant {

using namespace mlx::core;

class PolarQuantize : public Primitive {
 public:
  PolarQuantize(Stream stream, int block_size)
      : Primitive(stream), block_size_(block_size) {}

  void eval_cpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override;
  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override;

  const char* name() const override { return "PolarQuantize"; }

  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const PolarQuantize&>(other);
    return block_size_ == o.block_size_;
  }

 private:
  int block_size_;
};

class PolarDequantize : public Primitive {
 public:
  PolarDequantize(Stream stream, int block_size)
      : Primitive(stream), block_size_(block_size) {}

  void eval_cpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override;
  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override;

  const char* name() const override { return "PolarDequantize"; }

  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const PolarDequantize&>(other);
    return block_size_ == o.block_size_;
  }

 private:
  int block_size_;
};

std::vector<array> polar_quantize(
    const array& vectors,
    const array& precondition,
    const array& cb_l1,
    const array& cb_l2,
    const array& cb_l3,
    const array& cb_l4,
    int block_size = 16,
    StreamOrDevice s = {});

array polar_dequantize(
    const array& indices_l1,
    const array& indices_l2,
    const array& indices_l3,
    const array& indices_l4,
    const array& radii,
    const array& precondition,
    const array& cb_l1,
    const array& cb_l2,
    const array& cb_l3,
    const array& cb_l4,
    int block_size = 16,
    StreamOrDevice s = {});

// Packed quantize: outputs bit-packed indices (3.56x compression)
std::vector<array> polar_quantize_packed(
    const array& vectors,
    const array& precondition,
    const array& cb_l1, const array& cb_l2, const array& cb_l3, const array& cb_l4,
    int block_size = 16,
    StreamOrDevice s = {});

// Packed dequantize: reads bit-packed indices
array polar_dequantize_packed(
    const array& packed_l1,  // uint32
    const array& packed_l2,  // uint8
    const array& packed_l3,  // uint8
    const array& packed_l4,  // uint8
    const array& radii,
    const array& cb_l1, const array& cb_l2, const array& cb_l3, const array& cb_l4,
    int D, int block_size = 16,
    StreamOrDevice s = {});

// Fast packed dequantize with precomputed cos/sin tables (ZERO trig ops)
array polar_dequantize_packed_fast(
    const array& packed_l1, const array& packed_l2,
    const array& packed_l3, const array& packed_l4,
    const array& radii,
    const array& cs_l1, const array& cs_l2,  // float2 arrays: {cos, sin} per codebook entry
    const array& cs_l3, const array& cs_l4,
    int D, int block_size = 16,
    StreamOrDevice s = {});

// Compressed-domain SDPA: score directly from compressed keys
class CompressedSDPA : public Primitive {
 public:
  CompressedSDPA(Stream stream, int block_size, float scale)
      : Primitive(stream), block_size_(block_size), scale_(scale) {}

  void eval_cpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override;
  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override;

  const char* name() const override { return "CompressedSDPA"; }

  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const CompressedSDPA&>(other);
    return block_size_ == o.block_size_ && scale_ == o.scale_;
  }

 private:
  int block_size_;
  float scale_;
};

// Compressed attention: scores from compressed keys + softmax + value weighted sum
array compressed_sdpa(
    const array& queries,       // (Nq, D) preconditioned queries (Pq)
    const array& indices_l1,    // compressed key indices
    const array& indices_l2,
    const array& indices_l3,
    const array& indices_l4,
    const array& radii,
    const array& cb_l1,
    const array& cb_l2,
    const array& cb_l3,
    const array& cb_l4,
    const array& values,        // (N, D)
    int block_size = 16,
    float scale = 0.0884f,      // 1/sqrt(128)
    StreamOrDevice s = {});

// Fast dequantize: inverse polar ONLY (no precondition matmul)
// Use with preconditioned queries: <Pq, y_recon> = <q, P^T y_recon>
array polar_dequantize_fast(
    const array& indices_l1, const array& indices_l2,
    const array& indices_l3, const array& indices_l4,
    const array& radii,
    const array& cb_l1, const array& cb_l2,
    const array& cb_l3, const array& cb_l4,
    int D, int block_size = 16,
    StreamOrDevice s = {});

// Compressed scoring only: outputs (Nq, N) scores from compressed keys
array compressed_score(
    const array& queries,
    const array& indices_l1, const array& indices_l2,
    const array& indices_l3, const array& indices_l4,
    const array& radii,
    const array& cb_l1, const array& cb_l2,
    const array& cb_l3, const array& cb_l4,
    int N, int block_size = 16, float scale = 0.0884f,
    StreamOrDevice s = {});

// Fused int4 SDPA: scores from int4 quantized K + online softmax + V weighted sum
// All in one Metal kernel dispatch — no intermediate matrices.
// Beats dequantize+SDPA at all context lengths by avoiding K materialization.
class FusedInt4SDPA : public Primitive {
 public:
  FusedInt4SDPA(Stream stream, int head_dim, int gqa_factor, float scale, int sliding_window)
      : Primitive(stream), head_dim_(head_dim), gqa_factor_(gqa_factor),
        scale_(scale), sliding_window_(sliding_window) {}

  void eval_cpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override;
  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override;

  const char* name() const override { return "FusedInt4SDPA"; }

  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const FusedInt4SDPA&>(other);
    return head_dim_ == o.head_dim_ && gqa_factor_ == o.gqa_factor_ &&
           scale_ == o.scale_ && sliding_window_ == o.sliding_window_;
  }

 private:
  int head_dim_;
  int gqa_factor_;
  float scale_;
  int sliding_window_;
};

// Dispatch fused int4 attention
// q: (num_heads, head_dim) float32
// k_quant: (num_kv_heads, N, head_dim/8) uint32
// k_scales, k_biases: (num_kv_heads, N, head_dim/64) float32
// v_quant, v_scales, v_biases: same shapes as K
// Returns: (num_heads, head_dim) float32
array sdpa_int4(
    const array& queries,
    const array& k_quant, const array& k_scales, const array& k_biases,
    const array& v_quant, const array& v_scales, const array& v_biases,
    int gqa_factor, float scale = 1.0f, int sliding_window = 0,
    StreamOrDevice s = {});

void set_metallib_path(const std::string& path);

}  // namespace turboquant
