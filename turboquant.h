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

void set_metallib_path(const std::string& path);

}  // namespace turboquant
