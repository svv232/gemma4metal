// TurboQuant Python bindings via nanobind (MLX extension framework)
#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/string.h>
#include "turboquant.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

NB_MODULE(turboquant_ext, m) {
    m.doc() = "TurboQuant: Fused int4 SDPA Metal kernel for MLX";

    m.def("sdpa_int4", [](const array& q, const array& kq, const array& ks, const array& kb,
                           const array& vq, const array& vs, const array& vb,
                           int gqa, float scale, int sw) {
            return turboquant::sdpa_int4(q, kq, ks, kb, vq, vs, vb, gqa, scale, sw);
        },
        "queries"_a, "k_quant"_a, "k_scales"_a, "k_biases"_a,
        "v_quant"_a, "v_scales"_a, "v_biases"_a,
        "gqa_factor"_a, "scale"_a = 1.0f, "sliding_window"_a = 0,
        R"(Fused int4 SDPA: attention from compressed K/V without dequantizing.

        35% faster than dequantize + scaled_dot_product_attention at 786 tokens.
        Saves ~640MB peak memory by never materializing dequantized K matrix.

        Args:
            queries: (num_heads, head_dim) float32
            k_quant: (num_kv_heads, N, head_dim/8) uint32
            k_scales: (num_kv_heads, N, head_dim/64) float32
            k_biases: (num_kv_heads, N, head_dim/64) float32
            v_quant, v_scales, v_biases: same layout as K
            gqa_factor: num_heads / num_kv_heads
            scale: attention scale (default 1.0)
            sliding_window: 0 for no window restriction

        Returns:
            array: (num_heads, head_dim) attention output
        )");

    m.def("set_metallib_path", &turboquant::set_metallib_path,
        "path"_a, "Set path to compiled turboquant.metallib");
}
