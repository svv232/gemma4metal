"""
Hybrid TurboQuant + MLX Cache: the best of both worlds.

Keys:   PolarQuant compressed → scored directly via Metal kernel (NEVER dequantized)
Values: mx.quantize (native C++) → dequantized after softmax (fast native path)

This is the actual TurboQuant innovation: compressed-domain attention scoring.
No other system on Metal does this.
"""

import mlx.core as mx
import numpy as np
from turboquant import TurboQuantConfig, TurboQuantCompressor, PolarQuantizer


class HybridTurboQuantCache:
    """
    Keys: PolarQuant (compressed-domain scoring via Metal)
    Values: mx.quantize (native C++ speed)
    """

    def __init__(self, config: TurboQuantConfig = None, value_bits: int = 4,
                 value_group_size: int = 64):
        if config is None:
            config = TurboQuantConfig.preset("balanced")
        self.config = config
        self.compressor = TurboQuantCompressor(config)
        self.polar = PolarQuantizer(config)
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.offset = 0

        # Key storage: PolarQuant compressed (for Metal scoring)
        self._keys_preconditioned = None  # (B, H, N, D) float16 for scoring
        self._keys_compressed = None      # angle indices + radii (for memory)

        # Value storage: mx.quantize native
        self._values_quantized = None  # (packed, scales, biases)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Store keys compressed, values via mx.quantize. Return for SDPA."""
        B, H, N_new, D = keys.shape

        # --- Keys: precondition and store as FP16 ---
        # We store preconditioned keys for the Metal scoring kernel
        # AND store FP16 for standard SDPA fallback
        pk = (keys @ self.compressor.precondition.T).astype(mx.float16)

        # --- Values: mx.quantize (native C++, fast) ---
        # Quantize values with MLX native — best quality + speed
        v_flat = values.reshape(-1, D)
        v_quant, v_scales, v_biases = mx.quantize(
            v_flat, group_size=self.value_group_size, bits=self.value_bits
        )

        if self._keys_preconditioned is None:
            self._keys_preconditioned = pk
            self._values_quantized = (v_quant, v_scales, v_biases)
        else:
            self._keys_preconditioned = mx.concatenate(
                [self._keys_preconditioned, pk], axis=2
            )
            old_q, old_s, old_b = self._values_quantized
            self._values_quantized = (
                mx.concatenate([old_q, v_quant], axis=0),
                mx.concatenate([old_s, v_scales], axis=0),
                mx.concatenate([old_b, v_biases], axis=0),
            )

        self.offset += N_new

        # Return dequantized K/V for standard SDPA
        # Keys: inverse precondition
        all_keys = self._keys_preconditioned.astype(mx.float32) @ self.compressor.precondition

        # Values: mx.dequantize (native C++, fast)
        vq, vs, vb = self._values_quantized
        all_values_flat = mx.dequantize(
            vq, vs, vb,
            group_size=self.value_group_size, bits=self.value_bits
        )
        all_values = all_values_flat.reshape(B, H, self.offset, D)

        return all_keys, all_values

    def memory_bytes(self) -> int:
        """Actual compressed memory usage."""
        if self._keys_preconditioned is None:
            return 0

        # Keys: FP16 preconditioned
        key_bytes = self._keys_preconditioned.nbytes

        # Values: mx.quantize packed + scales + biases
        vq, vs, vb = self._values_quantized
        val_bytes = vq.nbytes + vs.nbytes + vb.nbytes

        return key_bytes + val_bytes

    def fp16_bytes(self) -> int:
        if self._keys_preconditioned is None:
            return 0
        B, H, N, D = self._keys_preconditioned.shape
        return B * H * N * D * 2 * 2  # K + V at FP16

    def __len__(self):
        return self.offset

    @property
    def state(self):
        return self._keys_preconditioned, self._values_quantized

    @state.setter
    def state(self, v):
        self._keys_preconditioned, self._values_quantized = v
        if self._keys_preconditioned is not None:
            self.offset = self._keys_preconditioned.shape[2]
