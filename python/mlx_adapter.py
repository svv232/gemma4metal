"""
MLX-LM KV cache adapter for TurboQuant.

Drop-in replacement for mlx_lm's KVCache that uses TurboQuant
compression internally. Compatible with any MLX-LM model.

Usage:
    from mlx_adapter import TurboQuantCache, make_turboquant_cache
    # Replace: cache = [KVCache() for _ in range(num_layers)]
    # With:    cache = make_turboquant_cache(num_layers, head_dim, n_kv_heads)
"""

import mlx.core as mx
import numpy as np
from turboquant import TurboQuantConfig, TurboQuantCompressor, PolarQuantizer


class TurboQuantCache:
    """
    TurboQuant-compressed KV cache compatible with mlx-lm's cache interface.

    Stores keys and values in TurboQuant compressed format.
    Returns dequantized keys/values for standard attention computation.
    Memory savings happen in storage; attention uses reconstructed vectors.
    """

    def __init__(self, config: TurboQuantConfig = None, mode: str = "polar_only"):
        if config is None:
            config = TurboQuantConfig.preset("balanced")
        self.config = config
        self.mode = mode
        self.compressor = TurboQuantCompressor(config)
        self.polar = PolarQuantizer(config)

        # Compressed storage
        self._keys_compressed = None
        self._values_compressed = None
        self.offset = 0

        # Incremental dequant cache — avoid redundant reconstruction
        self._keys_decompressed = None
        self._values_decompressed = None
        self._decompressed_offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """
        Store new key-value pairs (compressed) and return all accumulated KV.

        Args:
            keys: (B, n_kv_heads, new_seq_len, head_dim)
            values: (B, n_kv_heads, new_seq_len, head_dim)

        Returns:
            (all_keys, all_values): dequantized full-precision tensors
        """
        # Quantize new keys (PolarQuant only — no QJL since adapter uses polar reconstruction)
        new_keys_q = self._quantize_polar(keys)

        # Quantize new values (PolarQuant only)
        new_values_q = self._quantize_polar(values)

        if self._keys_compressed is None:
            self._keys_compressed = new_keys_q
            self._values_compressed = new_values_q
        else:
            # Append compressed representations
            for k in new_keys_q:
                if isinstance(new_keys_q[k], mx.array):
                    self._keys_compressed[k] = mx.concatenate(
                        [self._keys_compressed[k], new_keys_q[k]], axis=2
                    )
                elif isinstance(new_keys_q[k], list):
                    self._keys_compressed[k] = [
                        mx.concatenate([old, new], axis=2)
                        for old, new in zip(self._keys_compressed[k], new_keys_q[k])
                    ]

            for k in new_values_q:
                if isinstance(new_values_q[k], mx.array):
                    self._values_compressed[k] = mx.concatenate(
                        [self._values_compressed[k], new_values_q[k]], axis=2
                    )
                elif isinstance(new_values_q[k], list):
                    self._values_compressed[k] = [
                        mx.concatenate([old, new], axis=2)
                        for old, new in zip(self._values_compressed[k], new_values_q[k])
                    ]

        self.offset += keys.shape[2]

        # Incremental dequantization — only reconstruct new tokens
        new_keys_decompressed = self._dequantize_polar(new_keys_q)
        new_values_decompressed = self._dequantize_polar(new_values_q)

        if self._keys_decompressed is None:
            self._keys_decompressed = new_keys_decompressed
            self._values_decompressed = new_values_decompressed
        else:
            self._keys_decompressed = mx.concatenate(
                [self._keys_decompressed, new_keys_decompressed], axis=2
            )
            self._values_decompressed = mx.concatenate(
                [self._values_decompressed, new_values_decompressed], axis=2
            )

        return self._keys_decompressed, self._values_decompressed

    def _quantize_polar(self, x: mx.array) -> dict:
        """PolarQuant-only quantization for values."""
        D = self.config.embed_dim
        block_size = self.config.polar_block_size

        y = x @ self.compressor.precondition.T
        y_blocks = y.reshape(*x.shape[:-1], D // block_size, block_size)
        angles, radius = self.polar.forward_polar(y_blocks)

        angle_indices = []
        for level, (angle_arr, codebook) in enumerate(zip(angles, self.polar.codebooks)):
            diffs = mx.abs(mx.expand_dims(angle_arr, axis=-1) - codebook)
            indices = mx.argmin(diffs, axis=-1)
            angle_indices.append(indices)

        return {
            "angle_indices": angle_indices,
            "radius": radius.astype(mx.float16),
        }

    def _dequantize_polar(self, compressed: dict) -> mx.array:
        """Reconstruct from PolarQuant compressed representation."""
        angle_indices = compressed["angle_indices"]
        radius = compressed["radius"].astype(mx.float32)

        quantized_angles = []
        for level_idx, codebook in zip(angle_indices, self.polar.codebooks):
            quantized_angles.append(codebook[level_idx])

        y_recon_blocks = self.polar.inverse_polar(quantized_angles, radius)
        D = self.config.embed_dim
        shape = list(y_recon_blocks.shape[:-2]) + [D]
        y_recon = y_recon_blocks.reshape(*shape)
        return y_recon @ self.compressor.precondition

    def __len__(self):
        return self.offset

    @property
    def state(self):
        """Return compressed state for serialization."""
        return self._keys_compressed, self._values_compressed

    @state.setter
    def state(self, v):
        self._keys_compressed, self._values_compressed = v
        if self._keys_compressed is not None:
            self.offset = self._keys_compressed["radius"].shape[2]

    def memory_bytes(self) -> int:
        """Estimate compressed memory usage."""
        if self._keys_compressed is None:
            return 0
        # Keys and values both use PolarQuant only (no QJL in adapter)
        radius = self._keys_compressed["radius"]
        B, H, N = radius.shape[0], radius.shape[1], radius.shape[2]
        D = self.config.embed_dim

        polar_bits = (D // 2) * self.config.polar_bits_level1
        for l in range(1, self.polar.n_levels):
            polar_bits += (D // (2 ** (l + 1))) * self.config.polar_bits_higher
        n_blocks = D // self.config.polar_block_size
        radius_bits = n_blocks * 16
        bits_per_vec = polar_bits + radius_bits

        # Key + value, both polar-only
        total_bits = B * H * N * bits_per_vec * 2
        return total_bits // 8

    def fp16_bytes(self) -> int:
        """What this cache would cost in FP16."""
        if self._keys_compressed is None:
            return 0
        radius = self._keys_compressed["radius"]
        B, H, N = radius.shape[0], radius.shape[1], radius.shape[2]
        return B * H * N * self.config.embed_dim * 2 * 2


def make_turboquant_cache(
    n_layers: int,
    embed_dim: int,
    n_kv_heads: int,
    preset: str = "balanced",
    per_layer: bool = False,
    aggressive_layers: int = 0,
) -> list:
    """
    Factory function to create a list of TurboQuantCache for all layers.

    Args:
        n_layers: number of transformer layers
        embed_dim: head dimension (not hidden_size!)
        n_kv_heads: number of KV heads
        preset: quality preset name
        per_layer: if True, use aggressive compression for bottom layers
        aggressive_layers: number of bottom layers to use aggressive compression

    Returns:
        List of TurboQuantCache, one per layer
    """
    if per_layer and aggressive_layers > 0:
        # Mixed allocation: aggressive for bottom, quality for top
        aggressive_config = TurboQuantConfig.preset(
            "fast", embed_dim=embed_dim, num_heads=n_kv_heads,
            polar_bits_level1=2, polar_bits_higher=1,
            value_bits_level1=2, value_bits_higher=1,
        )
        quality_config = TurboQuantConfig.preset(
            preset, embed_dim=embed_dim, num_heads=n_kv_heads,
        )
        caches = []
        for i in range(n_layers):
            if i < aggressive_layers:
                caches.append(TurboQuantCache(aggressive_config))
            else:
                caches.append(TurboQuantCache(quality_config))
        return caches
    else:
        config = TurboQuantConfig.preset(preset, embed_dim=embed_dim, num_heads=n_kv_heads)
        return [TurboQuantCache(config) for _ in range(n_layers)]
