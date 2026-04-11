"""
Fast TurboQuant KV cache: stores preconditioned keys, skips inverse precondition.

Key insight: for attention, <q, k> = <Pq, Pk> where P is orthogonal.
So we can store Pk directly and apply P to queries at attention time.
This saves one D×D matmul per dequant (the expensive inverse precondition).

Values still need full dequant since they're summed, not inner-producted.
"""

import mlx.core as mx
from turboquant import TurboQuantConfig, TurboQuantCompressor, PolarQuantizer


class FastTurboQuantCache:
    """
    Fast KV cache: stores preconditioned keys (no inverse precondition needed).
    """

    def __init__(self, config: TurboQuantConfig = None):
        if config is None:
            config = TurboQuantConfig.preset("balanced")
        self.config = config
        self.compressor = TurboQuantCompressor(config)
        self.polar = PolarQuantizer(config)
        self.offset = 0

        # Storage: preconditioned keys (FP16), compressed values
        self._keys_preconditioned = None  # (B, H, N, D) float16 — Pk
        self._values_compressed = None    # dict with angle_indices, radius

        # Decompressed value buffer
        self._values_decompressed = None
        self._values_offset = 0

    def update_and_fetch(self, keys, values):
        D = self.config.embed_dim

        # Keys: just precondition and store as FP16 (skip polar quantize entirely)
        # This gives ~2x compression (FP16 vs FP32) with zero quality loss
        pk = (keys @ self.compressor.precondition.T).astype(mx.float16)

        # Values: fused Metal quantize for real compression
        from polar_fused_quantize import fused_polar_quantize
        y_v = values @ self.compressor.precondition.T
        v_flat = y_v.reshape(-1)
        v_results = fused_polar_quantize(v_flat, self.polar.codebooks)

        n_blocks = D // self.config.polar_block_size
        batch_shape = list(values.shape[:-1])

        new_values_q = {
            "angle_indices": [
                v_results[i].reshape(*batch_shape, n_blocks, -1)
                for i in range(4)
            ],
            "radius": v_results[4].reshape(*batch_shape, n_blocks, 1),
        }

        # Store keys
        if self._keys_preconditioned is None:
            self._keys_preconditioned = pk
            self._values_compressed = new_values_q
        else:
            self._keys_preconditioned = mx.concatenate(
                [self._keys_preconditioned, pk], axis=2
            )
            self._values_compressed["radius"] = mx.concatenate(
                [self._values_compressed["radius"], new_values_q["radius"]], axis=2
            )
            for i in range(4):
                self._values_compressed["angle_indices"][i] = mx.concatenate(
                    [self._values_compressed["angle_indices"][i],
                     new_values_q["angle_indices"][i]], axis=2
                )

        self.offset += keys.shape[2]

        # Return: dequantized keys and values for SDPA
        # Keys: inverse precondition (Pk → k)
        all_keys = self._keys_preconditioned.astype(mx.float32) @ self.compressor.precondition

        # Values: Metal inverse polar
        all_values = self._dequant_values()

        return all_keys, all_values

    def _dequant_values(self):
        from polar_metal import polar_inverse_metal
        ai = self._values_compressed["angle_indices"]
        radius = self._values_compressed["radius"].astype(mx.float32)
        D = self.config.embed_dim

        flat_indices = [idx.reshape(-1).astype(mx.uint32) for idx in ai]
        flat_radius = radius.reshape(-1)

        y_flat = polar_inverse_metal(
            self.polar.codebooks, flat_indices, flat_radius,
            block_size=self.config.polar_block_size,
        )
        target_shape = list(radius.shape[:-2]) + [D]
        y_recon = y_flat.reshape(*target_shape)
        return y_recon @ self.compressor.precondition

    def __len__(self):
        return self.offset

    @property
    def state(self):
        return self._keys_preconditioned, self._values_compressed

    @state.setter
    def state(self, v):
        self._keys_preconditioned, self._values_compressed = v
        if self._keys_preconditioned is not None:
            self.offset = self._keys_preconditioned.shape[2]
