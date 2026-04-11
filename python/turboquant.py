"""
TurboQuant: Extreme KV cache compression for Apple Silicon.

Implements QJL + PolarQuant as Metal compute shaders via MLX,
enabling Gemma 4 31B at full 256K context on M1 Max 64GB.
"""

import mlx.core as mx
import numpy as np
from pathlib import Path
from dataclasses import dataclass

KERNEL_DIR = Path(__file__).parent.parent / "kernels" / "metal"


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache compression."""
    embed_dim: int = 128          # per-head embedding dimension
    num_heads: int = 8            # number of KV heads
    sketch_dim: int = 128         # QJL projection dimension (typically = embed_dim)
    n_outliers: int = 32          # number of outlier channels
    polar_block_size: int = 16    # PolarQuant block size (must be power of 2)
    polar_bits_level1: int = 4    # bits for level-1 angles
    polar_bits_higher: int = 2    # bits for level 2+ angles
    radius_bits: int = 16         # bits for radius (FP16)
    value_bits_level1: int = 0    # 0 = same as key bits
    value_bits_higher: int = 0    # 0 = same as key bits
    use_hadamard: bool = True     # use randomized Hadamard instead of full random matrix
    seed: int = 42

    @property
    def val_bits_l1(self) -> int:
        return self.value_bits_level1 if self.value_bits_level1 > 0 else self.polar_bits_level1

    @property
    def val_bits_higher(self) -> int:
        return self.value_bits_higher if self.value_bits_higher > 0 else self.polar_bits_higher

    @classmethod
    def preset(cls, name: str, embed_dim: int = 128, num_heads: int = 8,
               **kwargs) -> "TurboQuantConfig":
        """Create config from a named quality preset.

        Presets (all fit Gemma 4 31B @ 256K in 64GB):
          fast:     cos_sim~0.94, 4.0x compression, ~23 GB KV cache
          balanced: cos_sim~0.98, 3.56x compression, ~26 GB KV cache (default)
          quality:  cos_sim~0.99, 3.24x compression, ~29 GB KV cache
          hifi:     cos_sim~0.995, 2.69x compression, ~35 GB KV cache
        """
        presets = {
            "fast":     {"polar_bits_level1": 3, "polar_bits_higher": 2},
            "balanced": {"polar_bits_level1": 4, "polar_bits_higher": 2},
            "quality":  {"polar_bits_level1": 4, "polar_bits_higher": 2,
                         "value_bits_level1": 4, "value_bits_higher": 3},
            "hifi":     {"polar_bits_level1": 4, "polar_bits_higher": 3,
                         "value_bits_level1": 6, "value_bits_higher": 3},
            "gemma4_256k": {"polar_bits_level1": 3, "polar_bits_higher": 1,
                            "value_bits_level1": 2, "value_bits_higher": 1},
        }
        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Choose from: {list(presets.keys())}")
        params = {**presets[name], "embed_dim": embed_dim, "num_heads": num_heads, **kwargs}
        return cls(**params)


class QJLProjection:
    """QJL random projection matrix and quantization."""

    def __init__(self, config: TurboQuantConfig):
        self.config = config
        # Generate random projection matrix via QR of Gaussian
        rng = np.random.RandomState(config.seed)
        gaussian = rng.randn(config.embed_dim, config.sketch_dim).astype(np.float32)
        q, r = np.linalg.qr(gaussian)
        # q is (embed_dim, min(embed_dim, sketch_dim)), take first sketch_dim columns
        # then transpose to get (sketch_dim, embed_dim)
        proj = q[:, :config.sketch_dim].T  # (sketch_dim, embed_dim)
        self.projection = mx.array(proj * np.sqrt(config.embed_dim))
        self.scale = np.sqrt(np.pi / 2) / config.sketch_dim

    def quantize_keys(self, keys: mx.array) -> dict:
        """
        Quantize key vectors to sign bits, packed into uint8.

        Args:
            keys: (batch, n_heads, seq_len, embed_dim)

        Returns:
            dict with packed_bits (uint8), norms
        """
        # TODO: Metal kernel implementation
        # For now, pure MLX reference implementation
        sketch = keys @ self.projection.T  # (B, H, N, sketch_dim)
        sign_bits = mx.where(sketch >= 0, mx.array(1, dtype=mx.uint8),
                             mx.array(0, dtype=mx.uint8))
        norms = mx.sqrt(mx.sum(keys * keys, axis=-1, keepdims=True))

        # Pack 8 sign bits into each uint8
        B, H, N, S = sign_bits.shape
        sign_bits_reshaped = sign_bits.reshape(B, H, N, S // 8, 8)
        shifts = mx.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=mx.uint8)
        packed = mx.sum(sign_bits_reshaped * mx.power(mx.array(2, dtype=mx.uint8), shifts), axis=-1)

        return {
            "packed_bits": packed,  # (B, H, N, sketch_dim/8)
            "sign_bits": sign_bits,  # keep unpacked for score() reference impl
            "norms": norms,
        }

    def score(self, queries: mx.array, quantized_keys: dict) -> mx.array:
        """
        Compute attention logits: full-precision query × quantized keys.

        Args:
            queries: (batch, n_heads, seq_len_q, embed_dim)
            quantized_keys: output of quantize_keys

        Returns:
            scores: (batch, n_heads, seq_len_q, seq_len_k)
        """
        # TODO: Metal kernel implementation
        query_sketch = queries @ self.projection.T  # (B, H, Nq, sketch_dim)
        sign_bits = quantized_keys["sign_bits"]  # (B, H, Nk, sketch_dim)
        norms = quantized_keys["norms"]  # (B, H, Nk, 1)

        # Convert 0/1 bits to -1/+1
        signs = 2.0 * sign_bits.astype(mx.float32) - 1.0

        # Asymmetric inner product: query_sketch @ signs^T
        signs_t = mx.transpose(signs, axes=(0, 1, 3, 2))
        inner = query_sketch @ signs_t

        # Scale by sqrt(pi/2) / sketch_dim * ||k||
        norms_t = mx.transpose(norms, axes=(0, 1, 3, 2))
        scores = mx.array(self.scale) * inner * norms_t

        return scores


class PolarQuantizer:
    """PolarQuant recursive polar coordinate quantization."""

    def __init__(self, config: TurboQuantConfig):
        self.config = config
        self.n_levels = int(np.log2(config.polar_block_size))
        self.codebooks = self._compute_codebooks()

    def _compute_codebooks(self) -> list:
        """Compute Lloyd-Max codebooks for each level from analytical distributions."""
        from scipy import integrate, special

        codebooks = []

        for level in range(1, self.n_levels + 1):
            if level == 1:
                n_bits = self.config.polar_bits_level1
                n_centroids = 2 ** n_bits
                # Uniform on [0, 2pi)
                cb = np.linspace(0, 2 * np.pi, n_centroids, endpoint=False)
                cb += np.pi / n_centroids  # center of each bin
            else:
                n_bits = self.config.polar_bits_higher
                n_centroids = 2 ** n_bits
                n = 2 ** (level - 1)

                def pdf(psi, n=n):
                    log_norm = (special.gammaln(n)
                                - (n - 2) * np.log(2)
                                - 2 * special.gammaln(n / 2))
                    return np.exp(log_norm) * np.sin(2 * psi) ** (n - 1)

                # Lloyd-Max iteration
                cb = np.linspace(0, np.pi / 2, n_centroids + 2)[1:-1]
                for _ in range(50):
                    boundaries = np.concatenate(
                        [[0], (cb[:-1] + cb[1:]) / 2, [np.pi / 2]]
                    )
                    for i in range(n_centroids):
                        num, _ = integrate.quad(
                            lambda x: x * pdf(x), boundaries[i], boundaries[i + 1]
                        )
                        den, _ = integrate.quad(
                            pdf, boundaries[i], boundaries[i + 1]
                        )
                        if den > 1e-15:
                            cb[i] = num / den

            codebooks.append(mx.array(cb.astype(np.float32)))

        return codebooks

    def forward_polar(self, y: mx.array) -> tuple:
        """
        Forward polar transform: Cartesian → (angles, radius).

        Args:
            y: (..., block_size) preconditioned vector block

        Returns:
            angles: list of angle arrays per level
            radius: final radius scalar
        """
        # TODO: Metal kernel implementation
        r = y
        all_angles = []

        for level in range(self.n_levels):
            block_len = r.shape[-1]
            r_reshaped = r.reshape(*r.shape[:-1], block_len // 2, 2)
            a = r_reshaped[..., 0]
            b = r_reshaped[..., 1]

            if level == 0:
                angles = mx.arctan2(b, a)  # returns [-pi, pi]
                # Wrap to [0, 2pi) for level 1 codebook
                angles = mx.where(angles < 0, angles + 2 * np.pi, angles)
            else:
                angles = mx.arctan2(b, a)  # [0, pi/2] since inputs are norms (positive)

            radius = mx.sqrt(a * a + b * b)
            all_angles.append(angles)
            r = radius

        return all_angles, r  # r is now scalar radius

    def inverse_polar(self, quantized_angles: list, radius: mx.array) -> mx.array:
        """
        Inverse polar transform: (quantized angles, radius) → Cartesian.

        Args:
            quantized_angles: list of angle arrays (one per level), already dequantized to float
            radius: final radius

        Returns:
            y: (..., block_size) reconstructed vector
        """
        # TODO: Metal kernel implementation
        r = radius

        for level in range(self.n_levels - 1, -1, -1):
            angles = quantized_angles[level]
            a = r * mx.cos(angles)
            b = r * mx.sin(angles)
            r = mx.concatenate([
                mx.expand_dims(a, axis=-1),
                mx.expand_dims(b, axis=-1)
            ], axis=-1).reshape(*a.shape[:-1], a.shape[-1] * 2)

        return r


class TurboQuantCompressor:
    """
    TurboQuant_prod: PolarQuant (b-1 bits) + QJL (1-bit residual correction).

    This is Algorithm 2 from the TurboQuant paper.
    """

    def __init__(self, config: TurboQuantConfig):
        self.config = config
        self.polar = PolarQuantizer(config)
        self.qjl = QJLProjection(config)
        # Preconditioning matrix (random orthogonal)
        rng = np.random.RandomState(config.seed + 1)
        gaussian = rng.randn(config.embed_dim, config.embed_dim).astype(np.float32)
        q, _ = np.linalg.qr(gaussian)
        self.precondition = mx.array(q)

    def quantize(self, x: mx.array, use_metal: bool = False) -> dict:
        """
        TurboQuant_prod quantization: PolarQuant (b-1 bits) + QJL (1 bit) on residual.

        Args:
            x: (..., embed_dim) vectors to quantize
            use_metal: use Metal kernel for polar forward transform

        Returns:
            dict with angle_indices, radius, sign_bits, residual_norm
        """
        orig_shape = x.shape
        D = self.config.embed_dim
        block_size = self.config.polar_block_size

        # Step 1: Precondition
        y = x @ self.precondition.T  # (..., D)

        # Step 2: PolarQuant (b-1 bits)
        y_blocks = y.reshape(*orig_shape[:-1], D // block_size, block_size)

        if use_metal:
            from polar_metal import polar_forward_metal
            y_flat = y_blocks.reshape(-1)
            a_l1, a_l2, a_l3, a_l4, rad = polar_forward_metal(y_flat)
            # Reshape to match Python forward_polar output:
            #   angles[l] shape: (*batch_dims, n_blocks, angles_per_block_at_level)
            #   radius shape: (*batch_dims, n_blocks, 1)
            # batch_dims = y_blocks.shape[:-2] = (B, H, N, n_blocks)
            # But y_blocks.shape = (B, H, N, n_blocks_per_vec, block_size)
            # So batch_dims for angles = (B, H, N, n_blocks_per_vec)
            n_blocks_per_vec = D // block_size
            outer_dims = list(y_blocks.shape[:-2])  # (B, H, N, n_blocks_per_vec)
            # Metal processes all blocks flat: total_blocks * angles_per_level
            angles = [
                a_l1.reshape(*outer_dims, 8),
                a_l2.reshape(*outer_dims, 4),
                a_l3.reshape(*outer_dims, 2),
                a_l4.reshape(*outer_dims, 1),
            ]
            radius = rad.reshape(*outer_dims, 1)
        else:
            angles, radius = self.polar.forward_polar(y_blocks)

        # Quantize angles to nearest codebook entries
        # Use uint8 for indices (max 16 centroids at 4-bit = fits in uint8)
        angle_indices = []
        quantized_angles = []
        for level, (angle_arr, codebook) in enumerate(zip(angles, self.polar.codebooks)):
            diffs = mx.abs(mx.expand_dims(angle_arr, axis=-1) - codebook)
            indices = mx.argmin(diffs, axis=-1).astype(mx.uint8)
            angle_indices.append(indices)
            quantized_angles.append(codebook[indices.astype(mx.int32)])

        # Reconstruct from quantized angles to compute residual
        y_recon_blocks = self.polar.inverse_polar(quantized_angles, radius)
        y_recon = y_recon_blocks.reshape(orig_shape)
        x_recon = y_recon @ self.precondition  # P^T for orthogonal P

        # Step 3: QJL on residual (1 bit)
        residual = x - x_recon
        residual_norm = mx.sqrt(mx.sum(residual * residual, axis=-1, keepdims=True))
        sketch = residual @ self.qjl.projection.T
        sign_bits = mx.where(sketch >= 0, mx.array(1, dtype=mx.uint8),
                             mx.array(0, dtype=mx.uint8))

        return {
            "angle_indices": angle_indices,
            "radius": radius.astype(mx.float16),
            "sign_bits": sign_bits,
            "residual_norm": residual_norm,
        }

    def _reconstruct_polar(self, quantized: dict, use_metal: bool = False) -> mx.array:
        """Reconstruct PolarQuant component from stored indices."""
        angle_indices = quantized["angle_indices"]
        radius = quantized["radius"].astype(mx.float32)
        D = self.config.embed_dim

        if use_metal:
            from polar_metal import polar_inverse_metal
            # Flatten all indices and radius for Metal dispatch
            orig_shape = radius.shape  # (..., n_blocks_per_vec, 1) or (..., n_blocks_per_vec)
            flat_indices = [idx.reshape(-1).astype(mx.uint32) for idx in angle_indices]
            flat_radius = radius.reshape(-1)
            y_flat = polar_inverse_metal(
                self.polar.codebooks, flat_indices, flat_radius, block_size=16
            )
            # Reshape: flat → (..., D)
            # radius shape is (B, H, N, n_blocks_per_vec, 1) or (B, H, N, n_blocks_per_vec)
            target_shape = list(radius.shape[:-2]) + [D]
            y_recon = y_flat.reshape(*target_shape)
        else:
            quantized_angles = []
            for level_idx, codebook in zip(angle_indices, self.polar.codebooks):
                quantized_angles.append(codebook[level_idx.astype(mx.int32)])

            y_recon_blocks = self.polar.inverse_polar(quantized_angles, radius)
            shape = list(y_recon_blocks.shape[:-2]) + [D]
            y_recon = y_recon_blocks.reshape(*shape)

        return y_recon @ self.precondition  # inverse precondition

    def dequantize(self, quantized: dict) -> mx.array:
        """Reconstruct vector from TurboQuant compressed representation."""
        x_polar = self._reconstruct_polar(quantized)
        residual_norm = quantized["residual_norm"]
        sign_bits = quantized["sign_bits"]

        signs = 2.0 * sign_bits.astype(mx.float32) - 1.0
        scale = np.sqrt(np.pi / 2) / self.config.sketch_dim
        x_qjl = scale * residual_norm * (signs @ self.qjl.projection)

        return x_polar + x_qjl

    def score(self, queries: mx.array, quantized: dict,
              use_block_fused: bool = True, skip_qjl: bool = False) -> mx.array:
        """
        Asymmetric inner product: full-precision query × TurboQuant compressed key.

        Uses block-fused Metal kernel for polar scoring when available.

        Args:
            queries: (B, H, Nq, D)
            quantized: output of quantize()

        Returns:
            scores: (B, H, Nq, Nk)
        """
        residual_norm = quantized["residual_norm"]       # (B, H, Nk, 1)
        sign_bits = quantized["sign_bits"]               # (B, H, Nk, sketch_dim)
        D = self.config.embed_dim

        # Precondition query: Pq (O(D²) once, not per key)
        pq = queries @ self.precondition.T  # (B, H, Nq, D)

        if use_block_fused and D == 128 and self.config.polar_block_size == 16:
            from turboquant_blockscore import polar_batched_block_score
            B, H, Nq, _ = queries.shape
            Nk = sign_bits.shape[2]
            BH = B * H

            # Flatten all indices for batched dispatch
            flat_l1 = quantized["angle_indices"][0].reshape(-1).astype(mx.uint32)
            flat_l2 = quantized["angle_indices"][1].reshape(-1).astype(mx.uint32)
            flat_l3 = quantized["angle_indices"][2].reshape(-1).astype(mx.uint32)
            flat_l4 = quantized["angle_indices"][3].reshape(-1).astype(mx.uint32)
            flat_radii = quantized["radius"].reshape(-1).astype(mx.float32)

            # Single dispatch for all heads
            score_polar = polar_batched_block_score(
                pq.reshape(BH, Nq, D),
                self.polar.codebooks,
                [flat_l1, flat_l2, flat_l3, flat_l4],
                flat_radii, Nq, Nk, BH,
            ).reshape(B, H, Nq, Nk)
        else:
            # Fallback: materialize full reconstruction
            angle_indices = quantized["angle_indices"]
            radius = quantized["radius"].astype(mx.float32)
            quantized_angles = []
            for level_idx, codebook in zip(angle_indices, self.polar.codebooks):
                quantized_angles.append(codebook[level_idx.astype(mx.int32)])
            y_recon_blocks = self.polar.inverse_polar(quantized_angles, radius)
            shape = list(y_recon_blocks.shape[:-2]) + [D]
            y_recon = y_recon_blocks.reshape(*shape)
            y_recon_t = mx.transpose(y_recon, axes=(0, 1, 3, 2))
            score_polar = pq @ y_recon_t

        if skip_qjl:
            return score_polar

        # QJL component on residual
        query_sketch = queries @ self.qjl.projection.T  # (B, H, Nq, sketch_dim)
        signs = 2.0 * sign_bits.astype(mx.float32) - 1.0
        signs_t = mx.transpose(signs, axes=(0, 1, 3, 2))
        inner = query_sketch @ signs_t                    # (B, H, Nq, Nk)

        scale = np.sqrt(np.pi / 2) / self.config.sketch_dim
        norms_t = mx.transpose(residual_norm, axes=(0, 1, 3, 2))
        score_qjl = scale * inner * norms_t

        return score_polar + score_qjl


class TurboQuantKVCache:
    """
    TurboQuant-compressed KV cache for transformer inference.

    Combines PolarQuant (b-1 bits) + QJL (1-bit residual) for
    near-lossless b-bit KV cache compression.
    """

    def __init__(self, config: TurboQuantConfig, use_metal: bool = False,
                 mode: str = "qjl_only"):
        """
        Args:
            config: TurboQuantConfig
            use_metal: use Metal kernels for scoring
            mode: "qjl_only", "turboquant" (PolarQuant+QJL), or "polar_only" (PolarQuant, no QJL)
        """
        self.config = config
        self.mode = mode
        self.use_metal = use_metal

        if mode == "qjl_only":
            self.qjl = QJLProjection(config)
        elif mode in ("turboquant", "polar_only"):
            self.compressor = TurboQuantCompressor(config)
            self.qjl = self.compressor.qjl
        self.polar = PolarQuantizer(config)

        # Value-specific PolarQuantizer for asymmetric bit allocation
        if config.value_bits_level1 > 0 or config.value_bits_higher > 0:
            val_config = TurboQuantConfig(
                embed_dim=config.embed_dim, num_heads=config.num_heads,
                polar_bits_level1=config.val_bits_l1,
                polar_bits_higher=config.val_bits_higher,
                polar_block_size=config.polar_block_size,
                seed=config.seed,
            )
            self.val_polar = PolarQuantizer(val_config)
        else:
            self.val_polar = self.polar

        if use_metal:
            from qjl_metal import QJLProjectionMetal
            self.qjl_metal = QJLProjectionMetal(
                config.embed_dim, config.sketch_dim, config.seed
            )

        # Storage
        self.keys = None
        self.values = None
        self.length = 0

    def update(self, keys: mx.array, values: mx.array):
        """
        Add new key-value pairs to the cache (quantized).

        Args:
            keys: (batch, n_heads, new_seq_len, embed_dim)
            values: (batch, n_heads, new_seq_len, embed_dim)
        """
        if self.mode in ("turboquant", "polar_only"):
            quantized = self.compressor.quantize(keys)
        else:
            quantized = self.qjl.quantize_keys(keys)

        # Compress values
        if self.mode in ("turboquant", "polar_only"):
            values_compressed = self._quantize_values(values)
        else:
            values_compressed = values.astype(mx.float16)

        if self.keys is None:
            self.keys = quantized
            self.values = values_compressed
        else:
            for k in quantized:
                if isinstance(quantized[k], mx.array):
                    self.keys[k] = mx.concatenate(
                        [self.keys[k], quantized[k]], axis=2
                    )
                elif isinstance(quantized[k], list):
                    self.keys[k] = [
                        mx.concatenate([old, new], axis=2)
                        for old, new in zip(self.keys[k], quantized[k])
                    ]
            if isinstance(self.values, dict):
                # Append compressed value dicts
                for k in values_compressed:
                    if isinstance(values_compressed[k], mx.array):
                        self.values[k] = mx.concatenate(
                            [self.values[k], values_compressed[k]], axis=2
                        )
                    elif isinstance(values_compressed[k], list):
                        self.values[k] = [
                            mx.concatenate([old, new], axis=2)
                            for old, new in zip(self.values[k], values_compressed[k])
                        ]
            else:
                self.values = mx.concatenate([self.values, values_compressed], axis=2)

        self.length += keys.shape[2]

    def attention(self, queries: mx.array) -> mx.array:
        """
        Compute attention output using quantized KV cache.

        Args:
            queries: (batch, n_heads, seq_len_q, embed_dim)

        Returns:
            output: (batch, n_heads, seq_len_q, embed_dim)
        """
        if self.mode in ("turboquant", "polar_only"):
            return self._attention_turboquant(queries)

        # QJL-only path
        scores = self.qjl.score(queries, self.keys)
        scale = self.config.embed_dim ** -0.5
        scores = scores * scale
        scores = mx.softmax(scores, axis=-1)
        output = scores @ self.values
        return output

    def _quantize_values(self, values: mx.array) -> dict:
        """Quantize values with PolarQuant (possibly asymmetric bit allocation)."""
        D = self.config.embed_dim
        block_size = self.config.polar_block_size

        y = values @ self.compressor.precondition.T
        y_blocks = y.reshape(*values.shape[:-1], D // block_size, block_size)
        angles, radius = self.val_polar.forward_polar(y_blocks)

        angle_indices = []
        for level, (angle_arr, codebook) in enumerate(zip(angles, self.val_polar.codebooks)):
            diffs = mx.abs(mx.expand_dims(angle_arr, axis=-1) - codebook)
            indices = mx.argmin(diffs, axis=-1).astype(mx.uint8)
            angle_indices.append(indices)

        return {
            "angle_indices": angle_indices,
            "radius": radius.astype(mx.float16),
        }

    def _dequantize_values(self, compressed_values: dict) -> mx.array:
        """Reconstruct values from PolarQuant compressed representation."""
        angle_indices = compressed_values["angle_indices"]
        radius = compressed_values["radius"].astype(mx.float32)

        quantized_angles = []
        for level_idx, codebook in zip(angle_indices, self.val_polar.codebooks):
            quantized_angles.append(codebook[level_idx.astype(mx.int32)])

        y_recon_blocks = self.val_polar.inverse_polar(quantized_angles, radius)
        D = self.config.embed_dim
        shape = list(y_recon_blocks.shape[:-2]) + [D]
        y_recon = y_recon_blocks.reshape(*shape)
        return y_recon @ self.compressor.precondition  # inverse precondition

    def _attention_turboquant(self, queries: mx.array) -> mx.array:
        """Full TurboQuant attention: PolarQuant + QJL scoring."""
        skip_qjl = (self.mode == "polar_only")
        scores = self.compressor.score(queries, self.keys, skip_qjl=skip_qjl)
        scale = self.config.embed_dim ** -0.5
        scores = scores * scale
        scores = mx.softmax(scores, axis=-1)

        # Dequantize values if compressed
        if isinstance(self.values, dict):
            values = self._dequantize_values(self.values)
        else:
            values = self.values

        output = scores @ values
        return output

    def memory_bytes(self) -> int:
        """Estimate memory usage of the compressed cache."""
        if self.keys is None:
            return 0

        if self.mode in ("turboquant", "polar_only"):
            shape = self.keys["sign_bits"].shape  # (B, H, N, sketch_dim)
            B, H, N, S = shape
            D = self.config.embed_dim
            block_size = self.config.polar_block_size

            # Polar angle indices bits per token:
            #   Level 1: D/2 angles × polar_bits_level1
            #   Level 2: D/4 angles × polar_bits_higher
            #   Level 3: D/8 angles × polar_bits_higher
            #   Level 4: D/16 angles × polar_bits_higher
            polar_bits = (D // 2) * self.config.polar_bits_level1
            for level in range(1, self.polar.n_levels):
                polar_bits += (D // (2 ** (level + 1))) * self.config.polar_bits_higher

            # Radius: FP16 per block
            n_blocks = D // self.config.polar_block_size
            radius_bits = n_blocks * 16

            # QJL: 1 bit per sketch_dim + residual norm (FP32)
            qjl_bits = S + 32

            total_bits_per_token = polar_bits + radius_bits + qjl_bits
            key_bytes = B * H * N * total_bits_per_token // 8

            # Values: PolarQuant compressed (same bit allocation as keys)
            val_polar_bits = polar_bits + radius_bits
            val_bytes = B * H * N * val_polar_bits // 8
            return key_bytes + val_bytes

        # QJL-only
        B, H, N, _ = self.keys["packed_bits"].shape
        key_bytes = B * H * N * (self.config.sketch_dim // 8)
        norm_bytes = B * H * N * 4
        val_bytes = B * H * N * self.config.embed_dim * 2  # FP16
        return key_bytes + norm_bytes + val_bytes

    def fp16_equivalent_bytes(self) -> int:
        """What this cache would cost in FP16."""
        if self.keys is None:
            return 0
        if self.mode in ("turboquant", "polar_only"):
            shape = self.keys["sign_bits"].shape
            B, H, N = shape[0], shape[1], shape[2]
        else:
            B, H, N, _ = self.keys["packed_bits"].shape
        return B * H * N * self.config.embed_dim * 2 * 2  # keys + values, FP16


if __name__ == "__main__":
    # Quick smoke test
    config = TurboQuantConfig(embed_dim=128, num_heads=8)
    cache = TurboQuantKVCache(config)

    # Simulate a batch of key-value pairs
    B, H, N, D = 1, 8, 32, 128
    keys = mx.random.normal((B, H, N, D))
    values = mx.random.normal((B, H, N, D))
    queries = mx.random.normal((B, H, 1, D))

    cache.update(keys, values)
    output = cache.attention(queries)

    print(f"Output shape: {output.shape}")
    print(f"Compressed cache: {cache.memory_bytes() / 1024:.1f} KB")
    print(f"FP16 equivalent:  {cache.fp16_equivalent_bytes() / 1024:.1f} KB")
    print(f"Compression ratio: {cache.fp16_equivalent_bytes() / cache.memory_bytes():.1f}x")
