"""
Bit-packing utilities for PolarQuant angle indices.

Level 1 indices: 4 bits each, packed 8 per uint32
Level 2+ indices: 2 bits each, packed 16 per uint32
"""

import mlx.core as mx


def pack_indices_4bit(indices: mx.array) -> mx.array:
    """Pack 4-bit indices (0-15) into uint32, 8 per word.

    Args:
        indices: (..., N) uint32 array with values 0-15

    Returns:
        packed: (..., ceil(N/8)) uint32
    """
    shape = indices.shape
    N = shape[-1]
    batch_shape = shape[:-1]

    # Pad to multiple of 8
    pad_n = (8 - N % 8) % 8
    if pad_n > 0:
        indices = mx.concatenate([indices, mx.zeros((*batch_shape, pad_n), dtype=mx.uint32)], axis=-1)

    # Reshape: (..., N/8, 8)
    reshaped = indices.reshape(*batch_shape, -1, 8)

    # Pack: shift each of 8 values into position
    shifts = mx.array([28, 24, 20, 16, 12, 8, 4, 0], dtype=mx.uint32)
    packed = mx.sum(reshaped * mx.power(mx.array(2, dtype=mx.uint32), shifts), axis=-1)

    return packed


def unpack_indices_4bit(packed: mx.array, n_indices: int) -> mx.array:
    """Unpack 4-bit indices from uint32.

    Args:
        packed: (..., ceil(N/8)) uint32
        n_indices: number of indices to unpack

    Returns:
        indices: (..., n_indices) uint32
    """
    batch_shape = packed.shape[:-1]

    # Expand: (..., n_packed, 8)
    expanded = mx.expand_dims(packed, axis=-1)
    shifts = mx.array([28, 24, 20, 16, 12, 8, 4, 0], dtype=mx.uint32)
    mask = mx.array(0xF, dtype=mx.uint32)
    indices = (expanded >> shifts) & mask
    indices = indices.reshape(*batch_shape, -1)

    return indices[..., :n_indices]


def pack_indices_2bit(indices: mx.array) -> mx.array:
    """Pack 2-bit indices (0-3) into uint32, 16 per word."""
    shape = indices.shape
    N = shape[-1]
    batch_shape = shape[:-1]

    pad_n = (16 - N % 16) % 16
    if pad_n > 0:
        indices = mx.concatenate([indices, mx.zeros((*batch_shape, pad_n), dtype=mx.uint32)], axis=-1)

    reshaped = indices.reshape(*batch_shape, -1, 16)
    shifts = mx.array([30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0], dtype=mx.uint32)
    packed = mx.sum(reshaped * mx.power(mx.array(2, dtype=mx.uint32), shifts), axis=-1)

    return packed


def unpack_indices_2bit(packed: mx.array, n_indices: int) -> mx.array:
    """Unpack 2-bit indices from uint32."""
    batch_shape = packed.shape[:-1]
    expanded = mx.expand_dims(packed, axis=-1)
    shifts = mx.array([30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0], dtype=mx.uint32)
    mask = mx.array(0x3, dtype=mx.uint32)
    indices = (expanded >> shifts) & mask
    indices = indices.reshape(*batch_shape, -1)
    return indices[..., :n_indices]
