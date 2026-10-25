"""Thin ctypes bindings for meshoptimizer calls the Python wrapper leaves undeclared."""

from __future__ import annotations

import ctypes

import numpy as np
from meshoptimizer import SIMPLIFY_ERROR_ABSOLUTE, SIMPLIFY_LOCK_BORDER
from meshoptimizer._loader import lib

_f = ctypes.POINTER(ctypes.c_float)
_u = ctypes.POINTER(ctypes.c_uint)
_b = ctypes.POINTER(ctypes.c_ubyte)
_size = ctypes.c_size_t

lib.meshopt_simplifyWithAttributes.argtypes = [
    _u, _u, _size, _f, _size, _size, _f, _size, _f, _size, _b, _size, ctypes.c_float, ctypes.c_uint, _f,
]
lib.meshopt_simplifyWithAttributes.restype = _size


def simplify_absolute(
    indices: np.ndarray,
    positions: np.ndarray,
    attributes: np.ndarray,
    weights: np.ndarray,
    max_error: float,
) -> np.ndarray:
    """Removes triangles while keeping geometric deviation under max_error world units.

    Open borders stay locked so neighbouring meshes keep sharing edges without cracks.
    """
    indices = np.ascontiguousarray(indices, dtype=np.uint32)
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    attributes = np.ascontiguousarray(attributes, dtype=np.float32)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    destination = np.empty_like(indices)
    count = lib.meshopt_simplifyWithAttributes(
        destination.ctypes.data_as(_u),
        indices.ctypes.data_as(_u),
        len(indices),
        positions.ctypes.data_as(_f),
        len(positions),
        positions.strides[0],
        attributes.ctypes.data_as(_f),
        attributes.shape[1] * 4,
        weights.ctypes.data_as(_f),
        len(weights),
        _b(),
        0,
        float(max_error),
        SIMPLIFY_ERROR_ABSOLUTE | SIMPLIFY_LOCK_BORDER,
        _f(),
    )
    return destination[:count]
