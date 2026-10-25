"""ctypes bindings for meshoptimizer entry points the Python package leaves undeclared or unexposed."""

from __future__ import annotations

import ctypes

import numpy as np
from meshoptimizer._loader import lib

_P = ctypes.c_void_p
_SIZE = ctypes.c_size_t

lib.meshopt_simplifyWithAttributes.argtypes = [
    _P, _P, _SIZE, _P, _SIZE, _SIZE, _P, _SIZE, _P, _SIZE, _P, _SIZE, ctypes.c_float, ctypes.c_uint, _P
]
lib.meshopt_simplifyWithAttributes.restype = _SIZE
lib.meshopt_encodeFilterExp.argtypes = [_P, _SIZE, _SIZE, ctypes.c_int, _P, ctypes.c_int]
lib.meshopt_encodeFilterExp.restype = None

SIMPLIFY_LOCK_BORDER = 1
SIMPLIFY_ERROR_ABSOLUTE = 4
EXP_SHARED_COMPONENT = 2


def simplify_with_attributes(
    indices: np.ndarray,
    positions: np.ndarray,
    attributes: np.ndarray,
    weights: np.ndarray,
    target_error: float,
    options: int = SIMPLIFY_LOCK_BORDER | SIMPLIFY_ERROR_ABSOLUTE,
) -> np.ndarray:
    """Error-bound simplification (no triangle target); returns the new index list."""
    indices = np.ascontiguousarray(indices, dtype=np.uint32)
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    attributes = np.ascontiguousarray(attributes, dtype=np.float32)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    destination = np.zeros_like(indices)
    error = ctypes.c_float(0.0)
    count = lib.meshopt_simplifyWithAttributes(
        destination.ctypes.data,
        indices.ctypes.data,
        len(indices),
        positions.ctypes.data,
        len(positions),
        positions.shape[1] * 4,
        attributes.ctypes.data,
        attributes.shape[1] * 4,
        weights.ctypes.data,
        len(weights),
        None,
        0,
        target_error,
        options,
        ctypes.addressof(error),
    )
    return destination[:count]


def encode_filter_exp(values: np.ndarray, bits: int, mode: int = EXP_SHARED_COMPONENT) -> np.ndarray:
    """Encodes float vectors for the EXPONENTIAL vertex filter; returns uint32 with the same shape."""
    data = np.ascontiguousarray(values, dtype=np.float32)
    out = np.zeros(data.shape, dtype=np.uint32)
    lib.meshopt_encodeFilterExp(out.ctypes.data, len(data), data.shape[1] * 4, bits, data.ctypes.data, mode)
    return out
