"""Portable BC6H (unsigned) encoder in numpy, used where texconv (Windows-only) is unavailable.

Blocks use the one-region modes 11-14 (10-bit endpoints, or 11/12/16-bit base endpoints with a signed
delta) and 4-bit indices; each block keeps the mode with the lowest error. Endpoints are fitted in the
codec's interpolation domain (half-float bit patterns scaled by 64/31, roughly logarithmic in radiance)
and refined by least squares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_WEIGHTS = np.array([0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64], dtype=np.int64)
_CHUNK = 16384
_REFINEMENTS = 2


@dataclass(frozen=True)
class _Mode:
    code: int
    base_bits: int
    delta_bits: int  # 0: the second endpoint is stored at full precision

    def fields(self) -> list[tuple[str, int, int, bool]]:
        """(endpoint, channel, bit count, reversed) groups following the 10 low bits of each base channel."""
        if not self.delta_bits:
            return [("q1", c, 10, False) for c in range(3)]
        out = []
        for c in range(3):
            out.append(("d", c, self.delta_bits, False))
            out.append(("q0hi", c, self.base_bits - 10, self.base_bits > 11))
        return out


_MODES = (_Mode(0x03, 10, 0), _Mode(0x07, 11, 9), _Mode(0x0B, 12, 8), _Mode(0x0F, 16, 4))


def _half_domain(layer: np.ndarray) -> np.ndarray:
    """RGB of a float image as the pre-finish interpolation values the decoder produces."""
    rgb = np.clip(np.nan_to_num(layer[..., :3], nan=0.0, posinf=65000.0), 0.0, 65000.0).astype(np.float16)
    # the decoder ends with (value * 31) >> 6, so aim at the middle of each half value's input range
    return (rgb.view(np.uint16).astype(np.float32) + 0.5) * (64.0 / 31.0)


def _blocks(values: np.ndarray) -> np.ndarray:
    h, w = values.shape[:2]
    ph, pw = -h % 4, -w % 4
    if ph or pw:
        values = np.pad(values, ((0, ph), (0, pw), (0, 0)), mode="edge")
    bh, bw = values.shape[0] // 4, values.shape[1] // 4
    return values.reshape(bh, 4, bw, 4, 3).transpose(0, 2, 1, 3, 4).reshape(bh * bw, 16, 3)


def _unquantize(q: np.ndarray, bits: int) -> np.ndarray:
    if bits >= 15:
        return q
    return np.where(q == 0, 0, np.where(q == (1 << bits) - 1, 0xFFFF, ((q << 16) + 0x8000) >> bits))


def _quantize(mode: _Mode, low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bits = mode.base_bits
    step = float(1 << (16 - bits))

    def q(value: np.ndarray) -> np.ndarray:
        return np.clip(np.rint((value - step / 2) / step), 0, (1 << bits) - 1).astype(np.int64)

    q0, q1 = q(low), q(high)
    if mode.delta_bits:
        # symmetric range so that swapping the endpoints keeps the delta representable
        limit = (1 << (mode.delta_bits - 1)) - 1
        q1 = q0 + np.clip(q1 - q0, -limit, limit)
    return q0, q1


def _fit(pixels: np.ndarray, mode: _Mode, q0: np.ndarray, q1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best palette index per pixel and the block error for quantized endpoints."""
    e0 = _unquantize(q0, mode.base_bits)[:, None, :]
    e1 = _unquantize(q1, mode.base_bits)[:, None, :]
    palette = (((64 - _WEIGHTS)[None, :, None] * e0 + _WEIGHTS[None, :, None] * e1 + 32) >> 6).astype(np.float32)
    distance = np.square(pixels[:, :, None, :] - palette[:, None, :, :]).sum(axis=-1)
    indices = distance.argmin(axis=-1)
    error = np.take_along_axis(distance, indices[..., None], axis=-1)[..., 0].sum(axis=-1)
    return indices, error


def _principal_endpoints(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = pixels.mean(axis=1, keepdims=True)
    centered = pixels - mean
    covariance = np.einsum("npi,npj->nij", centered, centered)
    axis = np.ones((pixels.shape[0], 3), np.float32)
    for _ in range(8):
        axis = np.einsum("nij,nj->ni", covariance, axis)
        axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    projection = np.einsum("npi,ni->np", centered, axis)
    low = mean[:, 0] + projection.min(axis=1)[:, None] * axis
    high = mean[:, 0] + projection.max(axis=1)[:, None] * axis
    return low, high


def _least_squares(pixels: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha = _WEIGHTS[indices].astype(np.float32) / 64.0
    beta = 1.0 - alpha
    aa = (alpha * alpha).sum(axis=1)
    bb = (beta * beta).sum(axis=1)
    ab = (alpha * beta).sum(axis=1)
    ap = np.einsum("np,npc->nc", alpha, pixels)
    bp = np.einsum("np,npc->nc", beta, pixels)
    det = aa * bb - ab * ab
    singular = np.abs(det) < 1e-6
    det = np.where(singular, 1.0, det)[:, None]
    low = (aa[:, None] * bp - ab[:, None] * ap) / det
    high = (bb[:, None] * ap - ab[:, None] * bp) / det
    mean = pixels.mean(axis=1)
    return np.where(singular[:, None], mean, low), np.where(singular[:, None], mean, high)


def _encode_mode(pixels: np.ndarray, mode: _Mode, start: tuple[np.ndarray, np.ndarray]):
    q0, q1 = _quantize(mode, *start)
    indices, error = _fit(pixels, mode, q0, q1)
    for _ in range(_REFINEMENTS):
        r0, r1 = _quantize(mode, *_least_squares(pixels, indices))
        r_indices, r_error = _fit(pixels, mode, r0, r1)
        better = r_error < error
        q0 = np.where(better[:, None], r0, q0)
        q1 = np.where(better[:, None], r1, q1)
        indices = np.where(better[:, None], r_indices, indices)
        error = np.minimum(error, r_error)
    return q0, q1, indices, error


def _reverse(value: np.ndarray, bits: int) -> np.ndarray:
    out = np.zeros_like(value)
    for bit in range(bits):
        out |= ((value >> np.uint64(bit)) & np.uint64(1)) << np.uint64(bits - 1 - bit)
    return out


def _pack(mode: _Mode, q0: np.ndarray, q1: np.ndarray, indices: np.ndarray) -> np.ndarray:
    # the first pixel's index is stored without its top bit, so it must be below 8
    flip = indices[:, 0] >= 8
    q0, q1 = np.where(flip[:, None], q1, q0), np.where(flip[:, None], q0, q1)
    indices = np.where(flip[:, None], 15 - indices, indices).astype(np.uint64)
    q0 = q0.astype(np.uint64)
    values = {
        "q1": q1.astype(np.uint64),
        "d": (q1 - q0.astype(np.int64)).astype(np.uint64) & np.uint64((1 << max(mode.delta_bits, 1)) - 1),
        "q0hi": q0 >> np.uint64(10),
    }

    fields = [(np.full(len(q0), mode.code, np.uint64), 5)]
    fields += [(q0[:, c] & np.uint64(0x3FF), 10) for c in range(3)]
    for name, c, bits, reverse in mode.fields():
        value = values[name][:, c]
        fields.append((_reverse(value, bits) if reverse else value, bits))
    fields += [(indices[:, 0], 3)] + [(indices[:, p], 4) for p in range(1, 16)]

    low = np.zeros(len(q0), np.uint64)
    high = np.zeros(len(q0), np.uint64)
    offset = 0
    for value, bits in fields:
        if offset >= 64:
            high |= value << np.uint64(offset - 64)
        else:
            low |= value << np.uint64(offset)
            if offset + bits > 64:
                high |= value >> np.uint64(64 - offset)
        offset += bits
    assert offset == 128
    return np.stack([low, high], axis=1).astype("<u8")


def encode_layer(layer: np.ndarray) -> bytes:
    """Encodes one float RGB(A) image (alpha ignored) to BC6H_UF16 blocks in row-major order."""
    blocks = _blocks(_half_domain(layer))
    out = []
    for start in range(0, len(blocks), _CHUNK):
        pixels = blocks[start : start + _CHUNK]
        initial = _principal_endpoints(pixels)
        best_error = np.full(len(pixels), np.inf)
        best_mode = np.zeros(len(pixels), np.int64)
        results = []
        for number, mode in enumerate(_MODES):
            result = _encode_mode(pixels, mode, initial)
            results.append(result)
            better = result[3] < best_error
            best_mode[better] = number
            best_error[better] = result[3][better]
        packed = np.empty((len(pixels), 2), "<u8")
        for number, mode in enumerate(_MODES):
            chosen = best_mode == number
            if chosen.any():
                q0, q1, indices, _ = results[number]
                packed[chosen] = _pack(mode, q0[chosen], q1[chosen], indices[chosen])
        out.append(packed.tobytes())
    return b"".join(out)


def decode_layer(data: bytes, width: int, height: int) -> np.ndarray:
    """Decodes BC6H_UF16 blocks written by encode_layer (modes 11-14) to float32 RGB; for tests and checks."""
    words = np.frombuffer(data, dtype="<u8").reshape(-1, 2)
    low, high = words[:, 0], words[:, 1]

    def bits(offset: int, count: int) -> np.ndarray:
        mask = np.uint64((1 << count) - 1)
        if offset >= 64:
            return (high >> np.uint64(offset - 64)) & mask
        value = low >> np.uint64(offset)
        if offset + count > 64:
            value |= high << np.uint64(64 - offset)
        return value & mask

    codes = bits(0, 5)
    q0 = np.stack([bits(5 + 10 * c, 10) for c in range(3)], axis=1)
    q1 = np.zeros_like(q0)
    base_bits = np.zeros(len(codes), np.int64)
    known = np.zeros(len(codes), bool)
    for mode in _MODES:
        chosen = codes == mode.code
        known |= chosen
        base_bits[chosen] = mode.base_bits
        offset = 35
        hi = np.zeros_like(q0)
        for name, c, count, reverse in mode.fields():
            value = bits(offset, count)
            value = _reverse(value, count) if reverse else value
            if name == "q1":
                q1[chosen, c] = value[chosen]
            elif name == "q0hi":
                hi[:, c] = value
            offset += count
        if mode.delta_bits:
            q0[chosen] |= hi[chosen] << np.uint64(10)
            for c in range(3):
                value = bits(35 + 10 * c, mode.delta_bits)
                signed = value.astype(np.int64) - ((value >> np.uint64(mode.delta_bits - 1)) << np.uint64(mode.delta_bits)).astype(np.int64)
                q1[chosen, c] = ((q0[chosen, c].astype(np.int64) + signed[chosen]) & ((1 << mode.base_bits) - 1)).astype(np.uint64)
    if not known.all():
        raise ValueError("only BC6H modes 11-14 are supported")
    q0 = q0.astype(np.int64)
    q1 = q1.astype(np.int64)
    e0 = np.empty_like(q0)
    e1 = np.empty_like(q1)
    for mode in _MODES:
        chosen = base_bits == mode.base_bits
        e0[chosen] = _unquantize(q0[chosen], mode.base_bits)
        e1[chosen] = _unquantize(q1[chosen], mode.base_bits)
    indices = np.stack([bits(65, 3)] + [bits(68 + 4 * (p - 1), 4) for p in range(1, 16)], axis=1).astype(np.int64)
    weights = _WEIGHTS[indices][..., None]
    values = (((64 - weights) * e0[:, None, :] + weights * e1[:, None, :] + 32) >> 6) * 31 >> 6
    rgb = values.astype(np.uint16).view(np.float16).astype(np.float32)
    bw, bh = -(-width // 4), -(-height // 4)
    image = rgb.reshape(bh, bw, 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 3)
    return image[:height, :width]
