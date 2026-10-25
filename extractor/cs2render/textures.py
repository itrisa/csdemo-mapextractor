"""Bakes CS2 material layers into packed, block-compressed texture array layers.

Each texture set produces three streams:
  C  sRGB albedo with colour correction + tint-mask colour adjust baked in; alpha = opacity.
  N  tangent-space normal xy (runtime convention, contrast baked), z is reconstructed.
  M  linear: R roughness, G ambient occlusion (levels baked), B tint mask, A height / metalness / self-illum.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import etcpak
import numpy as np
from PIL import Image

from .materials import LayerBake, layer_matrices

Image.MAX_IMAGE_PIXELS = None

# WebGL internal format names the client maps to GL enums.
FORMAT_BC1_SRGB = "bc1-srgb"
FORMAT_BC7_SRGB = "bc7-srgb"
FORMAT_BC7 = "bc7"
FORMAT_BC5 = "bc5"
FORMAT_BC4 = "bc4"
FORMAT_BC6H = "bc6h"
FORMAT_BC1 = "bc1"

_FORMAT_CODES = {
    FORMAT_BC1_SRGB: 1,
    FORMAT_BC7_SRGB: 2,
    FORMAT_BC7: 3,
    FORMAT_BC5: 4,
    FORMAT_BC4: 5,
    FORMAT_BC6H: 6,
    FORMAT_BC1: 7,
}
_BLOCK_BYTES = {
    FORMAT_BC1_SRGB: 8,
    FORMAT_BC1: 8,
    FORMAT_BC4: 8,
    FORMAT_BC7_SRGB: 16,
    FORMAT_BC7: 16,
    FORMAT_BC5: 16,
    FORMAT_BC6H: 16,
}

# Storage format of every texture role.
FORMATS: dict[str, str] = {
    "color": FORMAT_BC1_SRGB,
    "color_alpha": FORMAT_BC7_SRGB,
    "normal": FORMAT_BC5,
    "mask": FORMAT_BC7,
    "overlay": FORMAT_BC1,
    "directional": FORMAT_BC7,
    "shadows": FORMAT_BC4,
    "hdr": FORMAT_BC6H,
}


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return np.where(value <= 0.0031308, value * 12.92, 1.055 * np.power(value, 1.0 / 2.4) - 0.055)


class TextureSource:
    """Loads exported PNGs as float32 [0, 1] arrays, resized on demand."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def path(self, vtex: str) -> Path:
        return self.directory / Path(vtex).name.replace(".vtex", ".png")

    def size(self, vtex: str | None) -> tuple[int, int]:
        if not vtex:
            return (4, 4)
        with Image.open(self.path(vtex)) as image:
            return image.size

    def load(self, vtex: str | None, size: int) -> np.ndarray | None:
        if not vtex:
            return None
        return _load_resized(str(self.path(vtex)), size)


@lru_cache(maxsize=64)
def _load_resized(path: str, size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGBA")
        if image.size != (size, size):
            # Per channel: Pillow premultiplies RGBA when resampling, which zeroes packed masks whose alpha is 0.
            method = Image.Resampling.BOX if image.width >= size else Image.Resampling.BICUBIC
            image = Image.merge("RGBA", [band.resize((size, size), method) for band in image.split()])
        return np.asarray(image, dtype=np.float32) / 255.0


@lru_cache(maxsize=512)
def _average_linear(path: str) -> tuple[float, float, float]:
    data = _load_resized(path, 64)
    linear = srgb_to_linear(data[..., :3].astype(np.float64))
    return tuple(float(x) for x in linear.reshape(-1, 3).mean(axis=0))


def _contrast(value: np.ndarray, contrast_brightness: tuple[float, float]) -> np.ndarray:
    contrast, brightness = contrast_brightness
    return np.clip(((value - 0.5) * contrast + 0.5) * brightness, 0.0, 1.0)


def _ao_levels(ao: np.ndarray, levels: tuple[float, float, float]) -> np.ndarray:
    low, gamma, high = levels
    return low + (high - low) * np.power(np.maximum(ao, 0.0), max(gamma, 0.001))


def bake_color(layer: LayerBake, source: TextureSource, size: int) -> np.ndarray:
    """Returns (size, size, 4) float: linear RGB albedo + opacity."""
    color = source.load(layer.color, size)
    if color is None:
        rgb = np.array(layer.constant_color or (1.0, 1.0, 1.0), dtype=np.float32)
        color = np.ones((size, size, 4), dtype=np.float32)
        color[..., :3] = rgb
        linear = srgb_to_linear(color[..., :3])
    else:
        linear = srgb_to_linear(color[..., :3])
    alpha = color[..., 3:4]

    if layer.kind == "environment":
        height = source.load(layer.height, size)
        tint_mask = _contrast(height[..., 1:2], layer.tint_mask_contrast) if height is not None else np.zeros_like(alpha)
        average = np.array(_average_linear(str(source.path(layer.color)))) if layer.color else linear.reshape(-1, 3).mean(0)
        adjust, color_adjust = layer_matrices(layer, average)
        vector = np.concatenate([linear, alpha], axis=-1)
        corrected = (vector @ adjust)[..., :3] if layer.color_correct else linear
        tinted = (vector @ color_adjust)[..., :3]
        linear = corrected + (tinted - corrected) * tint_mask

    opacity = alpha if layer.needs_alpha else np.ones_like(alpha)
    if layer.alpha_remap != (0.0, 1.0):
        low, high = layer.alpha_remap
        opacity = low + (high - low) * opacity
    return np.concatenate([np.clip(linear, 0.0, 1.0), np.clip(opacity, 0.0, 1.0)], axis=-1).astype(np.float32)


def bake_normal(layer: LayerBake, source: TextureSource, size: int) -> np.ndarray:
    """Returns (size, size, 3) float unit normals in the runtime tangent convention."""
    raw = source.load(layer.normal, size)
    if raw is None:
        normal = np.zeros((size, size, 3), dtype=np.float32)
        normal[..., 2] = 1.0
        return normal
    x = raw[..., 0] * 2.0 - 1.0
    y = -(raw[..., 1] * 2.0 - 1.0)
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
    normal = np.stack([x, y, z], axis=-1)
    if layer.normal_contrast != 1.0:
        normal = np.array([0.0, 0.0, 1.0]) + (normal - np.array([0.0, 0.0, 1.0])) * layer.normal_contrast
        normal[..., 2] = np.maximum(normal[..., 2], 0.05)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    return normal.astype(np.float32)


def bake_mask(layer: LayerBake, source: TextureSource, size: int) -> np.ndarray:
    """Returns (size, size, 4) float packed roughness / AO / tint mask / extra."""
    normal = source.load(layer.normal, size)
    roughness = normal[..., 3] if normal is not None else np.full((size, size), 0.7, dtype=np.float32)
    mask = np.zeros((size, size, 4), dtype=np.float32)
    if layer.kind == "environment":
        mask[..., 0] = _contrast(roughness, layer.rough_contrast)
        height = source.load(layer.height, size)
        color = source.load(layer.color, size)
        if layer.ao_source == "height.b" and height is not None:
            ao = height[..., 2]
        elif color is not None:
            ao = color[..., 3]
        else:
            ao = np.ones((size, size), dtype=np.float32)
        mask[..., 1] = _ao_levels(ao, layer.ao_levels)
        mask[..., 2] = _contrast(height[..., 1], layer.tint_mask_contrast) if height is not None else 0.0
        mask[..., 3] = height[..., 0] if height is not None else 0.5
    else:
        mask[..., 0] = roughness
        ao = source.load(layer.ao, size)
        mask[..., 1] = ao[..., 0] if ao is not None else 1.0
        tint = source.load(layer.tint_mask, size)
        mask[..., 2] = tint[..., 0] if tint is not None else 1.0
        if layer.mask_alpha == "metal.g":
            mask[..., 3] = source.load(layer.metal, size)[..., 1]
        elif layer.mask_alpha == "selfillum.r":
            mask[..., 3] = source.load(layer.self_illum, size)[..., 0]
    return mask


# --- Mip chains ---


def _downsample(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return image.reshape(h // 2, 2, w // 2, 2, -1).mean(axis=(1, 3))


def _coverage(alpha: np.ndarray, reference: float) -> float:
    return float((alpha >= reference).mean())


def _scale_alpha_to_coverage(alpha: np.ndarray, target: float, reference: float) -> np.ndarray:
    """Preserves alpha-test coverage across mips so foliage does not thin out with distance."""
    lo, hi = 0.0, 4.0
    for _ in range(12):
        mid = (lo + hi) / 2
        if _coverage(np.clip(alpha * mid, 0, 1), reference) < target:
            lo = mid
        else:
            hi = mid
    return np.clip(alpha * ((lo + hi) / 2), 0.0, 1.0)


def mip_chain(image: np.ndarray, kind: str, alpha_reference: float | None = None) -> list[np.ndarray]:
    """kind: color (linear rgb + alpha), normal (unit xyz), mask (linear)."""
    levels = [image]
    coverage = _coverage(image[..., 3], alpha_reference) if alpha_reference is not None else None
    current = image
    while current.shape[0] > 4:
        current = _downsample(current)
        if kind == "normal":
            current = current / np.maximum(np.linalg.norm(current, axis=-1, keepdims=True), 1e-6)
        elif kind == "color" and coverage is not None:
            current = current.copy()
            current[..., 3] = _scale_alpha_to_coverage(current[..., 3], coverage, alpha_reference)
        levels.append(current)
    return levels


# --- Encoding ---


def _to_rgba8(image: np.ndarray) -> bytes:
    data = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    if data.shape[-1] < 4:
        pad = np.zeros(data.shape[:2] + (4 - data.shape[-1],), dtype=np.uint8)
        pad[..., -1] = 255
        data = np.concatenate([data, pad], axis=-1)
    return np.ascontiguousarray(data).tobytes()


def encode_level(image: np.ndarray, fmt: str) -> bytes:
    """image: float (h, w, c) already in the storage encoding (sRGB for colour)."""
    h, w = image.shape[:2]
    data = _to_rgba8(image)
    if fmt in (FORMAT_BC1, FORMAT_BC1_SRGB):
        return etcpak.compress_bc1(data, w, h)
    if fmt in (FORMAT_BC7, FORMAT_BC7_SRGB):
        return etcpak.compress_bc7(data, w, h)
    if fmt == FORMAT_BC5:
        return etcpak.compress_bc5(data, w, h)
    if fmt == FORMAT_BC4:
        return etcpak.compress_bc4(data, w, h)
    raise ValueError(fmt)


def encode_color_chain(levels: list[np.ndarray], fmt: str) -> list[bytes]:
    out = []
    for level in levels:
        storage = np.concatenate([linear_to_srgb(level[..., :3]), level[..., 3:4]], axis=-1)
        out.append(encode_level(storage, fmt))
    return out


def encode_normal_chain(levels: list[np.ndarray], fmt: str = FORMAT_BC5) -> list[bytes]:
    """xy in RG, z is reconstructed by the shader."""
    return [encode_level(np.concatenate([level[..., :2] * 0.5 + 0.5, np.zeros_like(level[..., :1])], axis=-1), fmt) for level in levels]


def encode_mask_chain(levels: list[np.ndarray], fmt: str = FORMAT_BC7) -> list[bytes]:
    return [encode_level(level, fmt) for level in levels]


@dataclass
class ArrayTexture:
    fmt: str
    width: int
    height: int
    layers: list[list[bytes]]  # per layer, per mip

    def write(self, path: Path) -> int:
        """CTEX: magic, version, format, width, height, layers, mips, per-mip byte sizes, then mip-major data."""
        mips = len(self.layers[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.write(b"CTEX")
            handle.write(struct.pack("<6I", 1, _FORMAT_CODES[self.fmt], self.width, self.height, len(self.layers), mips))
            for mip in range(mips):
                handle.write(struct.pack("<I", sum(len(layer[mip]) for layer in self.layers)))
            for mip in range(mips):
                for layer in self.layers:
                    handle.write(layer[mip])
        return path.stat().st_size


def write_single(path: Path, fmt: str, width: int, height: int, levels: list[bytes]) -> int:
    return ArrayTexture(fmt, width, height, [levels]).write(path)


def block_bytes(fmt: str, width: int, height: int) -> int:
    return max(1, -(-width // 4)) * max(1, -(-height // 4)) * _BLOCK_BYTES[fmt]
