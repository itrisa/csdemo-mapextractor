"""Packs baked CS2 lighting: lightmaps and box-projected reflection probes."""

from __future__ import annotations

import math
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import OpenEXR
from PIL import Image

from . import bc6h
from . import textures as tx

Image.MAX_IMAGE_PIXELS = None

_DXGI_R16G16B16A16_FLOAT = 10
_DXGI_BC6H_UF16 = 95


def read_exr(path: Path) -> np.ndarray:
    channels = OpenEXR.File(str(path)).parts[0].channels
    if "RGBA" in channels:
        return np.asarray(channels["RGBA"].pixels, dtype=np.float32)
    if "RGB" in channels:
        rgb = np.asarray(channels["RGB"].pixels, dtype=np.float32)
        return np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], axis=-1)
    stacked = [np.asarray(channels[name].pixels, dtype=np.float32) for name in ("R", "G", "B")]
    return np.stack(stacked + [np.ones_like(stacked[0])], axis=-1)


def _masked_downsample(image: np.ndarray, factor: int, valid: np.ndarray | None) -> np.ndarray:
    if factor == 1:
        return image
    h, w, c = image.shape
    if valid is None:
        return image.reshape(h // factor, factor, w // factor, factor, c).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
    weights = valid.astype(np.float32).reshape(h // factor, factor, w // factor, factor, 1)
    summed = (image.reshape(h // factor, factor, w // factor, factor, c) * weights).sum(axis=(1, 3))
    count = weights.sum(axis=(1, 3))
    fallback = image.reshape(h // factor, factor, w // factor, factor, c).mean(axis=(1, 3))
    return np.where(count > 0, summed / np.maximum(count, 1e-6), fallback).astype(np.float32)


# --- BC6H via texconv, or the portable numpy encoder without it ---


def _write_dds_f16_array(path: Path, layers: list[np.ndarray]) -> None:
    h, w = layers[0].shape[:2]
    header = bytearray(128)
    struct.pack_into("<4sI", header, 0, b"DDS ", 124)
    struct.pack_into("<IIIII", header, 8, 0x1 | 0x2 | 0x4 | 0x1000 | 0x8, h, w, w * 8, 0)
    struct.pack_into("<I", header, 28, 1)
    struct.pack_into("<II4s", header, 76, 32, 0x4, b"DX10")
    struct.pack_into("<I", header, 108, 0x1000)
    dx10 = struct.pack("<IIIII", _DXGI_R16G16B16A16_FLOAT, 3, 0, len(layers), 0)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(dx10)
        for layer in layers:
            data = np.ascontiguousarray(np.clip(layer, 0.0, 65000.0).astype(np.float16))
            handle.write(data.tobytes())


def encode_bc6h(layers: list[np.ndarray], texconv: Path | None) -> list[bytes]:
    """Encodes equally sized float RGBA images to BC6H (unsigned); returns per-layer bytes."""
    h, w = layers[0].shape[:2]
    if texconv is None:
        return [bc6h.encode_layer(layer) for layer in layers]
    with tempfile.TemporaryDirectory(prefix="cs2render_bc6h_") as tmp:
        source = Path(tmp) / "input.dds"
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        _write_dds_f16_array(source, layers)
        subprocess.run(
            [str(texconv), "-nologo", "-y", "-f", "BC6H_UF16", "-m", "1", "-o", str(out_dir), str(source)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        data = (out_dir / "input.dds").read_bytes()
    assert data[:4] == b"DDS " and data[84:88] == b"DX10"
    fmt = struct.unpack_from("<I", data, 128)[0]
    if fmt != _DXGI_BC6H_UF16:
        raise RuntimeError(f"texconv produced DXGI format {fmt}, expected BC6H_UF16")
    size = tx.block_bytes(tx.FORMAT_BC6H, w, h)
    payload = data[148:]
    return [payload[i * size : (i + 1) * size] for i in range(len(layers))]


# --- Lightmaps ---


@dataclass
class LightmapResult:
    size: int
    files: dict[str, str]
    bytes: int


def pack_lightmaps(lightmap_dir: Path, out_dir: Path, size: int, shadow_size: int, texconv: Path | None) -> LightmapResult:
    total = 0
    files = {"irradiance": "lm_irradiance.ctex", "directional": "lm_directional.ctex", "shadows": "lm_shadows.ctex"}
    out_dir.mkdir(parents=True, exist_ok=True)

    irradiance = read_exr(lightmap_dir / "irradiance.exr")
    valid = irradiance[..., :3].max(axis=-1) > 0
    factor = irradiance.shape[0] // size
    irradiance = _masked_downsample(irradiance, factor, valid)
    irradiance[..., 3] = 1.0
    del valid
    blocks = encode_bc6h([irradiance], texconv)
    del irradiance
    total += tx.write_single(out_dir / files["irradiance"], tx.FORMATS["hdr"], size, size, [blocks[0]])

    with Image.open(lightmap_dir / "directional_irradiance.png") as image:
        directional = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    directional = _masked_downsample(directional, max(1, directional.shape[0] // size), None)
    fmt = tx.FORMATS["directional"]
    total += tx.write_single(out_dir / files["directional"], fmt, size, size, [tx.encode_level(directional, fmt)])

    with Image.open(lightmap_dir / "direct_light_shadows.png") as image:
        shadows = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    shadows = _masked_downsample(shadows, max(1, shadows.shape[0] // shadow_size), None)
    fmt = tx.FORMATS["shadows"]
    total += tx.write_single(out_dir / files["shadows"], fmt, shadow_size, shadow_size, [tx.encode_level(shadows, fmt)])
    return LightmapResult(size=size, files=files, bytes=total)


# --- Reflection probes ---


def equirect_directions(height: int, width: int) -> np.ndarray:
    """Source-space sample directions of Source2Viewer's equirect cube export (row 0 is up)."""
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    theta = (-2.0 * u - 1.0) * math.pi
    phi = (-2.0 * v - 1.0) * (math.pi / 2)
    cos_phi = np.cos(phi)[:, None]
    return np.stack(
        [cos_phi * np.cos(theta)[None, :], cos_phi * np.sin(theta)[None, :], -np.repeat(np.sin(phi)[:, None], width, axis=1)],
        axis=-1,
    )


def equirect_solid_angles(height: int, width: int) -> np.ndarray:
    v = (np.arange(height) + 0.5) / height
    latitude = (0.5 - v) * math.pi
    return np.repeat((np.cos(latitude) * (2 * math.pi / width) * (math.pi / height))[:, None], width, axis=1)


def octahedral_directions(size: int) -> np.ndarray:
    """Matches the client's octDecode: uv in [0,1]^2 -> unit direction (Source space, z up)."""
    f = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    x, y = np.meshgrid(f, f)
    z = 1.0 - np.abs(x) - np.abs(y)
    t = np.clip(-z, 0.0, None)
    x = x - np.where(x >= 0, t, -t)
    y = y - np.where(y >= 0, t, -t)
    d = np.stack([x, y, z], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _downsample_equirect(image: np.ndarray, height: int) -> np.ndarray:
    factor = image.shape[0] // height
    return image.reshape(height, factor, image.shape[1] // factor, factor, -1).mean(axis=(1, 3))


def _sample_equirect(equirect: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Bilinear lookup, the inverse of equirect_directions."""
    h, w = equirect.shape[:2]
    theta = np.arctan2(directions[:, 1], directions[:, 0])
    phi = -np.arcsin(np.clip(directions[:, 2], -1.0, 1.0))
    u = np.mod((-(theta / math.pi) - 1.0) / 2.0, 1.0)
    v = np.mod((-(phi / (math.pi / 2)) - 1.0) / 2.0, 1.0)
    x = u * w - 0.5
    y = np.clip(v * h - 0.5, 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x0 %= w
    x1 = (x0 + 1) % w
    y1 = np.minimum(y0 + 1, h - 1)
    rgb = equirect[..., :3]
    top = rgb[y0, x0] * (1 - fx) + rgb[y0, x1] * fx
    bottom = rgb[y1, x0] * (1 - fx) + rgb[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def prefilter_probe(equirect: np.ndarray, size: int, levels: int) -> list[np.ndarray]:
    """GGX-prefiltered octahedral mips; mip k is lit for perceptual roughness (k / (levels - 1))^2."""
    out = []
    for level in range(levels):
        level_size = max(size >> level, 4)
        directions = octahedral_directions(level_size).reshape(-1, 3)
        sqrt_roughness = level / max(levels - 1, 1)
        alpha = max((sqrt_roughness**2) ** 2, 1e-4)
        if level <= 1:
            # The lobe is narrower than an output texel; resample from a matching-resolution source.
            source = _downsample_equirect(equirect, min(level_size * 2, equirect.shape[0]))
            result = _sample_equirect(source, directions).astype(np.float32)
            image = np.concatenate([result, np.ones((result.shape[0], 1), np.float32)], axis=1)
            out.append(image.reshape(level_size, level_size, 4))
            continue
        source_height = int(np.clip(level_size, 16, 64))
        source = _downsample_equirect(equirect, min(source_height, equirect.shape[0]))
        sh, sw = source.shape[:2]
        source_dirs = equirect_directions(sh, sw).reshape(-1, 3)
        solid = equirect_solid_angles(sh, sw).reshape(-1)
        radiance = source.reshape(-1, source.shape[-1])[:, :3]
        result = np.empty((directions.shape[0], 3), dtype=np.float32)
        a2 = alpha * alpha
        for start in range(0, directions.shape[0], 2048):
            n_dot_l = directions[start : start + 2048] @ source_dirs.T
            # N = V = R: half-vector lobe in terms of n.l; D(h) with cos(theta_h) = sqrt((1 + n.l) / 2).
            n_dot_h2 = np.clip((1.0 + n_dot_l) * 0.5, 0.0, 1.0)
            d = a2 / np.square(n_dot_h2 * (a2 - 1.0) + 1.0)
            weight = d * np.clip(n_dot_l, 0.0, None) * solid[None, :]
            total = weight.sum(axis=1, keepdims=True)
            result[start : start + 2048] = (weight @ radiance) / np.maximum(total, 1e-12)
        image = np.concatenate([result, np.ones((result.shape[0], 1), np.float32)], axis=1)
        out.append(image.reshape(level_size, level_size, 4))
    return out


def normalization_sh(equirect: np.ndarray) -> list[float]:
    """L1 luminance harmonic packed as VRF's EnvMapData.NormalizationSH (dot with (N, 1))."""
    small = _downsample_equirect(equirect, 64)
    h, w = small.shape[:2]
    directions = equirect_directions(h, w).reshape(-1, 3)
    solid = equirect_solid_angles(h, w).reshape(-1)
    luma = small.reshape(-1, small.shape[-1])[:, :3] @ np.array([0.2125, 0.7154, 0.0721])
    c0 = float((luma * solid).sum() * 0.282095)
    c1 = (luma[:, None] * directions * solid[:, None]).sum(axis=0) * 0.488603
    return [float(c1[0] * 0.325735), float(c1[1] * 0.325735), float(c1[2] * 0.325735), c0 * 0.282095]


def _angles_matrix(angles: tuple[float, float, float]) -> np.ndarray:
    """Source QAngle (pitch, yaw, roll) to a 3x3 rotation whose columns are forward, left, up."""
    pitch, yaw, roll = (math.radians(a) for a in angles)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    sr, cr = math.sin(roll), math.cos(roll)
    forward = [cp * cy, cp * sy, -sp]
    left = [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp]
    up = [cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp]
    return np.array([forward, left, up]).T


def _vec(value: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if value is None:
        return np.array(default, dtype=np.float64)
    if isinstance(value, str):
        return np.array([float(x) for x in value.split()], dtype=np.float64)
    return np.array([float(x) for x in value], dtype=np.float64)


BOUNDS_EXTEND = 0.0


def pack_probes(
    cubemap_dir: Path,
    entities: list[dict[str, Any]],
    out_dir: Path,
    size: int,
    levels: int,
    texconv: Path | None,
) -> tuple[list[dict[str, Any]], int]:
    """Writes probes.ctex into `out_dir`; returns probe metadata and the file size."""
    probes = []
    for entity in entities:
        if entity.get("classname") not in ("env_combined_light_probe_volume", "env_cubemap_box"):
            continue
        index = int(entity.get("array_index", -1))
        source = cubemap_dir / f"env_cubemap_array_f{index:02d}.exr"
        if index < 0 or not source.is_file():
            continue
        rotation = _angles_matrix(tuple(_vec(entity.get("angles"))))
        origin = _vec(entity.get("origin"))
        probes.append(
            {
                "arrayIndex": index,
                "origin": origin,
                "rotation": rotation,
                "boxMins": _vec(entity.get("box_mins"), (-72, -72, -72)) - BOUNDS_EXTEND,
                "boxMaxs": _vec(entity.get("box_maxs"), (72, 72, 72)) + BOUNDS_EXTEND,
                "edgeFade": _vec(entity.get("edge_fade_dists"), (8, 8, 8)) + BOUNDS_EXTEND,
                "priority": int(entity.get("priority", 0) or 0),
                "source": source,
            }
        )
    # VRF iterates probes in scene order; keep priority first so higher-priority volumes win the blend.
    probes.sort(key=lambda p: (-p["priority"], float(np.prod(p["boxMaxs"] - p["boxMins"]))))

    per_level: list[list[np.ndarray]] = [[] for _ in range(levels)]
    metadata = []
    for slot, probe in enumerate(probes):
        equirect = read_exr(probe["source"])
        for level, image in enumerate(prefilter_probe(equirect, size, levels)):
            per_level[level].append(image)
        rotation = probe["rotation"]
        world_to_local = np.eye(4)
        world_to_local[:3, :3] = rotation.T
        world_to_local[:3, 3] = -rotation.T @ probe["origin"]
        metadata.append(
            {
                "slot": slot,
                "worldToLocal": [round(float(x), 6) for x in world_to_local[:3, :].T.reshape(-1)],
                "boxMins": [round(float(x), 3) for x in probe["boxMins"]],
                "boxMaxs": [round(float(x), 3) for x in probe["boxMaxs"]],
                "invEdge": [round(float(1.0 / max(x, 1e-3)), 6) for x in probe["edgeFade"]],
                "normalizationSH": [round(x, 6) for x in normalization_sh(equirect)],
            }
        )

    encoded = [encode_bc6h(layers, texconv) for layers in per_level]
    layers = [[encoded[level][slot] for level in range(levels)] for slot in range(len(probes))]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = tx.ArrayTexture(tx.FORMATS["hdr"], size, size, layers).write(out_dir / "probes.ctex")
    return metadata, written


class ProbeLocator:
    """Vectorized form of probe_for_point."""

    def __init__(self, metadata: list[dict[str, Any]]) -> None:
        self.slots = np.array([p["slot"] for p in metadata], dtype=np.int64)
        self.matrices = np.stack([np.array(p["worldToLocal"]).reshape(4, 3).T for p in metadata]) if metadata else None
        self.mins = np.array([p["boxMins"] for p in metadata])
        self.maxs = np.array([p["boxMaxs"] for p in metadata])

    def __call__(self, point: np.ndarray) -> int:
        if self.matrices is None:
            return 0
        local = self.matrices[:, :, :3] @ point + self.matrices[:, :, 3]
        outside = np.maximum(self.mins - local, 0) + np.maximum(local - self.maxs, 0)
        distance = np.linalg.norm(outside, axis=1)
        inside = np.flatnonzero(distance == 0)
        return int(self.slots[inside[0]] if len(inside) else self.slots[int(np.argmin(distance))])


def probe_for_point(metadata: list[dict[str, Any]], point: np.ndarray) -> int:
    """First (highest-priority, smallest) probe box containing the point, else the nearest box."""
    best, best_distance = 0, math.inf
    for probe in metadata:
        m = np.array(probe["worldToLocal"]).reshape(4, 3).T
        local = m[:, :3] @ point + m[:, 3]
        mins, maxs = np.array(probe["boxMins"]), np.array(probe["boxMaxs"])
        outside = np.maximum(mins - local, 0) + np.maximum(local - maxs, 0)
        distance = float(np.linalg.norm(outside))
        if distance == 0:
            return probe["slot"]
        if distance < best_distance:
            best, best_distance = probe["slot"], distance
    return best
