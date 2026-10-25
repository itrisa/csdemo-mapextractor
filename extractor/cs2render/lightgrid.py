"""Lighting for dynamic objects (weapons, later players): CS2's light probe volumes resampled to small grids.

The game lights a moving model from the light probe volume atlas around it: a six-direction ambient cube plus
the baked sun shadow. The full atlas (millions of probes, float RGB) is far too big for the web, so the
package keeps what a single model needs, sampled once per object on the CPU:

  luma    ambient-cube luminance per direction at a fine grid (the part that changes from sun to shade)
  chroma  ambient-cube colour per direction at a coarse grid (sky blue from above, warm bounce from below)
  sun     sun visibility at the finest grid (walking out of a doorway into the sun)

lightgrid.bin is gzip-compressed (the client inflates it with DecompressionStream). Its planes are uint8
grids, x fastest, in manifest order: 6 luma planes, 12 chroma planes, 1 sun plane. Direction order is
+X, -X, +Y, -Y, +Z, -Z. Luma byte b > 0 decodes to 2^((b - LUMA_BIAS) / LUMA_STEPS); 0 is black. Chroma
stores r / (r + g + b) and g / (r + g + b) per direction as byte / 255. Sun stores visibility as byte / 255.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import numpy as np

from .probes import ProbeVolumes

LUMA_CELL = (64.0, 64.0, 48.0)
CHROMA_CELL = (128.0, 128.0, 96.0)
SUN_CELL = (32.0, 32.0, 32.0)
LUMA_STEPS = 16.0
LUMA_BIAS = 160.0
LUMINANCE = np.array([0.2125, 0.7154, 0.0721])
DIRECTIONS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)


def _grid(world_min: np.ndarray, world_max: np.ndarray, cell: tuple[float, float, float]) -> tuple[list[int], np.ndarray]:
    size = np.array(cell)
    dims = np.maximum(np.ceil((world_max - world_min) / size).astype(int), 1)
    axes = [world_min[i] + (np.arange(dims[i]) + 0.5) * size[i] for i in range(3)]
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    return [int(d) for d in dims], np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def _ambient_cube(volumes: ProbeVolumes, positions: np.ndarray) -> np.ndarray:
    """(n, 6, 3) irradiance per direction."""
    return np.stack(
        [volumes.sample(positions, np.tile(direction, (len(positions), 1)))[0] for direction in DIRECTIONS], axis=1
    )


def encode_luma(luma: np.ndarray) -> np.ndarray:
    encoded = np.round(np.log2(np.maximum(luma, 1e-9)) * LUMA_STEPS + LUMA_BIAS)
    return np.where(luma > 2.0 ** ((0.5 - LUMA_BIAS) / LUMA_STEPS), np.clip(encoded, 1, 255), 0).astype(np.uint8)


def write(volumes: ProbeVolumes, world_min: np.ndarray, world_max: np.ndarray, path: Path) -> dict[str, Any] | None:
    if not volumes.available or not volumes.volumes:
        return None
    world_min = np.asarray(world_min, dtype=np.float64)
    world_max = np.asarray(world_max, dtype=np.float64)

    luma_dims, positions = _grid(world_min, world_max, LUMA_CELL)
    luma = encode_luma(_ambient_cube(volumes, positions) @ LUMINANCE)  # (n, 6)

    chroma_dims, positions = _grid(world_min, world_max, CHROMA_CELL)
    cube = _ambient_cube(volumes, positions)
    total = cube.sum(axis=2, keepdims=True)
    neutral = np.full_like(cube[..., :2], 1.0 / 3.0)
    chroma = np.where(total > 1e-6, cube[..., :2] / np.maximum(total, 1e-6), neutral)
    chroma = np.clip(np.round(chroma * 255.0), 0, 255).astype(np.uint8).reshape(len(positions), 12)

    sun_dims, positions = _grid(world_min, world_max, SUN_CELL)
    _, visibility = volumes.sample(positions, np.tile(DIRECTIONS[4], (len(positions), 1)))
    sun = np.clip(np.round(visibility * 255.0), 0, 255).astype(np.uint8)

    planes = np.concatenate([luma.T.ravel(), chroma.T.ravel(), sun])
    path.write_bytes(gzip.compress(planes.tobytes(), compresslevel=9, mtime=0))
    origin = [round(float(x), 2) for x in world_min]
    return {
        "file": path.name,
        "encoding": "gzip",
        "origin": origin,
        "luma": {"cell": list(LUMA_CELL), "dims": luma_dims, "steps": LUMA_STEPS, "bias": LUMA_BIAS},
        "chroma": {"cell": list(CHROMA_CELL), "dims": chroma_dims},
        "sun": {"cell": list(SUN_CELL), "dims": sun_dims},
        "bytes": path.stat().st_size,
    }
