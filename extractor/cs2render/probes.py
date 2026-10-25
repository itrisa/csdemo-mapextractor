"""Bakes CS2 light probe volumes (ambient cube atlases) into per-vertex lighting for non-lightmapped meshes."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .lighting import _angles_matrix, _vec, read_exr


def _by_slice_number(paths: Iterable[Path]) -> list[Path]:
    """Atlas slices are named z000..z1607; a plain string sort would put z1000 before z101."""
    return sorted(paths, key=lambda path: int(re.search(r"_z(\d+)$", path.stem).group(1)))


@dataclass
class ProbeVolume:
    world_to_local: np.ndarray  # 3x4
    mins: np.ndarray
    size: np.ndarray
    atlas_size: np.ndarray  # probe counts per axis
    atlas_offset: np.ndarray  # texel offset in the atlas
    volume: float


class ProbeVolumes:
    def __init__(self, lightmap_dir: Path, entities: list[dict[str, Any]]) -> None:
        slices = _by_slice_number(lightmap_dir.glob("env_light_probe_volume_atlas_z*.exr"))
        shadow_slices = _by_slice_number(lightmap_dir.glob("env_light_probe_volume_atlas_dlshd_z*.png"))
        self.available = bool(slices) and bool(shadow_slices)
        self.volumes: list[ProbeVolume] = []
        if not self.available:
            return
        self.irradiance = np.stack([read_exr(path)[..., :3] for path in slices])  # (6 * D, H, W, 3)
        self.shadows = np.stack(
            [np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)[..., 0] for path in shadow_slices]
        ).astype(np.float32) / 255.0  # (D, H, W): sun channel
        self.depth = self.shadows.shape[0]
        if self.irradiance.shape[0] != self.depth * 6:
            raise RuntimeError("light probe irradiance atlas is not six direction blocks of the shadow atlas depth")
        for entity in entities:
            if "light_probe_atlas_x" not in entity:
                continue
            rotation = _angles_matrix(tuple(_vec(entity.get("angles"))))
            origin = _vec(entity.get("origin"))
            mins = _vec(entity.get("box_mins"))
            maxs = _vec(entity.get("box_maxs"))
            world_to_local = np.concatenate([rotation.T, (-rotation.T @ origin)[:, None]], axis=1)
            size = np.maximum(maxs - mins, 1e-3)
            self.volumes.append(
                ProbeVolume(
                    world_to_local=world_to_local,
                    mins=mins,
                    size=size,
                    atlas_size=np.array(
                        [float(entity[f"light_probe_size_{axis}"]) for axis in ("x", "y", "z")], dtype=np.float64
                    ),
                    atlas_offset=np.array(
                        [
                            float(entity.get("light_probe_atlas_x")),
                            float(entity.get("light_probe_atlas_y", entity.get("light_probe_atlas_Y"))),
                            float(entity.get("light_probe_atlas_z", entity.get("light_probe_atlas_Z"))),
                        ],
                        dtype=np.float64,
                    ),
                    volume=float(np.prod(size)),
                )
            )
        # Smallest volumes win, matching the tightest-fitting probe the engine would pick for an object.
        self.volumes.sort(key=lambda v: v.volume)

    def _assign(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(positions)
        chosen = np.full(n, -1, dtype=np.int64)
        local_coords = np.zeros((n, 3))
        best_distance = np.full(n, np.inf)
        for index, volume in enumerate(self.volumes):
            local = positions @ volume.world_to_local[:, :3].T + volume.world_to_local[:, 3]
            normalized = (local - volume.mins) / volume.size
            outside = np.maximum(-normalized, 0) + np.maximum(normalized - 1, 0)
            distance = np.linalg.norm(outside * volume.size, axis=1)
            better = distance < best_distance
            chosen[better] = index
            best_distance[better] = distance[better]
            local_coords[better] = np.clip(normalized[better], 0.0, 1.0)
        return chosen, local_coords

    @staticmethod
    def _trilinear(atlas: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """coords (n, 3) texel-centre coordinates (x, y, z) into atlas[z, y, x]."""
        d, h, w = atlas.shape[:3]
        base = np.floor(coords).astype(np.int64)
        frac = coords - base
        result = 0.0
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    x = np.clip(base[:, 0] + dx, 0, w - 1)
                    y = np.clip(base[:, 1] + dy, 0, h - 1)
                    z = np.clip(base[:, 2] + dz, 0, d - 1)
                    weight = (
                        (frac[:, 0] if dx else 1 - frac[:, 0])
                        * (frac[:, 1] if dy else 1 - frac[:, 1])
                        * (frac[:, 2] if dz else 1 - frac[:, 2])
                    )
                    sample = atlas[z, y, x]
                    result = result + (sample * (weight[:, None] if sample.ndim > 1 else weight))
        return result

    def sample(self, positions: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (irradiance rgb (n, 3), sun visibility (n,)) per vertex."""
        n = len(positions)
        if not self.available or not self.volumes:
            return np.full((n, 3), 0.5, dtype=np.float32), np.ones(n, dtype=np.float32)
        chosen, local = self._assign(positions)
        irradiance = np.zeros((n, 3))
        visibility = np.ones(n)
        for index in np.unique(chosen):
            if index < 0:
                continue
            volume = self.volumes[index]
            mask = chosen == index
            texel = np.clip(local[mask] * volume.atlas_size, 0.5, volume.atlas_size - 0.5) + volume.atlas_offset - 0.5
            visibility[mask] = 1.0 - self._trilinear(self.shadows, texel)
            normal = normals[mask]
            squared = normal * normal
            total = np.zeros((mask.sum(), 3))
            for axis in range(3):
                block = np.where(normal[:, axis] >= 0, axis, axis + 3)
                for value in (axis, axis + 3):
                    rows = block == value
                    if not rows.any():
                        continue
                    shifted = texel[rows].copy()
                    shifted[:, 2] = np.clip(shifted[:, 2], volume.atlas_offset[2], volume.atlas_offset[2] + volume.atlas_size[2] - 1)
                    shifted[:, 2] += value * self.depth
                    total[rows] += self._trilinear(self.irradiance, shifted) * squared[rows, axis : axis + 1]
            irradiance[mask] = total
        return irradiance.astype(np.float32), np.clip(visibility, 0.0, 1.0).astype(np.float32)


RGBM_RANGE = 16.0


def encode_rgbm(rgb: np.ndarray) -> np.ndarray:
    """Packs linear rgb into (n, 2) uint16: (R | G << 8, B | M << 8), value = rgb8 / 255 * M / 255 * RANGE."""
    peak = np.clip(rgb.max(axis=1) / RGBM_RANGE, 1e-6, 1.0)
    m = np.clip(np.ceil(peak * 255.0), 1, 255)
    scaled = np.clip(np.round(rgb / (m[:, None] / 255.0 * RGBM_RANGE) * 255.0), 0, 255).astype(np.uint16)
    return np.stack([scaled[:, 0] | (scaled[:, 1] << 8), scaled[:, 2] | (m.astype(np.uint16) << 8)], axis=1)
