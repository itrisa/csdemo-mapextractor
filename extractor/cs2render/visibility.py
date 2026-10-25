"""CS2's baked voxel visibility (maps/<map>/world_visibility.vvis_c) reduced to a compact runtime PVS.

The vvis resource holds a sparse octree of ~32-inch voxel leaves. Each leaf lists regions; a region names a
visibility cluster and a 4x4x4 spatial mask inside the leaf. VisBlocks hold one PVS bitset row per cluster
(little-endian bits over clusters). See ValveResourceFormat Resource/Blocks/VoxelVisibility.cs.

The render package keeps CS2's octree (so the client can find the camera's cluster) and, instead of the
cluster-to-cluster PVS, a cluster-to-cell table: geometry is cut into small 3D cells, every cell is tagged with
the clusters its surfaces touch (sampled just off the surface), and a cell is visible from a cluster when any of
its clusters is in that cluster's PVS.

visibility.bin layout (little-endian, offsets in the manifest):
  nodes     u32  leaf (bit 0) | region count (bits 1-8) | child base or first region (bits 9-31)
  regions   u32  cluster (bits 0-14) | spatial mask index (bits 15-31)
  masks     u64  4x4x4 occupancy inside the leaf, bit = x + 4y + 16z
  rowIndex  u16  per cluster: index of its row in the deduplicated cell-visibility rows
  rows      u8   rowBytes per row, little-endian bits over cells
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .. import source2viewer

CELL_SIZE = (256.0, 256.0, 128.0)
SURFACE_OFFSET = 6.0  # samples sit this far off the surface, on both sides, so they land in empty voxels
RETRY_OFFSETS = (16.0, 32.0, 64.0)
SAMPLE_AREA = 48.0 * 48.0  # one extra interior sample per this much triangle area
MAX_EXTRA_SAMPLES = 32
BATCH_TRIANGLES = 400_000

_SUB1 = np.array([0, 2, 8, 10, 32, 34, 40, 42], dtype=np.uint64)
_SUB2 = np.array([0, 1, 4, 5, 16, 17, 20, 21], dtype=np.uint64)
_DIRECTIONS = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float64)


@dataclass
class VoxelVisibility:
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    clusters: int
    nodes: np.ndarray  # (N, 2) uint32
    regions: np.ndarray  # uint64
    masks: np.ndarray  # uint64
    vis: np.ndarray  # (rows, bytes) uint8

    @classmethod
    def parse(cls, raw: bytes, data_text: str) -> "VoxelVisibility":
        block_offset = int.from_bytes(raw[8:12], "little") + 8
        block_count = int.from_bytes(raw[12:16], "little")
        blocks: dict[str, int] = {}
        for i in range(block_count):
            entry = block_offset + i * 12
            name = raw[entry : entry + 4].decode("ascii", "replace")
            blocks[name] = entry + 4 + int.from_bytes(raw[entry + 4 : entry + 8], "little")
        if "VXVS" not in blocks:
            raise ValueError("vvis resource has no VXVS block")
        base = blocks["VXVS"]

        def number(key: str) -> float:
            return float(_search(rf"{key}\s*=\s*([-\d.eE+]+)", data_text).group(1))

        def vector(key: str) -> np.ndarray:
            return np.array([float(x) for x in _search(rf"{key}\s*=\s*\[([^\]]*)\]", data_text).group(1).split(",")])

        def block(key: str, dtype: Any, width: int = 1) -> np.ndarray:
            match = _search(rf"{key}\s*=\s*\{{\s*m_nOffset\s*=\s*(\d+)\s*m_nElementCount\s*=\s*(\d+)", data_text)
            offset, count = int(match.group(1)), int(match.group(2))
            return np.frombuffer(raw, dtype=dtype, count=count * width, offset=base + offset)

        clusters = int(number("m_nBaseClusterCount"))
        row_bytes = int(number("m_nPVSBytesPerCluster"))
        vis = block("m_nVisBlocks", np.uint8)
        return cls(
            bounds_min=vector("m_vMinBounds"),
            bounds_max=vector("m_vMaxBounds"),
            clusters=clusters,
            nodes=block("m_NodeBlock", np.uint32, 2).reshape(-1, 2),
            regions=block("m_RegionBlock", np.uint64),
            masks=block("m_MasksBlock", np.uint64),
            vis=vis[: len(vis) // row_bytes * row_bytes].reshape(-1, row_bytes),
        )

    def cluster_pairs(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(point index, cluster) for every region covering each point; like the engine's GetPVSForPoint, a point on a
        region boundary belongs to all of them. Points in solid or outside the octree yield no pair."""
        empty = np.zeros(0, dtype=np.int64)
        inside = np.nonzero(np.all((points >= self.bounds_min) & (points <= self.bounds_max), axis=1))[0]
        if len(inside) == 0:
            return empty, empty
        p = points[inside]
        lo = np.broadcast_to(self.bounds_min, p.shape).copy()
        hi = np.broadcast_to(self.bounds_max, p.shape).copy()
        leaf = (self.nodes[:, 0] & 1) != 0
        offset = (self.nodes[:, 0] >> 1).astype(np.int64)
        node = np.zeros(len(p), dtype=np.int64)
        active = np.nonzero(~leaf[node])[0]
        while len(active):
            o, lo[active], hi[active] = _octant(p[active], lo[active], hi[active])
            node[active] = offset[node[active]] + o
            active = active[~leaf[node[active]]]
        o1, lo, hi = _octant(p, lo, hi)
        o2, _, _ = _octant(p, lo, hi)
        spatial = np.uint64(1) << (_SUB1[o1] + _SUB2[o2])
        region_count = (self.nodes[node, 1] & 0xFF).astype(np.int64)
        region_cluster = (self.regions & np.uint64(0x7FFF)).astype(np.int64)
        region_mask = (self.regions >> np.uint64(40)).astype(np.int64)
        index, cluster = [empty], [empty]
        for r in range(int(region_count.max(initial=0))):
            sel = np.nonzero(region_count > r)[0]
            region = offset[node[sel]] + r
            hit = ((self.masks[region_mask[region]] & spatial[sel]) != 0) & (region_cluster[region] < self.clusters)
            index.append(inside[sel[hit]])
            cluster.append(region_cluster[region[hit]])
        return np.concatenate(index), np.concatenate(cluster)

    def clusters_for(self, points: np.ndarray) -> np.ndarray:
        """One cluster covering each point, or -1 (solid or outside the octree)."""
        out = np.full(len(points), -1, dtype=np.int64)
        index, cluster = self.cluster_pairs(points)
        out[index[::-1]] = cluster[::-1]
        return out

    def pvs(self) -> np.ndarray:
        """(clusters, clusters) bool: row c lists the clusters potentially visible from cluster c."""
        return np.unpackbits(self.vis[: self.clusters], axis=1, bitorder="little")[:, : self.clusters].astype(bool)


def _search(pattern: str, text: str) -> re.Match[str]:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"vvis DATA block has no match for {pattern!r}")
    return match


def _octant(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = (lo + hi) * 0.5
    upper = mid < p
    o = upper[:, 0] * 1 + upper[:, 1] * 2 + upper[:, 2] * 4
    return o, np.where(upper, mid, lo), np.where(upper, hi, mid)


def load(executable: Path, cs2_dir: Path, map_name: str, work_dir: Path) -> VoxelVisibility | None:
    """The map's voxel visibility, cached in work_dir; None when the map ships none."""
    resource = f"maps/{map_name}/world_visibility.vvis_c"
    raw_path = work_dir / "world_visibility.vvis_c"
    text_path = work_dir / "world_visibility.kv3"
    if not raw_path.is_file() or not text_path.is_file():
        vpk = (cs2_dir / "game" / "csgo" / "maps" / f"{map_name}.vpk").resolve()
        try:
            raw = source2viewer._extract_resource(executable, vpk, resource, Path(resource), decompile=False)
            text = source2viewer.extract_resource_block(executable, vpk, resource)
        except Exception as error:  # noqa: BLE001 - visibility is an optimisation; render everything without it
            print(f"[cs2render] no voxel visibility for {map_name}: {error}", flush=True)
            return None
        raw_path.write_bytes(raw)
        text_path.write_text(text, encoding="utf-8")
    return VoxelVisibility.parse(raw_path.read_bytes(), text_path.read_text(encoding="utf-8"))


class CellTable:
    """Compact, Morton-ordered numbering of the occupied visibility cells (neighbours get nearby indices)."""

    def __init__(self, origin: np.ndarray, size: tuple[float, float, float] = CELL_SIZE) -> None:
        self.origin = np.asarray(origin, dtype=np.float64) - 1.0
        self.size = np.asarray(size, dtype=np.float64)
        self._keys: list[np.ndarray] = []
        self.sorted_keys = np.zeros(0, dtype=np.int64)
        self.rank = np.zeros(0, dtype=np.int64)

    def keys(self, points: np.ndarray) -> np.ndarray:
        cell = np.clip(np.floor((points - self.origin) / self.size), 0, (1 << 20) - 1).astype(np.int64)
        return cell[:, 0] | (cell[:, 1] << 20) | (cell[:, 2] << 40)

    def add(self, points: np.ndarray) -> None:
        self._keys.append(np.unique(self.keys(points)))

    def finish(self) -> None:
        self.sorted_keys = np.unique(np.concatenate(self._keys)) if self._keys else np.zeros(0, dtype=np.int64)
        self._keys = []
        morton = np.zeros(len(self.sorted_keys), dtype=np.int64)
        for axis in range(3):
            coordinate = (self.sorted_keys >> (20 * axis)) & 0xFFFFF
            for bit in range(20):
                morton |= ((coordinate >> bit) & 1) << (3 * bit + axis)
        self.rank = np.empty(len(self.sorted_keys), dtype=np.int64)
        self.rank[np.argsort(morton, kind="stable")] = np.arange(len(self.sorted_keys))

    def __len__(self) -> int:
        return len(self.sorted_keys)

    def index(self, points: np.ndarray) -> np.ndarray:
        keys = self.keys(points)
        slot = np.clip(np.searchsorted(self.sorted_keys, keys), 0, max(len(self.sorted_keys) - 1, 0))
        if len(keys) and not np.array_equal(self.sorted_keys[slot], keys):
            raise ValueError("point outside the registered visibility cells")
        return self.rank[slot]


@dataclass
class CellVisibility:
    cell_clusters: np.ndarray  # unique int64 codes cell * clusters + cluster
    always_visible: np.ndarray  # bool per cell (surfaces that touch no cluster)
    triangles: int = 0
    unassigned: int = 0


def sample_cells(
    vis: VoxelVisibility, table: CellTable, triangle_batches: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> CellVisibility:
    """Tags every cell with the clusters its triangles touch. Batches are (a, b, c) corner arrays, float64 (n, 3)."""
    codes: list[np.ndarray] = []
    always = np.zeros(len(table), dtype=bool)
    rng = np.random.default_rng(0)
    total = unassigned = 0
    for a, b, c in triangle_batches:
        if len(a) == 0:
            continue
        cells = table.index((a + b + c) / 3.0)
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal, axis=1, keepdims=True)
        area = length[:, 0] * 0.5
        normal = normal / np.maximum(length, 1e-9)
        owners, points = _surface_samples(a, b, c, normal, area, rng)
        hit_owner, hit_cluster = _owned_pairs(vis, owners, points)
        for extra in RETRY_OFFSETS:
            has = np.zeros(len(a), dtype=bool)
            has[hit_owner] = True
            missing = np.nonzero(~has)[0]
            if len(missing) == 0:
                break
            retry_owner, retry_cluster = _owned_pairs(vis, *_retry_samples(missing, a, b, c, normal, extra))
            hit_owner = np.concatenate([hit_owner, retry_owner])
            hit_cluster = np.concatenate([hit_cluster, retry_cluster])
        codes.append(np.unique(cells[hit_owner] * vis.clusters + hit_cluster))
        has = np.zeros(len(a), dtype=bool)
        has[hit_owner] = True
        always[cells[~has]] = True
        total += len(a)
        unassigned += int((~has).sum())
    merged = np.unique(np.concatenate(codes)) if codes else np.zeros(0, dtype=np.int64)
    return CellVisibility(merged, always, total, unassigned)


def _owned_pairs(vis: VoxelVisibility, owners: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index, cluster = vis.cluster_pairs(points)
    return owners[index], cluster


def _surface_samples(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, normal: np.ndarray, area: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(len(a))
    owners, points = [], []
    for base in ((a + b + c) / 3.0, a, b, c):
        for sign in (1.0, -1.0):
            owners.append(index)
            points.append(base + sign * SURFACE_OFFSET * normal)
    extra = np.minimum(np.floor(area / SAMPLE_AREA), MAX_EXTRA_SAMPLES).astype(np.int64)
    big = np.repeat(index, extra)
    if len(big):
        u, v = rng.random(len(big)), rng.random(len(big))
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        inner = a[big] + u[:, None] * (b[big] - a[big]) + v[:, None] * (c[big] - a[big])
        for sign in (1.0, -1.0):
            owners.append(big)
            points.append(inner + sign * SURFACE_OFFSET * normal[big])
    return np.concatenate(owners), np.concatenate(points)


def _retry_samples(
    missing: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, normal: np.ndarray, extra: float
) -> tuple[np.ndarray, np.ndarray]:
    owners, points = [], []
    for base in ((a[missing] + b[missing] + c[missing]) / 3.0, a[missing], b[missing], c[missing]):
        for direction in _DIRECTIONS:
            owners.append(missing)
            points.append(base + extra * direction)
        for sign in (1.0, -1.0):
            owners.append(missing)
            points.append(base + sign * extra * normal[missing])
    return np.concatenate(owners), np.concatenate(points)


def write(vis: VoxelVisibility, table: CellTable, cells: CellVisibility, path: Path) -> dict[str, Any]:
    """Writes visibility.bin and returns its manifest entry."""
    clusters = vis.clusters
    membership = np.zeros((len(table), clusters), dtype=np.float32)
    membership[cells.cell_clusters // clusters, cells.cell_clusters % clusters] = 1.0
    visible = (vis.pvs().astype(np.float32) @ membership.T) > 0
    visible |= cells.always_visible[None, :]
    packed = np.packbits(visible, axis=1, bitorder="little")
    rows, row_index = np.unique(packed, axis=0, return_inverse=True)
    if len(rows) > 0xFFFF:
        raise ValueError(f"{len(rows)} distinct visibility rows do not fit u16 indices")

    first = vis.nodes[:, 0]
    region_count = vis.nodes[:, 1] & 0xFF
    child = first >> 1
    if child.max(initial=0) >= 1 << 23:
        raise ValueError("vvis octree too large for 23-bit node offsets")
    nodes = (first & 1) | (region_count << 1) | (child << 9)
    cluster = vis.regions & np.uint64(0x7FFF)
    mask = vis.regions >> np.uint64(40)
    if mask.max(initial=0) >= 1 << 17:
        raise ValueError("vvis spatial mask index exceeds 17 bits")
    regions = (cluster | (mask << np.uint64(15))).astype(np.uint32)

    parts: list[tuple[str, bytes, int]] = [
        ("nodes", nodes.astype("<u4").tobytes(), len(nodes)),
        ("regions", regions.astype("<u4").tobytes(), len(regions)),
        ("masks", vis.masks.astype("<u8").tobytes(), len(vis.masks)),
        ("rowIndex", row_index.reshape(-1).astype("<u2").tobytes(), clusters),
        ("rows", rows.tobytes(), len(rows)),
    ]
    entry: dict[str, Any] = {
        "file": path.name,
        "cellSize": list(table.size),
        "cells": len(table),
        "clusters": clusters,
        "boundsMin": [float(x) for x in vis.bounds_min],
        "boundsMax": [float(x) for x in vis.bounds_max],
        "rowBytes": int(packed.shape[1]),
    }
    offset = 0
    with path.open("wb") as handle:
        for name, data, count in parts:
            handle.write(data)
            entry[name] = [offset, count]
            offset += len(data)
            pad = -offset % 8
            handle.write(b"\0" * pad)
            offset += pad
    return entry
