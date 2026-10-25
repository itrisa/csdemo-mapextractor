"""Footstep surfaces of a map: which surface property the floor under each spot has.

The world physics (``world_physics.vmdl_c``, PHYS block printed by Source2Viewer as KV3 text) holds the collision
meshes, with one surface-property index per triangle (`m_Materials`), and convex hulls with one per hull. The
glTF physics export keeps only one surface per mesh, which is why the text dump is read. Up-facing triangles of
player-solid shapes are sampled into a grid of XY cells; each cell keeps its floor layers (height and surface)
so stacked floors (a balcony above a street) both resolve.
"""

from __future__ import annotations

import gzip
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

WALKABLE_NORMAL_Z = 0.7
CELL = 32.0
SAMPLE_SPACING = 12.0
LAYER_GAP = 24.0
# Collision attributes that are only clips (players stand on the real floor's sound below them) or never
# reached by players.
IGNORED_INTERACT_AS = {"playerclip", "npcclip", "csgo_grenadeclip", "sky"}

_ASSIGN = re.compile(r"^(m_\w+) = ?(.*)$")
_STRING = re.compile(r'"([^"]*)"')


@dataclass
class Shape:
    attribute: int
    surface: int
    kind: str = ""
    blobs: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Physics:
    floors: list[tuple[int, np.ndarray, np.ndarray]]  # (attribute, triangles (n, 3, 3), surface indices (n,))
    surface_hashes: list[int]
    attributes: list[dict[str, object]]


_WANTED = {
    "mesh": {"m_Materials", "m_Triangles", "m_Vertices"},
    "hull": {"m_VertexPositions", "m_Edges", "m_Faces"},
}


def _floors_of(shape: Shape) -> tuple[np.ndarray, np.ndarray] | None:
    if shape.kind == "mesh":
        vertices = np.frombuffer(bytes.fromhex(" ".join(shape.blobs.get("m_Vertices", []))), "<f4").reshape(-1, 3)
        indices = np.frombuffer(bytes.fromhex(" ".join(shape.blobs.get("m_Triangles", []))), "<i4").reshape(-1, 3)
        if not len(indices):
            return None
        triangles = vertices[indices]
        materials_text = " ".join(shape.blobs.get("m_Materials", []))
        materials = np.array([int(value) for value in re.findall(r"-?\d+", materials_text)], dtype=np.int32)
        if len(materials) != len(indices):
            materials = np.full(len(indices), shape.surface, dtype=np.int32)
    elif shape.kind == "hull":
        positions = np.frombuffer(bytes.fromhex(" ".join(shape.blobs.get("m_VertexPositions", []))), "<f4").reshape(-1, 3)
        edges = np.frombuffer(bytes.fromhex(" ".join(shape.blobs.get("m_Edges", []))), np.uint8).reshape(-1, 4)
        faces = np.frombuffer(bytes.fromhex(" ".join(shape.blobs.get("m_Faces", []))), np.uint8)
        polygons: list[np.ndarray] = []
        for start in faces:
            loop, edge = [], int(start)
            for _ in range(len(edges)):
                loop.append(int(edges[edge][2]))
                edge = int(edges[edge][0])
                if edge == start:
                    break
            for i in range(1, len(loop) - 1):
                polygons.append(positions[[loop[0], loop[i], loop[i + 1]]])
        if not polygons:
            return None
        triangles = np.stack(polygons)
        materials = np.full(len(triangles), shape.surface, dtype=np.int32)
    else:
        return None
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    keep = (lengths > 1e-6) & (normals[:, 2] >= WALKABLE_NORMAL_Z * np.maximum(lengths, 1e-6))
    if not keep.any():
        return None
    return triangles[keep].astype(np.float32), materials[keep]


def parse(lines: Iterable[str]) -> Physics:
    """Streams a Source2Viewer PHYS text dump (hundreds of MB for a large map) and keeps only the up-facing
    triangles, the surface-property hashes and the collision attributes."""
    floors: list[tuple[int, np.ndarray, np.ndarray]] = []
    hashes: list[int] = []
    attributes: list[dict[str, object]] = []
    shape: Shape | None = None
    collecting: list[str] | None = None
    pending: str | None = None
    section = ""
    attribute: dict[str, object] | None = None
    strings_key: str | None = None

    def flush() -> None:
        nonlocal shape
        if shape is not None:
            result = _floors_of(shape)
            if result is not None:
                floors.append((shape.attribute, *result))
        shape = None

    for raw in lines:
        line = raw.strip()
        if collecting is not None:
            if line in ("]", "],"):
                collecting = None
            else:
                collecting.append(line.rstrip(","))
            continue
        if pending is not None:
            key, pending = pending, None
            if line in ("[", "#["):
                collecting = _sink(key, shape, hashes)
                continue
        if strings_key is not None and attribute is not None:
            if line in ("]", "],"):
                strings_key = None
            elif line != "[":
                attribute.setdefault(strings_key, []).extend(_STRING.findall(line))  # type: ignore[union-attr]
            continue
        match = _ASSIGN.match(line)
        if not match:
            if section == "attributes" and line == "{":
                attribute = {}
                attributes.append(attribute)
            continue
        key, value = match.groups()
        if key == "m_nCollisionAttributeIndex":
            flush()
            shape = Shape(attribute=int(value), surface=0)
        elif key == "m_nSurfacePropertyIndex" and shape is not None:
            shape.surface = int(value)
        elif key in ("m_Hull", "m_Mesh") and shape is not None and not shape.kind:
            shape.kind = "hull" if key == "m_Hull" else "mesh"
        elif key == "m_surfacePropertyHashes":
            flush()
            section = "hashes"
            if value.startswith("["):
                hashes.extend(int(v) for v in re.findall(r"\d+", value))
            else:
                pending = key
        elif key == "m_collisionAttributes":
            flush()
            section = "attributes"
        elif section == "attributes" and attribute is not None and key in (
            "m_CollisionGroupString", "m_InteractAsStrings", "m_InteractExcludeStrings"
        ):
            if value.startswith('"'):
                attribute[key] = _STRING.findall(value)[0]
            elif value.startswith("["):
                attribute[key] = _STRING.findall(value)
            else:
                strings_key = key
        elif value in ("", "#[", "["):
            if value == "":
                pending = key
            else:
                collecting = _sink(key, shape, hashes)
        elif value.startswith("#[") and shape is not None and shape.kind and key in _WANTED[shape.kind]:
            shape.blobs[key] = [value[2:].rstrip("]").strip()]
    flush()
    return Physics(floors, hashes, attributes)


def _sink(key: str, shape: Shape | None, hashes: list[int]) -> list[str] | None:
    """Where the lines of a wanted multi-line list go; None leaves unwanted lists (and lists of objects) to the
    normal line handling, which ignores their numeric and hex lines."""
    if key == "m_surfacePropertyHashes":
        return _HashSink(hashes)
    if shape is not None and shape.kind and key in _WANTED[shape.kind]:
        return shape.blobs.setdefault(key, [])
    return None


class _HashSink(list):
    def __init__(self, target: list[int]) -> None:
        super().__init__()
        self.target = target

    def append(self, item: str) -> None:
        self.target.extend(int(v) for v in re.findall(r"\d+", item))


def read_lines(path: Path) -> Iterator[str]:
    with path.open("rb") as probe:
        bom = probe.read(2)
    encoding = "utf-16" if bom in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    with path.open("r", encoding=encoding, errors="replace") as handle:
        yield from handle


def walkable_attributes(attributes: list[dict[str, object]]) -> set[int]:
    walkable = set()
    for index, attribute in enumerate(attributes):
        excluded = {str(s).lower() for s in attribute.get("m_InteractExcludeStrings", []) or []}  # type: ignore[union-attr]
        interact_as = {str(s).lower() for s in attribute.get("m_InteractAsStrings", []) or []}  # type: ignore[union-attr]
        if "player" in excluded or (interact_as and interact_as <= IGNORED_INTERACT_AS):
            continue
        walkable.add(index)
    return walkable


def sample(triangles: np.ndarray, spacing: float = SAMPLE_SPACING) -> tuple[np.ndarray, np.ndarray]:
    """Points spread over each triangle (vertices, centroid and a barycentric lattice for large ones) and the
    index of the triangle each came from."""
    points, owners = [], []
    edge = np.max(np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2), axis=1)
    steps = np.clip(np.ceil(edge / spacing).astype(np.int64), 1, 256)
    for n in np.unique(steps):
        chosen = np.nonzero(steps == n)[0]
        weights = [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]
        weights.append((1 / 3, 1 / 3, 1 / 3))
        w = np.array(weights, dtype=np.float32)  # (k, 3)
        group = triangles[chosen]  # (m, 3, 3)
        lattice = np.einsum("kv,mvc->mkc", w, group).reshape(-1, 3)
        points.append(lattice)
        owners.append(np.repeat(chosen, len(w)))
    return np.concatenate(points), np.concatenate(owners)


@dataclass
class Grid:
    x0: float
    y0: float
    width: int
    height: int
    counts: np.ndarray  # (height * width,) uint8 layers per cell, row-major from (x0, y0)
    z: np.ndarray  # (layers,) int16, per cell ascending
    surface: np.ndarray  # (layers,) uint8 index into the surface name list

    def encode(self) -> bytes:
        header = struct.pack("<4sffffHHI", b"SGR1", self.x0, self.y0, CELL, 0.0, self.width, self.height, len(self.z))
        return gzip.compress(header + self.counts.tobytes() + self.z.astype("<i2").tobytes() + self.surface.tobytes(), 9)

    def lookup(self, x: float, y: float, z: float) -> int | None:
        cx, cy = int((x - self.x0) // CELL), int((y - self.y0) // CELL)
        if not (0 <= cx < self.width and 0 <= cy < self.height):
            return None
        cell = cy * self.width + cx
        start = int(self.counts[:cell].sum())
        count = int(self.counts[cell])
        if not count:
            return None
        best = None
        for layer in range(start, start + count):
            if self.z[layer] <= z + 20:
                best = layer
        return int(self.surface[best if best is not None else start])


def bake_grid(points: np.ndarray, surfaces: np.ndarray) -> Grid:
    """Cells of CELL units; per cell, sample heights closer than LAYER_GAP merge into one layer whose height is
    the highest sample and whose surface is that of the highest sample."""
    x0 = float(np.floor(points[:, 0].min() / CELL) * CELL)
    y0 = float(np.floor(points[:, 1].min() / CELL) * CELL)
    width = int((points[:, 0].max() - x0) // CELL) + 1
    height = int((points[:, 1].max() - y0) // CELL) + 1
    cells = ((points[:, 1] - y0) // CELL).astype(np.int64) * width + ((points[:, 0] - x0) // CELL).astype(np.int64)
    order = np.lexsort((points[:, 2], cells))
    cells, zs, kinds = cells[order], points[order, 2], surfaces[order]
    new_cell = np.r_[True, cells[1:] != cells[:-1]]
    new_layer = new_cell | np.r_[True, np.diff(zs) > LAYER_GAP]
    last = np.r_[np.nonzero(new_layer)[0][1:] - 1, len(zs) - 1]  # top sample of each layer
    layer_cells = cells[last]
    counts = np.bincount(layer_cells, minlength=width * height)
    if counts.max() > 255:
        raise RuntimeError("too many floor layers in one cell")
    return Grid(
        x0, y0, width, height,
        counts.astype(np.uint8),
        np.clip(np.round(zs[last]), -32768, 32767).astype(np.int16),
        kinds[last].astype(np.uint8),
    )


def surface_names(vsurf: dict) -> tuple[dict[int, str], dict[str, str]]:
    """Name hash → surface name, and lower-case name → base name, from surfaceproperties.vsurf."""
    by_hash, bases = {}, {}
    for entry in vsurf.get("SurfacePropertiesList", []):
        name = str(entry.get("surfacePropertyName", ""))
        if "m_nameHash" in entry:
            by_hash[int(entry["m_nameHash"])] = name
        if entry.get("base"):
            bases[name.lower()] = str(entry["base"])
    return by_hash, bases


def footstep_table(footsteps: dict) -> dict[str, dict[str, str]]:
    """Lower-case surface name → {'ct': event, 't': event} from scripts/surfaceproperties_footsteps.txt."""
    table: dict[str, dict[str, str]] = {}
    for side, key in (("ct", "ct_player"), ("t", "t_player")):
        for entry in footsteps.get(key, {}).get("SurfacePropertiesList", []):
            name = str(entry.get("surfacePropertyName", "")).lower()
            walk = entry.get("walkleft") or entry.get("walkright")
            if name and walk:
                table.setdefault(name, {})[side] = str(walk)
    return table


def footstep_surface(name: str, table: dict[str, dict[str, str]], bases: dict[str, str]) -> str:
    """The surface whose footsteps a surface uses: itself if listed, else the nearest listed base, else default."""
    current, seen = name.lower(), set()
    while current and current not in seen:
        if current in table and "ct" in table[current] and "t" in table[current]:
            return current
        seen.add(current)
        current = bases.get(current, "").lower()
    return "default"
