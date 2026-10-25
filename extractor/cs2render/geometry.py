"""Packs the render world into spatial, render-state-grouped, meshopt-compressed chunks.

Vertex streams (little-endian, meshopt vertex codec v1, decoded on the client with MeshoptDecoder):
  pos  u16x4  quantized position in the chunk box + material index
  nrm  u8x4   octahedral normal xy (value = (byte - 128) / 127), flags (bit1 probe-lit, bits2-7 sun visibility), probe slot
  uv   f32x2  layer-1 texture coordinates (f32x4 with layer-2 coordinates for blend chunks), EXPONENTIAL filter
  lm   u16x2  lightmap UV, or RGBM probe irradiance when probe-lit
  col  u8x4   vertex paint, applied as mix(1, rgb, a); white for materials that do not read COLOR_0
  bld  u8x4   blend values (blend chunks)
  ovl  f32x2  shared colour overlay UV (blend chunks), EXPONENTIAL filter

Tangent frames are not stored: the client rebuilds them per pixel from position and UV derivatives.

The geometry is written with one file per spatial cell. Every chunk may also carry a far LOD: a coarser index
list over the same vertices (absolute error bound in inches, unlocked borders, relaxed attribute weights) that
the client draws instead when the chunk is far enough that the error stays under about a pixel.

Inside a chunk, triangles are grouped by small visibility cell (see visibility.py) into contiguous index ranges,
"parts", so the client can draw only the parts that pass PVS and frustum tests in one multi-draw call. The parts
table follows the index buffer, one record per part:
  u32 first index, u32 index count, u32 visibility cell, u16x6 quantized box (min xyz, max xyz in the chunk box)
A chunk with a far LOD then has its own LOD index buffer and LOD parts table, in the same formats.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import meshoptimizer
import numpy as np

from . import meshopt_native
from .gltf_world import Primitive
from .materials import BLEND, PAINT_NONE, PAINT_OVERLAY, PAINT_SRGB, TRANSLUCENT, MaterialSpec
from .probes import ProbeVolumes, encode_rgbm
from .textures import srgb_to_linear

MAX_VERTICES = 65535
FILTERED_STREAMS = ("uv", "ovl")
UV_TEXEL_FRACTION = 4  # texture coordinates keep 1/4 texel of precision at the class resolution
OVERLAY_TEXELS = 256
LIGHTMAP_TEXELS = 4096.0
SIMPLIFY_WEIGHTS = {"normal": 1.0, "uv": 16.0, "lm": 0.25 * LIGHTMAP_TEXELS, "bld": 2.0, "col": 1.0}
LOD_MAX_SHARE = 0.8  # a chunk stores a far LOD only when it drops at least a fifth of the triangles
PART_DTYPE = np.dtype([("start", "<u4"), ("count", "<u4"), ("cell", "<u4"), ("box", "<u2", (6,))])

CellIndex = Callable[[np.ndarray], np.ndarray]

meshoptimizer.encode_vertex_version(1)
meshoptimizer.encode_index_version(1)


def oct_encode(vectors: np.ndarray) -> np.ndarray:
    v = vectors / np.maximum(np.abs(vectors).sum(axis=1, keepdims=True), 1e-12)
    xy = v[:, :2].copy()
    negative = v[:, 2] < 0
    folded = (1.0 - np.abs(xy[:, ::-1])) * np.where(xy >= 0, 1.0, -1.0)
    xy[negative] = folded[negative]
    return np.clip(xy, -1.0, 1.0)


def _unorm8_oct(values: np.ndarray) -> np.ndarray:
    return (np.clip(np.round(values * 127), -127, 127) + 128).astype(np.uint8)


@dataclass
class Piece:
    """A primitive (or part of one) ready for chunking: per-vertex arrays in final encodings."""

    material: int
    key: tuple
    cell: tuple[int, int]
    positions: np.ndarray  # float64
    streams: dict[str, np.ndarray]
    indices: np.ndarray  # uint32 local
    lod: np.ndarray | None = None  # far LOD indices over the same vertices; None when it would not reduce
    lod_done: bool = False

    def lod_indices(self) -> np.ndarray:
        return self.indices if self.lod is None else self.lod


@dataclass
class ChunkRecord:
    key: tuple
    cell: tuple[int, int]
    file: str
    vertex_count: int
    index_count: int
    box_min: list[float]
    box_max: list[float]
    streams: dict[str, list[Any]] = field(default_factory=dict)  # name -> [offset, length, stride, filter]
    index: list[int] = field(default_factory=list)  # [offset, length]
    parts: list[int] = field(default_factory=list)  # [offset, count]
    lod: dict[str, Any] | None = None  # {"index": [offset, length], "indexCount": n, "parts": [offset, count]}


def _split_by_vertex_budget(indices: np.ndarray, budget: int) -> list[np.ndarray]:
    triangles = indices.reshape(-1, 3)
    if len(np.unique(triangles)) <= budget:
        return [triangles.reshape(-1)]
    half = len(triangles) // 2
    return _split_by_vertex_budget(triangles[:half].reshape(-1), budget) + _split_by_vertex_budget(
        triangles[half:].reshape(-1), budget
    )


def layer_uv(primitive: Primitive, uv_set: int) -> np.ndarray:
    """The mesh uv a CS2 `g_nUVSet` value selects: 2 is the secondary uv, anything else the first."""
    return primitive.uv1 if uv_set == 2 and primitive.uv1 is not None else primitive.uv0


def _apply_uv_scale(uv: np.ndarray, axes: tuple[int, int], node_scale: np.ndarray) -> np.ndarray:
    if axes == (0, 0):
        return uv
    out = uv.astype(np.float64).copy()
    for component, axis in enumerate(axes):
        if 1 <= axis <= 3:
            out[:, component] *= node_scale[axis - 1]
    return out


def build_pieces(
    specs: list[MaterialSpec | None],
    material_key: Callable[[MaterialSpec], tuple],
    probe_slot: Callable[[np.ndarray], int],
    probe_volumes: ProbeVolumes,
    cell_size: float,
    origin: np.ndarray,
    detail_nodes: set[int],
    primitives: list[Primitive],
    node_objects: dict[int, int] | None = None,
) -> list[Piece]:
    """Pieces keyed by material key + (detail, object): object is the dynamic object id (0 = static world)."""
    pieces: list[Piece] = []
    node_objects = node_objects or {}
    for primitive in primitives:
        spec = specs[primitive.material]
        if spec is None:
            continue
        obj = node_objects.get(primitive.node, 0)
        key = material_key(spec) + (obj == 0 and primitive.node in detail_nodes, obj)
        pieces.extend(_pieces_for(primitive, spec, key, probe_slot, probe_volumes, cell_size, origin))
    return pieces


def vertex_paint(color: np.ndarray | None, mode: str, n: int) -> np.ndarray:
    """
    The `col` stream: rgba8 the shader applies as mix(1, rgb, a). Modes other than raw are folded into rgb with
    a = 255, so materials that ignore COLOR_0 (foliage keeps animation params there) stay white.
    """
    if color is None or mode == PAINT_NONE:
        return np.full((n, 4), 255, dtype=np.uint8)
    paint = np.clip(color.astype(np.float64), 0.0, 1.0)
    if mode == PAINT_SRGB:
        paint = np.concatenate([srgb_to_linear(paint[:, :3]), np.ones((n, 1))], axis=1)
    elif mode == PAINT_OVERLAY:
        unpainted = ~paint.any(axis=1)
        paint[:, 3] = 1.0
        paint[unpainted] = 1.0
    return np.round(paint * 255).astype(np.uint8)


def _pieces_for(
    primitive: Primitive,
    spec: MaterialSpec,
    key: tuple,
    probe_slot: Callable[[np.ndarray], int],
    probe_volumes: ProbeVolumes,
    cell_size: float,
    origin: np.ndarray,
) -> list[Piece]:
    n = len(primitive.positions)
    blend = spec.family == BLEND
    centroid = primitive.positions.mean(axis=0)
    cell = (int(np.floor((centroid[0] - origin[0]) / cell_size)), int(np.floor((centroid[1] - origin[1]) / cell_size)))

    normal_oct = _unorm8_oct(oct_encode(primitive.normals.astype(np.float64)))
    flags = np.zeros(n, dtype=np.uint8)

    lightmapped = primitive.lightmap_uv is not None
    if lightmapped:
        lm = np.clip(np.round(primitive.lightmap_uv * 65535), 0, 65535).astype(np.uint16)
    else:
        irradiance, visibility = probe_volumes.sample(primitive.positions, primitive.normals.astype(np.float64))
        lm = encode_rgbm(irradiance)
        flags |= np.uint8(2) | (np.clip(np.round(visibility * 63), 0, 63).astype(np.uint8) << 2)
    probe = np.full(n, probe_slot(centroid), dtype=np.uint8)
    nrm = np.stack([normal_oct[:, 0], normal_oct[:, 1], flags, probe], axis=1)

    uv1 = _apply_uv_scale(layer_uv(primitive, spec.uv_sets[0]), spec.uv_scale_axes, primitive.node_scale)
    shift = np.floor(uv1.mean(axis=0))
    uv1 = uv1 - shift
    if blend:
        uv2 = _apply_uv_scale(layer_uv(primitive, spec.uv_sets[1]), spec.uv2_scale_axes, primitive.node_scale)
        uv2 = uv2 - np.floor(uv2.mean(axis=0))
        uv = np.concatenate([uv1, uv2], axis=1)
    else:
        uv = uv1
    uv = uv.astype(np.float32).reshape(n, -1)

    color = vertex_paint(primitive.color, spec.params.get("vertexPaint", PAINT_NONE), n)
    streams: dict[str, np.ndarray] = {
        "nrm": nrm,
        "uv": uv,
        "lm": lm,
        "col": color,
    }
    if blend:
        blend_values = primitive.blend if primitive.blend is not None else np.zeros((n, 4), dtype=np.float32)
        streams["bld"] = np.clip(np.round(blend_values * 255), 0, 255).astype(np.uint8)
        overlay_uv = layer_uv(primitive, spec.overlay_uv_set)
        overlay_uv = overlay_uv - np.floor(overlay_uv.mean(axis=0))
        streams["ovl"] = overlay_uv.astype(np.float32).reshape(n, 2)

    pieces = []
    for part in _split_by_vertex_budget(primitive.indices, MAX_VERTICES):
        used, local = np.unique(part, return_inverse=True)
        pieces.append(
            Piece(
                material=spec.index,
                key=key,
                cell=cell,
                positions=primitive.positions[used],
                streams={name: data[used] for name, data in streams.items()},
                indices=local.astype(np.uint32),
            )
        )
    return pieces


def _simplified_indices(piece: Piece, error: float, attribute_scale: float, lock_border: bool) -> np.ndarray | None:
    """
    Error-bounded simplification over the piece's own vertices, or None when it would not remove anything. With
    `lock_border`, open borders (seams with neighbouring pieces) stay exact; without it they may move by up to
    `error`. `attribute_scale` relaxes how much UV, lightmap and paint changes count against the error.
    """
    if error <= 0 or piece.key[0] == TRANSLUCENT or len(piece.indices) < 3 * 16:
        return None
    streams = piece.streams
    normal = (streams["nrm"][:, :2].astype(np.float32) - 128.0) / 127.0
    lm = streams["lm"].astype(np.float32) / 65535.0
    lm[(streams["nrm"][:, 2] & 2) != 0] = 0.0
    parts = [
        normal * SIMPLIFY_WEIGHTS["normal"],
        streams["uv"][:, :2] * SIMPLIFY_WEIGHTS["uv"],
        lm * SIMPLIFY_WEIGHTS["lm"],
        streams["col"].astype(np.float32) / 255.0 * SIMPLIFY_WEIGHTS["col"],
    ]
    if "bld" in streams:
        parts.append(streams["uv"][:, 2:4] * SIMPLIFY_WEIGHTS["uv"])
        parts.append(streams["bld"].astype(np.float32) / 255.0 * SIMPLIFY_WEIGHTS["bld"])
    attributes = np.concatenate(parts, axis=1) * attribute_scale
    # Attributes are pre-weighted so the metric stays in inches.
    weights = np.ones(attributes.shape[1], dtype=np.float32)
    origin = piece.positions.mean(axis=0)
    options = meshopt_native.SIMPLIFY_ERROR_ABSOLUTE | (meshopt_native.SIMPLIFY_LOCK_BORDER if lock_border else 0)
    indices = meshopt_native.simplify_with_attributes(
        piece.indices, piece.positions - origin, attributes, weights, error, options
    )
    if len(indices) == 0 or len(indices) >= len(piece.indices):
        return None
    return indices


def add_lod(pieces: list[Piece], error: float, attribute_scale: float) -> None:
    """Sets each piece's far LOD: a coarser index list over its own vertices, with unlocked borders."""
    for piece in pieces:
        if not piece.lod_done:
            piece.lod = _simplified_indices(piece, error, attribute_scale, lock_border=False)
            piece.lod_done = True


def _exp_bits(values: np.ndarray, texels: int) -> int:
    peak = float(np.abs(values).max()) if values.size else 0.0
    exponent = math.frexp(peak)[1] if peak > 0 else 0
    step_bits = math.ceil(math.log2(max(texels, 1) * UV_TEXEL_FRACTION))
    return int(min(24, max(2, exponent + step_bits + 1)))


def _class_texels(class_id: str) -> int:
    digits = "".join(ch for ch in class_id.split("_")[0] if ch.isdigit())
    return int(digits) if digits else 0


@dataclass
class EncodedChunk:
    streams: dict[str, bytes]
    strides: dict[str, int]
    indices: bytes
    vertex_count: int
    index_count: int
    box_min: np.ndarray
    box_max: np.ndarray
    parts: np.ndarray
    lod_indices: bytes | None = None
    lod_index_count: int = 0
    lod_parts: np.ndarray | None = None


def _cell_ordered(
    indices: np.ndarray, positions: np.ndarray, cell_index: CellIndex | None, vertex_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Groups triangles into parts by visibility cell, each part cache-optimized on its own."""
    triangles = indices.reshape(-1, 3)
    if cell_index is not None:
        cells = cell_index(positions[triangles].mean(axis=1)).astype(np.int64)
    else:
        cells = np.zeros(len(triangles), dtype=np.int64)
    order = np.argsort(cells, kind="stable")
    triangles, cells = triangles[order], cells[order]
    starts = np.flatnonzero(np.r_[True, cells[1:] != cells[:-1]])
    ends = np.r_[starts[1:], len(cells)]
    optimized = np.empty(len(indices), dtype=np.uint32)
    for start, end in zip(starts, ends):
        part = np.ascontiguousarray(triangles[start:end].reshape(-1))
        result = np.zeros_like(part)
        meshoptimizer.optimize_vertex_cache(result, part, len(part), vertex_count)
        optimized[3 * start : 3 * end] = result
    return optimized, starts, ends, cells


def _parts_table(indices: np.ndarray, starts: np.ndarray, ends: np.ndarray, cells: np.ndarray, quantized: np.ndarray) -> np.ndarray:
    parts = np.zeros(len(starts), dtype=PART_DTYPE)
    for i, (start, end) in enumerate(zip(starts, ends)):
        corner = quantized[indices[3 * start : 3 * end], :3]
        parts[i] = (3 * start, 3 * (end - start), cells[start], np.concatenate([corner.min(axis=0), corner.max(axis=0)]))
    return parts


def _encode_chunk(pieces: list[Piece], cell_index: CellIndex | None) -> EncodedChunk:
    positions = np.concatenate([p.positions for p in pieces])
    materials = np.concatenate([np.full(len(p.positions), p.material, dtype=np.uint16) for p in pieces])
    offsets = np.cumsum([0] + [len(p.positions) for p in pieces[:-1]])
    indices = np.concatenate([p.indices + offset for p, offset in zip(pieces, offsets)]).astype(np.uint32)
    lod: np.ndarray | None = None
    if any(p.lod is not None for p in pieces):
        lod = np.concatenate([p.lod_indices() + offset for p, offset in zip(pieces, offsets)]).astype(np.uint32)
        if len(lod) > LOD_MAX_SHARE * len(indices):
            lod = None
    names = list(pieces[0].streams)
    streams = {name: np.concatenate([p.streams[name] for p in pieces]) for name in names}
    vertex_count = len(positions)

    box_min = positions.min(axis=0)
    box_max = positions.max(axis=0)
    extent = np.maximum(box_max - box_min, 1e-3)
    quantized = np.clip(np.round((positions - box_min) / extent * 65535), 0, 65535).astype(np.uint16)
    streams["pos"] = np.concatenate([quantized, materials[:, None]], axis=1)
    key = pieces[0].key
    texels = {"uv": max(_class_texels(key[2]), _class_texels(key[3])), "ovl": OVERLAY_TEXELS}
    for name in FILTERED_STREAMS:
        if name in streams:
            streams[name] = meshopt_native.encode_filter_exp(streams[name], _exp_bits(streams[name], texels[name]))

    optimized, starts, ends, cells = _cell_ordered(indices, positions, cell_index, vertex_count)
    remap = np.zeros(vertex_count, dtype=np.uint32)
    unique = meshoptimizer.optimize_vertex_fetch_remap(remap, optimized, len(optimized), vertex_count)
    indices = remap[optimized]
    encoded: dict[str, bytes] = {}
    strides: dict[str, int] = {}
    reordered_pos: np.ndarray | None = None
    for name, data in streams.items():
        reordered = np.empty((unique,) + data.shape[1:], dtype=data.dtype)
        valid = remap < unique
        reordered[remap[valid]] = data[valid]
        if name == "pos":
            reordered_pos = reordered
        rows = np.ascontiguousarray(reordered).view(np.uint8).reshape(unique, -1)
        strides[name] = rows.shape[1]
        encoded[name] = meshoptimizer.encode_vertex_buffer(rows, unique, rows.shape[1])
    assert reordered_pos is not None
    chunk = EncodedChunk(
        streams=encoded,
        strides=strides,
        indices=meshoptimizer.encode_index_buffer(indices.astype(np.uint32), len(indices), unique),
        vertex_count=unique,
        index_count=len(indices),
        box_min=box_min,
        box_max=box_max,
        parts=_parts_table(indices, starts, ends, cells, reordered_pos),
    )
    if lod is not None:
        # The LOD only references vertices the full index list uses, so the same fetch remap applies.
        lod_optimized, lod_starts, lod_ends, lod_cells = _cell_ordered(lod, positions, cell_index, vertex_count)
        lod = remap[lod_optimized]
        chunk.lod_indices = meshoptimizer.encode_index_buffer(lod.astype(np.uint32), len(lod), unique)
        chunk.lod_index_count = len(lod)
        chunk.lod_parts = _parts_table(lod, lod_starts, lod_ends, lod_cells, reordered_pos)
    return chunk


def piece_centroids(piece: Piece) -> np.ndarray:
    """Centroids of the piece's triangles, LOD triangles included."""
    triangles = piece.indices if piece.lod is None else np.concatenate([piece.indices, piece.lod])
    return piece.positions[triangles.reshape(-1, 3)].mean(axis=1)


def triangle_batches(pieces: list[Piece], batch: int):
    """Yields (a, b, c) triangle corner arrays over all pieces (LOD triangles included), about `batch` at a time."""
    pending: list[np.ndarray] = []
    count = 0
    for piece in pieces:
        triangles = piece.indices if piece.lod is None else np.concatenate([piece.indices, piece.lod])
        corners = piece.positions[triangles.reshape(-1, 3)]
        pending.append(corners)
        count += len(corners)
        if count >= batch:
            merged = np.concatenate(pending)
            yield merged[:, 0], merged[:, 1], merged[:, 2]
            pending, count = [], 0
    if pending:
        merged = np.concatenate(pending)
        yield merged[:, 0], merged[:, 1], merged[:, 2]


def write_chunks(pieces: list[Piece], out_dir: Path, cell_index: CellIndex | None = None, prefix: str = "") -> list[ChunkRecord]:
    """Groups pieces by (cell, key), packs them into <= 65535-vertex chunks, writes one file per cell."""
    buckets: dict[tuple, list[Piece]] = defaultdict(list)
    for piece in pieces:
        buckets[(piece.cell, piece.key)].append(piece)

    per_cell: dict[tuple[int, int], list[tuple[tuple, list[Piece]]]] = defaultdict(list)
    for (cell, key), bucket in buckets.items():
        bucket.sort(key=lambda p: p.material)
        current: list[Piece] = []
        count = 0
        for piece in bucket:
            if current and count + len(piece.positions) > MAX_VERTICES:
                per_cell[cell].append((key, current))
                current, count = [], 0
            current.append(piece)
            count += len(piece.positions)
        if current:
            per_cell[cell].append((key, current))

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[ChunkRecord] = []
    for cell, chunks in sorted(per_cell.items()):
        name = f"{prefix}geo_{cell[0]}_{cell[1]}.bin"
        offset = 0
        with (out_dir / name).open("wb") as handle:
            for key, chunk_pieces in chunks:
                chunk = _encode_chunk(chunk_pieces, cell_index)
                record = ChunkRecord(
                    key=key,
                    cell=cell,
                    file=name,
                    vertex_count=chunk.vertex_count,
                    index_count=chunk.index_count,
                    box_min=[round(float(x), 3) for x in chunk.box_min],
                    box_max=[round(float(x), 3) for x in chunk.box_max],
                )

                def write(data: bytes, align: int = 1) -> int:
                    nonlocal offset
                    pad = -offset % align
                    handle.write(b"\0" * pad + data)
                    offset += pad + len(data)
                    return offset - len(data)

                for stream_name, data in chunk.streams.items():
                    stream_filter = "EXPONENTIAL" if stream_name in FILTERED_STREAMS else "NONE"
                    record.streams[stream_name] = [write(data), len(data), chunk.strides[stream_name], stream_filter]
                record.index = [write(chunk.indices), len(chunk.indices)]
                record.parts = [write(chunk.parts.tobytes(), 4), len(chunk.parts)]
                if chunk.lod_indices is not None and chunk.lod_parts is not None:
                    record.lod = {
                        "index": [write(chunk.lod_indices), len(chunk.lod_indices)],
                        "indexCount": chunk.lod_index_count,
                        "parts": [write(chunk.lod_parts.tobytes(), 4), len(chunk.lod_parts)],
                    }
                records.append(record)
    return records


def records_json(records: list[ChunkRecord], key_fields: list[str]) -> list[dict[str, Any]]:
    return [
        {
            **dict(zip(key_fields, record.key)),
            "cell": list(record.cell),
            "file": record.file,
            "vertexCount": record.vertex_count,
            "indexCount": record.index_count,
            "boxMin": record.box_min,
            "boxMax": record.box_max,
            "streams": record.streams,
            "index": record.index,
            "parts": record.parts,
            **({"lod": record.lod} if record.lod else {}),
        }
        for record in records
    ]


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
