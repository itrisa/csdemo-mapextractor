"""Minimal uncompressed glTF 2.0 binary collision geometry codec."""

from __future__ import annotations

import json
import math
import struct
from typing import Any

from .geometry import CollisionGeometry
from .profiles import VISUAL_OCCLUDERS, includes_mesh


_HEADER = struct.Struct("<4sII")
_CHUNK_HEADER = struct.Struct("<II")
_GLB_MAGIC = b"glTF"
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942
_FLOAT = 5126
_UNSIGNED_SHORT = 5123
_UNSIGNED_INT = 5125


def dumps(
    geometry: CollisionGeometry,
    *,
    selection_profile: str = VISUAL_OCCLUDERS,
    meshopt: bool = False,
) -> bytes:
    if not geometry.positions or not geometry.triangles:
        raise ValueError("GLB geometry must contain at least one triangle")
    positions = b"".join(struct.pack("<3f", *position) for position in geometry.positions)
    flat_indices = tuple(index for triangle in geometry.triangles for index in triangle)
    index_component = _UNSIGNED_SHORT if len(geometry.positions) <= 65_536 else _UNSIGNED_INT
    index_format = "<H" if index_component == _UNSIGNED_SHORT else "<I"
    indices = b"".join(struct.pack(index_format, index) for index in flat_indices)
    if meshopt:
        binary, buffers, buffer_views = _compress(positions, indices, len(geometry.positions), len(flat_indices))
    else:
        binary = positions + indices
        binary += b"\x00" * (-len(binary) % 4)
        buffers = [{"byteLength": len(binary)}]
        buffer_views = [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962},
            {
                "buffer": 0,
                "byteOffset": len(positions),
                "byteLength": len(indices),
                "target": 34963,
            },
        ]

    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "csdemo-mapextractor"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": buffers,
        "bufferViews": buffer_views,
        "accessors": [
            {
                "bufferView": 0,
                "componentType": _FLOAT,
                "count": len(geometry.positions),
                "type": "VEC3",
                "min": _bounds(geometry)[0],
                "max": _bounds(geometry)[1],
            },
            {
                "bufferView": 1,
                "componentType": index_component,
                "count": len(flat_indices),
                "type": "SCALAR",
            },
        ],
        "extras": {
            "coordinate_system": "source2-hammer-z-up",
            "units": "hammer",
            "transform_applied": False,
            "selection_profile": selection_profile,
        },
    }
    if meshopt:
        document["extensionsUsed"] = ["EXT_meshopt_compression"]
        document["extensionsRequired"] = ["EXT_meshopt_compression"]
    encoded_json = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded_json += b" " * (-len(encoded_json) % 4)
    total_length = _HEADER.size + 2 * _CHUNK_HEADER.size + len(encoded_json) + len(binary)
    return b"".join(
        (
            _HEADER.pack(_GLB_MAGIC, 2, total_length),
            _CHUNK_HEADER.pack(len(encoded_json), _JSON_CHUNK),
            encoded_json,
            _CHUNK_HEADER.pack(len(binary), _BIN_CHUNK),
            binary,
        )
    )


def loads(data: bytes, *, selection_profile: str | None = None) -> CollisionGeometry:
    document, binary = _parse(data)
    meshes = document.get("meshes", [])
    primitives = [primitive for mesh in meshes for primitive in mesh.get("primitives", [])]
    if selection_profile is None and len(primitives) == 1:
        positions, indices = _primitive_data(document, binary, primitives[0])
        triangles = tuple(zip(indices[0::3], indices[1::3], indices[2::3], strict=True))
        return CollisionGeometry(positions, triangles)

    triangles = []
    for mesh in meshes:
        mesh_name = mesh.get("name")
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode", 4) != 4:
                raise ValueError("GLB primitive must use TRIANGLES mode")
            material_name = _material_name(document, primitive)
            if selection_profile is not None and not includes_mesh(selection_profile, mesh_name, material_name):
                continue
            positions, indices = _primitive_data(document, binary, primitive)
            triangles.extend(
                (positions[first], positions[second], positions[third])
                for first, second, third in zip(indices[0::3], indices[1::3], indices[2::3], strict=True)
            )
    if not triangles:
        raise ValueError("GLB contains no triangles selected by the collision profile")
    return CollisionGeometry.from_triangle_soup(triangles)


def _primitive_data(
    document: dict[str, Any], binary: bytes, primitive: dict[str, Any]
) -> tuple[tuple[tuple[float, float, float], ...], tuple[int, ...]]:
    if primitive.get("mode", 4) != 4:
        raise ValueError("GLB primitive must use TRIANGLES mode")
    position_accessor = document["accessors"][primitive["attributes"]["POSITION"]]
    positions = _read_positions(document, binary, position_accessor)
    if "indices" in primitive:
        index_accessor = document["accessors"][primitive["indices"]]
        indices = _read_indices(document, binary, index_accessor)
    else:
        indices = tuple(range(len(positions)))
    if len(indices) % 3:
        raise ValueError("GLB index count must be a multiple of three")
    return positions, indices


def inspect(data: bytes) -> dict[str, Any]:
    document, _ = _parse(data)
    geometry = loads(data)
    minimum, maximum = _bounds(geometry)
    return {
        "format": "glTF 2.0",
        "vertex_count": len(geometry.positions),
        "triangle_count": len(geometry.triangles),
        "bounds": {"min": minimum, "max": maximum},
        "extras": document.get("extras", {}),
    }


def _parse(data: bytes) -> tuple[dict[str, Any], bytes]:
    if len(data) < _HEADER.size:
        raise ValueError("GLB header is truncated")
    magic, version, declared_length = _HEADER.unpack_from(data)
    if magic != _GLB_MAGIC or version != 2 or declared_length != len(data):
        raise ValueError("invalid GLB header")

    offset = _HEADER.size
    chunks: dict[int, bytes] = {}
    while offset < len(data):
        if offset + _CHUNK_HEADER.size > len(data):
            raise ValueError("GLB chunk header is truncated")
        length, chunk_type = _CHUNK_HEADER.unpack_from(data, offset)
        offset += _CHUNK_HEADER.size
        end = offset + length
        if end > len(data):
            raise ValueError("GLB chunk is truncated")
        chunks[chunk_type] = data[offset:end]
        offset = end
    if _JSON_CHUNK not in chunks or _BIN_CHUNK not in chunks:
        raise ValueError("GLB requires JSON and BIN chunks")
    return json.loads(chunks[_JSON_CHUNK]), chunks[_BIN_CHUNK]


def _read_positions(
    document: dict[str, Any], binary: bytes, accessor: dict[str, Any]
) -> tuple[tuple[float, float, float], ...]:
    if accessor.get("componentType") != _FLOAT or accessor.get("type") != "VEC3":
        raise ValueError("POSITION accessor must be float32 VEC3")
    view = document["bufferViews"][accessor["bufferView"]]
    stride = view.get("byteStride", 12)
    view_data = _view_data(binary, view)
    start = accessor.get("byteOffset", 0)
    positions = tuple(struct.unpack_from("<3f", view_data, start + index * stride) for index in range(accessor["count"]))
    if not all(math.isfinite(value) for position in positions for value in position):
        raise ValueError("positions must contain finite coordinates")
    return positions


def _read_indices(
    document: dict[str, Any], binary: bytes, accessor: dict[str, Any]
) -> tuple[int, ...]:
    formats = {_UNSIGNED_SHORT: ("<H", 2), _UNSIGNED_INT: ("<I", 4)}
    if accessor.get("type") != "SCALAR" or accessor.get("componentType") not in formats:
        raise ValueError("indices must be unsigned SCALAR values")
    index_format, size = formats[accessor["componentType"]]
    view = document["bufferViews"][accessor["bufferView"]]
    stride = view.get("byteStride", size)
    view_data = _view_data(binary, view)
    start = accessor.get("byteOffset", 0)
    return tuple(struct.unpack_from(index_format, view_data, start + index * stride)[0] for index in range(accessor["count"]))


def _compress(
    positions: bytes, indices: bytes, vertex_count: int, index_count: int
) -> tuple[bytes, list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import meshoptimizer
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Meshopt output requires the 'meshopt' optional dependency") from error

    meshoptimizer.encode_vertex_version(0)
    meshoptimizer.encode_index_version(0)
    vertex_array = np.frombuffer(positions, dtype="<f4").reshape(vertex_count, 3)
    index_dtype = "<u2" if len(indices) == index_count * 2 else "<u4"
    index_array = np.frombuffer(indices, dtype=index_dtype).astype(np.uint32)
    encoded_positions = bytes(meshoptimizer.encode_vertex_buffer(vertex_array))
    encoded_indices = bytes(meshoptimizer.encode_index_sequence(index_array))
    index_offset = (len(encoded_positions) + 3) & ~3
    binary = encoded_positions + b"\x00" * (index_offset - len(encoded_positions)) + encoded_indices
    binary += b"\x00" * (-len(binary) % 4)
    index_stride = len(indices) // index_count
    extension = "EXT_meshopt_compression"
    buffers = [
        {"byteLength": len(binary)},
        {"byteLength": len(positions) + len(indices)},
    ]
    views = [
        {
            "buffer": 1,
            "byteOffset": 0,
            "byteLength": len(positions),
            "target": 34962,
            "extensions": {
                extension: {
                    "buffer": 0,
                    "byteOffset": 0,
                    "byteLength": len(encoded_positions),
                    "byteStride": 12,
                    "count": vertex_count,
                    "mode": "ATTRIBUTES",
                    "filter": "NONE",
                }
            },
        },
        {
            "buffer": 1,
            "byteOffset": len(positions),
            "byteLength": len(indices),
            "target": 34963,
            "extensions": {
                extension: {
                    "buffer": 0,
                    "byteOffset": index_offset,
                    "byteLength": len(encoded_indices),
                    "byteStride": index_stride,
                    "count": index_count,
                    "mode": "INDICES",
                    "filter": "NONE",
                }
            },
        },
    ]
    return binary, buffers, views


def _view_data(binary: bytes, view: dict[str, Any]) -> bytes:
    compression = view.get("extensions", {}).get("EXT_meshopt_compression")
    if compression is None:
        start = view.get("byteOffset", 0)
        return binary[start : start + view["byteLength"]]
    try:
        import meshoptimizer
    except ImportError as error:
        raise RuntimeError("Meshopt input requires the 'meshopt' optional dependency") from error
    start = compression.get("byteOffset", 0)
    encoded = binary[start : start + compression["byteLength"]]
    mode = compression["mode"]
    count = compression["count"]
    stride = compression["byteStride"]
    if mode == "ATTRIBUTES":
        return meshoptimizer.decode_vertex_buffer(count, stride, encoded).tobytes()
    if mode == "INDICES":
        return meshoptimizer.decode_index_sequence(count, stride, encoded).tobytes()
    if mode == "TRIANGLES":
        return meshoptimizer.decode_index_buffer(count, stride, encoded).tobytes()
    raise ValueError(f"unsupported Meshopt mode: {mode}")


def _material_name(document: dict[str, Any], primitive: dict[str, Any]) -> str | None:
    material_index = primitive.get("material")
    if material_index is None:
        return None
    return document.get("materials", [])[material_index].get("name")


def _bounds(geometry: CollisionGeometry) -> tuple[list[float], list[float]]:
    if not geometry.positions:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return (
        [min(position[axis] for position in geometry.positions) for axis in range(3)],
        [max(position[axis] for position in geometry.positions) for axis in range(3)],
    )