"""Layered collision geometry for inspection and client-side filtering."""

from __future__ import annotations

from dataclasses import dataclass
import json
import struct
from typing import Any

from .geometry import CollisionGeometry, Triangle
from .entity_data import EntityState
from . import glb


_METERS_PER_HAMMER_UNIT = 0.0254


@dataclass(frozen=True, slots=True)
class CollisionLayer:
    name: str
    tags: tuple[str, ...]
    geometry: CollisionGeometry
    surface_properties: tuple[str, ...] = ()


def from_vrf(data: bytes) -> tuple[CollisionLayer, ...]:
    document, binary = glb._parse(data)
    grouped: dict[tuple[str, ...], list[Triangle]] = {}
    surfaces: dict[tuple[str, ...], set[str]] = {}
    meshes = document.get("meshes", [])

    for node in document.get("nodes", []):
        mesh_index = node.get("mesh")
        if mesh_index is None:
            continue
        extras = node.get("extras", {})
        tags = tuple(sorted(str(tag) for tag in extras.get("InteractAs", [])))
        grouped.setdefault(tags, [])
        surface = extras.get("SurfaceProperty")
        if surface:
            surfaces.setdefault(tags, set()).add(str(surface))
        for primitive in meshes[mesh_index].get("primitives", []):
            positions, indices = glb._primitive_data(document, binary, primitive)
            grouped[tags].extend(
                (positions[first], positions[second], positions[third])
                for first, second, third in zip(indices[0::3], indices[1::3], indices[2::3], strict=True)
            )

    return tuple(
        CollisionLayer(
            name="default" if not tags else "+".join(tags),
            tags=tags,
            geometry=CollisionGeometry.from_triangle_soup(triangles),
            surface_properties=tuple(sorted(surfaces.get(tags, set()))),
        )
        for tags, triangles in sorted(grouped.items(), key=lambda item: (bool(item[0]), item[0]))
        if triangles
    )


def dumps(
    layers: tuple[CollisionLayer, ...],
    *,
    map_name: str,
    selection_profile: str = "all-physics-layered-v1",
) -> bytes:
    if not layers:
        raise ValueError("at least one collision layer is required")

    binary_parts: list[bytes] = []
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    compressed_offset = 0
    decoded_offset = 0

    for layer in layers:
        geometry = layer.geometry
        positions = b"".join(struct.pack("<3f", *position) for position in geometry.positions)
        flat_indices = tuple(index for triangle in geometry.triangles for index in triangle)
        index_component = 5123 if len(geometry.positions) <= 65_536 else 5125
        index_format = "<H" if index_component == 5123 else "<I"
        indices = b"".join(struct.pack(index_format, index) for index in flat_indices)
        encoded, _, views = glb._compress(positions, indices, len(geometry.positions), len(flat_indices))

        position_view_index = len(buffer_views)
        index_view_index = position_view_index + 1
        views[0]["byteOffset"] = decoded_offset
        views[1]["byteOffset"] = decoded_offset + len(positions)
        for view in views:
            extension = view["extensions"]["EXT_meshopt_compression"]
            extension["byteOffset"] += compressed_offset
            buffer_views.append(view)

        position_accessor = len(accessors)
        index_accessor = position_accessor + 1
        minimum, maximum = glb._bounds(geometry)
        accessors.extend(
            (
                {
                    "bufferView": position_view_index,
                    "componentType": 5126,
                    "count": len(geometry.positions),
                    "type": "VEC3",
                    "min": minimum,
                    "max": maximum,
                },
                {
                    "bufferView": index_view_index,
                    "componentType": index_component,
                    "count": len(flat_indices),
                    "type": "SCALAR",
                },
            )
        )
        meshes.append(
            {
                "name": layer.name,
                "primitives": [{"attributes": {"POSITION": position_accessor}, "indices": index_accessor}],
            }
        )
        nodes.append(
            {
                "name": layer.name,
                "mesh": len(meshes) - 1,
                "extras": {
                    "collision_tags": list(layer.tags),
                    "surface_properties": list(layer.surface_properties),
                    "triangle_count": len(geometry.triangles),
                },
            }
        )
        binary_parts.append(encoded)
        compressed_offset += len(encoded)
        decoded_offset += len(positions) + len(indices)

    binary = b"".join(binary_parts)
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "csdemo-mapextractor"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "buffers": [{"byteLength": len(binary)}, {"byteLength": decoded_offset}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "extensionsUsed": ["EXT_meshopt_compression"],
        "extensionsRequired": ["EXT_meshopt_compression"],
        "extras": {
            "map": map_name,
            "coordinate_system": "source2-hammer-z-up",
            "units": "hammer",
            "transform_applied": False,
            "selection_profile": selection_profile,
        },
    }
    encoded_json = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded_json += b" " * (-len(encoded_json) % 4)
    total_length = 12 + 8 + len(encoded_json) + 8 + len(binary)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total_length),
            struct.pack("<II", len(encoded_json), 0x4E4F534A),
            encoded_json,
            struct.pack("<II", len(binary), 0x004E4942),
            binary,
        )
    )


def entity_layers(
    data: bytes, classname: str, *, states: tuple[EntityState, ...] = ()
) -> tuple[CollisionLayer, ...]:
    document, binary = glb._parse(data)
    meshes = document.get("meshes", [])
    grouped: dict[tuple[str, ...], list[Triangle]] = {}
    surfaces: dict[tuple[str, ...], set[str]] = {}

    for node in document.get("nodes", []):
        if node.get("name") != classname or node.get("mesh") is None:
            continue
        matrix = node.get("matrix")
        if matrix is None or len(matrix) != 16:
            raise ValueError(f"{classname} physics node requires a 4x4 world matrix")
        extras = node.get("extras", {})
        origin = _to_hammer_world((0.0, 0.0, 0.0), matrix)
        matching_states = [
            state
            for state in states
            if max(abs(origin[axis] - state.origin[axis]) for axis in range(3)) < 0.01
        ]
        if matching_states and all(state.disabled for state in matching_states):
            continue
        tags = tuple(sorted(str(tag) for tag in extras.get("InteractAs", [])))
        grouped.setdefault(tags, [])
        if surface := extras.get("SurfaceProperty"):
            surfaces.setdefault(tags, set()).add(str(surface))
        for primitive in meshes[node["mesh"]].get("primitives", []):
            positions, indices = glb._primitive_data(document, binary, primitive)
            world_positions = tuple(_to_hammer_world(position, matrix) for position in positions)
            grouped[tags].extend(
                (world_positions[first], world_positions[second], world_positions[third])
                for first, second, third in zip(indices[0::3], indices[1::3], indices[2::3], strict=True)
            )
    return tuple(
        CollisionLayer(
            name=f"entity:{classname}" if not tags else f"entity:{classname}:{'+'.join(tags)}",
            tags=tags,
            geometry=CollisionGeometry.from_triangle_soup(triangles),
            surface_properties=tuple(sorted(surfaces.get(tags, set()))),
        )
        for tags, triangles in sorted(grouped.items(), key=lambda item: (bool(item[0]), item[0]))
        if triangles
    )


def _to_hammer_world(position: tuple[float, float, float], matrix: list[float]) -> tuple[float, float, float]:
    x, y, z = position
    gltf_x = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]
    gltf_y = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]
    gltf_z = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]
    return (
        gltf_z / _METERS_PER_HAMMER_UNIT,
        gltf_x / _METERS_PER_HAMMER_UNIT,
        gltf_y / _METERS_PER_HAMMER_UNIT,
    )