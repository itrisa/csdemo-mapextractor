"""Serialise coloured collision layers as a meshopt-compressed GLB for csdemo-client.

Each layer becomes one node/mesh with POSITION (f32), COLOR_0 (rgba8, normalised) and indices. Identical
position+colour vertices are merged and reordered with meshopt's vertex-cache and vertex-fetch optimisers
(lossless), then encoded with EXT_meshopt_compression (codec v0, ATTRIBUTES/INDICES modes, filter NONE), the
same encoding extractor/glb.py uses for the plain collision GLBs.
"""

from __future__ import annotations

import json
import struct

import numpy as np

SELECTION_PROFILE = "default-plus-enabled-func-brush-v1+albedo-v2"


def _optimize(P: np.ndarray, C: np.ndarray, I: np.ndarray):
    import meshoptimizer

    key = np.concatenate([P.view(np.uint32), C.astype(np.uint32)], 1)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    P, C, I = P[first], C[first], inverse.ravel()[I].astype(np.uint32)
    cache = np.empty_like(I)
    meshoptimizer.optimize_vertex_cache(cache, I, len(I), len(P))
    remap = np.empty(len(P), np.uint32)
    meshoptimizer.optimize_vertex_fetch_remap(remap, cache, len(cache), len(P))
    used = remap != 0xFFFFFFFF
    order = np.empty(int(remap[used].max()) + 1, np.int64)
    order[remap[used]] = np.flatnonzero(used)
    return P[order], C[order], remap[cache]


def encode(layers, map_name: str) -> bytes:
    """layers: [(name, tags, V (n,3), T (m,3), rgba (n,4) uint8)]."""
    import meshoptimizer

    meshoptimizer.encode_vertex_version(0)
    meshoptimizer.encode_index_version(0)
    encoded = bytearray()
    raw_length = 0
    views: list[dict] = []
    accessors: list[dict] = []

    def add(array: np.ndarray, mode: str, target: int, accessor: dict) -> int:
        nonlocal raw_length
        array = np.ascontiguousarray(array)
        if mode == "INDICES":
            data = bytes(meshoptimizer.encode_index_sequence(array.astype(np.uint32).ravel()))
            stride = array.dtype.itemsize
        else:
            stride = array.shape[1] * array.dtype.itemsize
            data = bytes(meshoptimizer.encode_vertex_buffer(array.view(np.uint8).reshape(len(array), stride)))
        offset = len(encoded)
        encoded.extend(data)
        encoded.extend(b"\x00" * (-len(encoded) % 4))
        views.append(
            {
                "buffer": 1,
                "byteOffset": raw_length,
                "byteLength": array.nbytes,
                "target": target,
                "extensions": {
                    "EXT_meshopt_compression": {
                        "buffer": 0,
                        "byteOffset": offset,
                        "byteLength": len(data),
                        "byteStride": stride,
                        "count": len(array),
                        "mode": mode,
                        "filter": "NONE",
                    }
                },
            }
        )
        raw_length += array.nbytes + (-array.nbytes % 4)
        accessor["bufferView"] = len(views) - 1
        accessors.append(accessor)
        return len(accessors) - 1

    meshes, nodes = [], []
    for name, tags, V, T, rgba in layers:
        P = np.ascontiguousarray(V, np.float32)
        C = np.ascontiguousarray(rgba, np.uint8)
        I = np.asarray(T, np.uint32).ravel()
        triangle_count = len(I) // 3
        P, C, I = _optimize(P, C, I)
        index = I.astype("<u2") if len(P) <= 65536 else I.astype("<u4")
        position = add(
            P.astype("<f4"),
            "ATTRIBUTES",
            34962,
            {
                "componentType": 5126,
                "count": len(P),
                "type": "VEC3",
                "min": P.min(0).tolist(),
                "max": P.max(0).tolist(),
            },
        )
        color = add(
            C,
            "ATTRIBUTES",
            34962,
            {"componentType": 5121, "normalized": True, "count": len(C), "type": "VEC4"},
        )
        indices = add(
            index,
            "INDICES",
            34963,
            {
                "componentType": 5123 if index.dtype.itemsize == 2 else 5125,
                "count": len(index),
                "type": "SCALAR",
            },
        )
        meshes.append(
            {
                "name": name,
                "primitives": [{"attributes": {"POSITION": position, "COLOR_0": color}, "indices": indices}],
            }
        )
        nodes.append(
            {
                "name": name,
                "mesh": len(meshes) - 1,
                "extras": {"collision_tags": list(tags), "triangle_count": triangle_count},
            }
        )

    document = {
        "asset": {"version": "2.0", "generator": "csdemo-mapextractor albedo v2"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "buffers": [
            {"byteLength": len(encoded)},
            {"byteLength": raw_length, "extensions": {"EXT_meshopt_compression": {"fallback": True}}},
        ],
        "bufferViews": views,
        "accessors": accessors,
        "extensionsUsed": ["EXT_meshopt_compression"],
        "extensionsRequired": ["EXT_meshopt_compression"],
        "extras": {
            "map": map_name,
            "coordinate_system": "source2-hammer-z-up",
            "units": "hammer",
            "selection_profile": SELECTION_PROFILE,
        },
    }
    js = json.dumps(document, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(encoded)
    return (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(js), 0x4E4F534A)
        + js
        + struct.pack("<II", len(encoded), 0x004E4942)
        + bytes(encoded)
    )
