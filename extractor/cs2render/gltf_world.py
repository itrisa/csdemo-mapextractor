"""Reads Source2Viewer world.gltf exports back into Source world space."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from PIL import Image

from .bindpose import mesh_key

# Source2Viewer bakes Source->glTF as gltf = 0.0254 * (y, z, x); this is its inverse.
INCHES_PER_METER = 1.0 / 0.0254
_SOURCE_FROM_GLTF = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# Source2Viewer pushes overlay primitives along their normals by 0.01 / 0.0254 glTF units (~0.39 m,
# meant as 1 cm) to avoid z-fighting; it is undone here and the runtime depth-biases decals instead.
_OVERLAY_NORMAL_OFFSET = 0.01 / 0.0254

_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _uses_secondary_uv(material: dict[str, Any]) -> bool:
    """Whether the material's vertex shader reads TEXCOORD_1: complex shaders' F_FORCE_UV2 / F_SECONDARY_UV, or an
    environment layer, detail normal or shared colour overlay set to UV set 2 (VRF csgo_environment.vert)."""
    vmat = material.get("extras", {}).get("vmat", {})
    ints = vmat.get("IntParams", {})
    if any(int(ints.get(flag, 0)) == 1 for flag in ("F_FORCE_UV2", "F_SECONDARY_UV")):
        return True
    shader = vmat.get("ShaderName", "")
    if shader not in ("csgo_environment.vfx", "csgo_environment_blend.vfx"):
        return False
    layers = ["1"]
    if shader == "csgo_environment_blend.vfx":
        layers.append("2")
        if int(ints.get("F_ENABLE_LAYER_3", 0)) == 1:
            layers.append("3")
    sets = []
    for n in layers:
        uv_set = int(ints.get(f"g_nUVSet{n}", 1))
        sets.append(uv_set)
        if int(ints.get("F_DETAIL_NORMAL", 0)) == 1:
            detail = int(ints.get(f"g_nDetailUVSet{n}", -1))
            sets.append(uv_set if detail == -1 else detail)
    if shader == "csgo_environment_blend.vfx" and int(ints.get("F_SHARED_COLOR_OVERLAY", 0)) == 1:
        sets.append(int(ints.get("g_nColorOverlayUVSet", 2)))
    return 2 in sets


def is_overlay_material(material: dict[str, Any]) -> bool:
    """Mirrors Source2Viewer's IsMaterialOverlay, i.e. the materials whose glTF positions it offsets."""
    vmat = material.get("extras", {}).get("vmat", {})
    ints = vmat.get("IntParams", {})
    shader = vmat.get("ShaderName", "")
    return (
        any(int(ints.get(flag, 0)) == 1 for flag in ("F_OVERLAY", "F_DEPTHBIAS", "F_DEPTH_BIAS"))
        or shader.endswith("static_overlay.vfx")
        or shader == "citadel_overlay.vfx"
    )
_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_LIGHTMAP_UV_EPSILON = 1e-3
# Least share of triangles inside one lightmap chart for a uv set other than the last to count as the lightmap.
_CHART_MATCH_MIN = 0.5


@dataclass(slots=True)
class Primitive:
    node: int
    material: int
    positions: np.ndarray  # (n, 3) float64 Source world units
    normals: np.ndarray  # (n, 3) float32 world
    tangents: np.ndarray  # (n, 4) float32 world xyz + bitangent sign
    uv0: np.ndarray  # (n, 2) float32
    uv1: np.ndarray | None  # secondary UV (vTEXCOORD1)
    lightmap_uv: np.ndarray | None  # (n, 2) in [0, 1] lightmap atlas space
    color: np.ndarray | None  # (n, 4) float32 vertex paint
    blend: np.ndarray | None  # (n, 4) float32 vColorBlendValues
    indices: np.ndarray  # (m,) uint32 into the compacted vertex arrays
    node_scale: np.ndarray  # (3,) placement scale along Source model axes


class GltfWorld:
    def __init__(self, path: Path, probe_lit: frozenset[str] = frozenset()) -> None:
        self.path = Path(path)
        # Mesh names lit from light probes, whose uv sets never address the lightmap (see worldnodes.py).
        self.probe_lit = probe_lit
        self.document: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        self._buffers = [
            np.memmap(self.path.parent / buffer["uri"], dtype=np.uint8, mode="r") for buffer in self.document["buffers"]
        ]
        self.materials: list[dict[str, Any]] = self.document.get("materials", [])
        self.images: list[dict[str, Any]] = self.document.get("images", [])
        self.textures: list[dict[str, Any]] = self.document.get("textures", [])
        self._world = self._world_matrices()
        self._charts: np.ndarray | None = None

    def accessor(self, index: int) -> np.ndarray:
        accessor = self.document["accessors"][index]
        view = self.document["bufferViews"][accessor["bufferView"]]
        components = _COMPONENTS[accessor["type"]]
        dtype = np.dtype(_DTYPES[accessor["componentType"]])
        count = accessor["count"]
        offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        stride = view.get("byteStride") or components * dtype.itemsize
        raw = self._buffers[view["buffer"]]
        if stride == components * dtype.itemsize:
            data = np.frombuffer(raw, dtype=dtype, count=count * components, offset=offset)
        else:
            rows = np.lib.stride_tricks.as_strided(
                np.frombuffer(raw, dtype=np.uint8, count=stride * (count - 1) + components * dtype.itemsize, offset=offset),
                shape=(count, components * dtype.itemsize),
                strides=(stride, 1),
            )
            data = np.ascontiguousarray(rows).view(dtype).reshape(-1)
        if accessor.get("normalized"):
            data = data.astype(np.float32) / float(np.iinfo(dtype).max)
        return data.reshape(count, components) if components > 1 else data

    def accessor_range(self, index: int) -> tuple[list[float], list[float]] | None:
        accessor = self.document["accessors"][index]
        if "min" in accessor and "max" in accessor:
            return accessor["min"], accessor["max"]
        return None

    def _lightmap_range(self, index: int) -> bool:
        """Whether a uv accessor could hold lightmap coordinates: in [0, 1] and not all zero."""
        extent = self.accessor_range(index)
        return extent is None or (
            any(v != 0 for v in extent[0] + extent[1])
            and min(extent[0]) >= -_LIGHTMAP_UV_EPSILON
            and max(extent[1]) <= 1.0 + _LIGHTMAP_UV_EPSILON
        )

    def _chart_score(self, uv: np.ndarray, triangles: np.ndarray) -> float:
        """Share of triangles that sit inside one lightmap chart (Source2Viewer's debug_chart_color export),
        sampled at the centroid and towards each corner; 0 when the chart image is missing."""
        if self._charts is None:
            path = self.path.parent / "lightmaps" / "debug_chart_color.png"
            if not path.exists():
                self._charts = np.zeros((0, 0), dtype=np.uint32)
            else:
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint32)
                self._charts = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        charts = self._charts
        if charts.size == 0 or len(triangles) < 3:
            return 0.0
        corners = uv[triangles.reshape(-1, 3)].astype(np.float64)
        centroid = corners.mean(axis=1)
        h, w = charts.shape

        def sample(points: np.ndarray) -> np.ndarray:
            x = np.clip((points[:, 0] * w).astype(np.int64), 0, w - 1)
            y = np.clip((points[:, 1] * h).astype(np.int64), 0, h - 1)
            return charts[y, x]

        reference = sample(centroid)
        inside = reference != 0
        for corner in range(3):
            inside &= sample(centroid * 0.6 + corners[:, corner] * 0.4) == reference
        return float(inside.mean())

    def _world_matrices(self) -> dict[int, np.ndarray]:
        nodes = self.document["nodes"]
        world: dict[int, np.ndarray] = {}

        def local(node: dict[str, Any]) -> np.ndarray:
            if "matrix" in node:
                return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
            x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
            rotation = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            matrix = np.eye(4)
            matrix[:3, :3] = rotation * np.array(node.get("scale", [1.0, 1.0, 1.0]))
            matrix[:3, 3] = node.get("translation", [0.0, 0.0, 0.0])
            return matrix

        stack = [(root, np.eye(4)) for root in self.document["scenes"][self.document.get("scene", 0)]["nodes"]]
        while stack:
            index, parent = stack.pop()
            matrix = parent @ local(nodes[index])
            world[index] = matrix
            for child in nodes[index].get("children", []):
                stack.append((child, matrix))
        return world

    def mesh_nodes(self) -> Iterator[tuple[int, dict[str, Any]]]:
        for index, node in enumerate(self.document["nodes"]):
            if "mesh" in node and index in self._world:
                yield index, node

    def pose_nodes(self, corrections: dict[str, np.ndarray], nodes: Iterable[int]) -> int:
        """Applies glTF-space model corrections (see bindpose) to the given mesh nodes; returns how many changed."""
        changed = 0
        for index in nodes:
            correction = corrections.get(mesh_key(str(self.document["nodes"][index].get("name", ""))))
            if correction is not None and index in self._world:
                self._world[index] = self._world[index] @ correction
                changed += 1
        return changed

    def node_origin(self, index: int) -> np.ndarray:
        """A node's placement origin in Source world inches."""
        return _SOURCE_FROM_GLTF @ self._world[index][:3, 3] * INCHES_PER_METER

    def primitives(self) -> Iterator[Primitive]:
        """Yields every drawn primitive with compacted vertices in Source world coordinates."""
        meshes = self.document["meshes"]
        for node_index, node in self.mesh_nodes():
            world = self._world[node_index]
            # Column-vector transform from glTF-local metres to Source world inches.
            linear = _SOURCE_FROM_GLTF @ world[:3, :3] * INCHES_PER_METER
            translation = _SOURCE_FROM_GLTF @ world[:3, 3] * INCHES_PER_METER
            normal_matrix = np.linalg.inv(linear).T
            # Placement in Source model space, used for ScaleTexCoordByModelScaleAxis.
            source_placement = linear @ _SOURCE_FROM_GLTF.T * 0.0254
            node_scale = np.linalg.norm(source_placement, axis=0)
            mesh = meshes[node["mesh"]]
            probe_lit = mesh.get("name") in self.probe_lit
            for primitive in mesh["primitives"]:
                if primitive.get("mode", 4) != 4 or "material" not in primitive:
                    continue
                yield self._primitive(node_index, primitive, linear, translation, normal_matrix, node_scale, probe_lit)

    def _primitive(
        self,
        node_index: int,
        primitive: dict[str, Any],
        linear: np.ndarray,
        translation: np.ndarray,
        normal_matrix: np.ndarray,
        node_scale: np.ndarray,
        probe_lit: bool = False,
    ) -> Primitive:
        attributes = primitive["attributes"]
        indices = self.accessor(primitive["indices"]).astype(np.uint32)
        used, remapped = np.unique(indices, return_inverse=True)

        def take(name: str) -> np.ndarray | None:
            return self.accessor(attributes[name])[used] if name in attributes else None

        local_positions = take("POSITION").astype(np.float64)
        local_normals = take("NORMAL").astype(np.float64)
        if is_overlay_material(self.materials[primitive["material"]]):
            local_positions = local_positions - local_normals * _OVERLAY_NORMAL_OFFSET
        positions = local_positions @ linear.T + translation
        normals = local_normals @ normal_matrix.T
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        tangents_raw = take("TANGENT")
        if tangents_raw is None:
            tangents_raw = np.zeros((len(used), 4), dtype=np.float32)
            tangents_raw[:, 0] = 1.0
            tangents_raw[:, 3] = 1.0
        tangent_xyz = tangents_raw[:, :3].astype(np.float64) @ linear.T
        tangent_xyz /= np.maximum(np.linalg.norm(tangent_xyz, axis=1, keepdims=True), 1e-12)
        handedness = np.sign(tangents_raw[:, 3]) * np.sign(np.linalg.det(linear))
        handedness[handedness == 0] = 1.0
        tangents = np.concatenate([tangent_xyz, handedness[:, None]], axis=1).astype(np.float32)

        texcoords = sorted(
            (int(name.split("_")[1]), name) for name in attributes if name.startswith("TEXCOORD_")
        )
        uv0 = take("TEXCOORD_0")
        if uv0 is None:
            uv0 = np.zeros((len(used), 2), dtype=np.float32)
        # The exporter numbers TEXCOORDs per vertex buffer: [uv, secondary uv?, lightmap uv?], and the vertex
        # layout follows the material, so a material that reads a secondary uv always has it as TEXCOORD_1.
        # The lightmap set lies in [0, 1]; an all-zero range means no baked lighting. Some meshes carry an unused
        # secondary uv, which the range test keeps from being read as a lightmap, and some carry an unused extra
        # [0, 1] set after the lightmap one, which only the lightmap's chart layout can tell apart.
        secondary = _uses_secondary_uv(self.materials[primitive["material"]])
        candidates = [] if probe_lit else [
            name for _, name in texcoords[2 if secondary else 1 :] if self._lightmap_range(attributes[name])
        ]
        last_name = texcoords[-1][1] if texcoords else None
        lightmap_name = None
        lightmap_uv = None
        if candidates == [last_name]:
            lightmap_name = last_name
            lightmap_uv = take(last_name)
        elif candidates:
            options = [(name, take(name)) for name in candidates]
            scores = [self._chart_score(uv, remapped) for _, uv in options]
            best = max(range(len(options)), key=lambda i: (scores[i], i))
            if options[best][0] == last_name or scores[best] >= _CHART_MATCH_MIN:
                lightmap_name, lightmap_uv = options[best]
            elif last_name in candidates:
                lightmap_name, lightmap_uv = options[candidates.index(last_name)]
        uv1 = None
        rest = [name for _, name in texcoords[1:] if name != lightmap_name]
        if rest:
            uv1 = take(rest[0])
        color = take("COLOR_0")
        blend = take("_TEXCOORD_4")
        if color is not None and color.shape[1] == 3:
            color = np.concatenate([color, np.ones((len(color), 1), dtype=color.dtype)], axis=1)
        return Primitive(
            node=node_index,
            material=primitive["material"],
            positions=positions,
            normals=normals.astype(np.float32),
            tangents=tangents,
            uv0=np.asarray(uv0, dtype=np.float32),
            uv1=None if uv1 is None else np.asarray(uv1, dtype=np.float32),
            lightmap_uv=None if lightmap_uv is None else np.asarray(lightmap_uv, dtype=np.float32),
            color=None if color is None else np.asarray(color, dtype=np.float32),
            blend=None if blend is None else np.asarray(blend, dtype=np.float32),
            indices=remapped.astype(np.uint32),
            node_scale=node_scale,
        )

    def image_path(self, vtex_name: str) -> Path:
        return self.path.parent / vtex_name.replace(".vtex", ".png")
