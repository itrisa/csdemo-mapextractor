"""Reference-pose corrections for skinned entity models.

Source2Viewer's map export places entity model meshes without their skeleton, so their vertices stay in bind space.
CS2 draws them in the model's reference animation pose, which can differ: Nuke's `metal_door_001_br` binds its only
bone at -90 degrees yaw while its `ref` pose is identity, so the exported door hangs 90 degrees open. Exporting the
model once with animations gives, per mesh, `pose(joint) @ inverse bind matrix`; GltfWorld applies it in glTF space.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np

_TOLERANCE = 1e-3
_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}


def _trs(translation: Iterable[float], rotation: Iterable[float], scale: Iterable[float]) -> np.ndarray:
    x, y, z, w = rotation
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    ) * np.array(list(scale))
    matrix[:3, 3] = list(translation)
    return matrix


def mesh_key(node_name: str) -> str:
    """`<model stem>.<mesh>`, the node name the map export uses, from either export's node name."""
    model, separator, mesh = node_name.replace("\\", "/").partition(".vmdl_c.")
    return f"{PurePosixPath(model).name}.{mesh}".lower() if separator else node_name.lower()


def corrections_from_gltf(path: Path) -> dict[str, np.ndarray]:
    """Rigid glTF-space corrections of a model export's skinned meshes, skipping identity and multi-pose meshes."""
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    buffer = (path.parent / document["buffers"][0]["uri"]).read_bytes() if document.get("buffers") else b""

    def accessor(index: int) -> np.ndarray:
        info = document["accessors"][index]
        view = document["bufferViews"][info["bufferView"]]
        width = _COMPONENTS[info["type"]]
        data = np.frombuffer(
            buffer,
            dtype=_DTYPES[info["componentType"]],
            count=info["count"] * width,
            offset=view.get("byteOffset", 0) + info.get("byteOffset", 0),
        )
        return data.reshape(info["count"], width)

    nodes: list[dict[str, Any]] = document["nodes"]
    pose: dict[int, dict[str, list[float]]] = {}
    animations = document.get("animations") or []
    reference = next((a for a in animations if a.get("name") == "ref"), animations[0] if animations else None)
    for channel in (reference or {}).get("channels", []):
        target = channel["target"]
        sampler = reference["samplers"][channel["sampler"]]
        pose.setdefault(target["node"], {})[target["path"]] = accessor(sampler["output"])[0].astype(float).tolist()

    def local(index: int) -> np.ndarray:
        node = nodes[index]
        override = pose.get(index, {})
        if "matrix" in node and not override:
            return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
        return _trs(
            override.get("translation", node.get("translation", [0.0, 0.0, 0.0])),
            override.get("rotation", node.get("rotation", [0.0, 0.0, 0.0, 1.0])),
            override.get("scale", node.get("scale", [1.0, 1.0, 1.0])),
        )

    # Joint poses relative to the model root, which the map export replaces with the entity placement.
    world: dict[int, np.ndarray] = {}
    roots = document["scenes"][document.get("scene", 0)]["nodes"]
    stack = [(child, np.eye(4)) for root in roots for child in nodes[root].get("children", [])]
    while stack:
        index, parent = stack.pop()
        world[index] = parent @ local(index)
        stack.extend((child, world[index]) for child in nodes[index].get("children", []))

    corrections: dict[str, np.ndarray] = {}
    for node in nodes:
        if "mesh" not in node or "skin" not in node:
            continue
        skin = document["skins"][node["skin"]]
        joints = skin["joints"]
        inverse_binds = (
            accessor(skin["inverseBindMatrices"]).astype(np.float64).reshape(-1, 4, 4).transpose(0, 2, 1)
            if "inverseBindMatrices" in skin
            else np.repeat(np.eye(4)[None], len(joints), axis=0)
        )
        used: set[int] = set()
        for primitive in document["meshes"][node["mesh"]]["primitives"]:
            attributes = primitive["attributes"]
            if "JOINTS_0" not in attributes:
                continue
            indices = accessor(attributes["JOINTS_0"]).astype(np.int64)
            weights = accessor(attributes["WEIGHTS_0"]).astype(np.float64) if "WEIGHTS_0" in attributes else None
            used.update(np.unique(indices[weights > 0] if weights is not None else indices[:, 0]).tolist())
        matrices = [world.get(joints[j], np.eye(4)) @ inverse_binds[j] for j in sorted(used) if j < len(joints)]
        if not matrices or any(not np.allclose(m, matrices[0], atol=_TOLERANCE) for m in matrices[1:]):
            continue
        if not np.allclose(matrices[0], np.eye(4), atol=_TOLERANCE):
            corrections[mesh_key(str(node.get("name", "")))] = matrices[0]
    return corrections


def _export(executable: Path, vpk: Path, resources: list[str], out_dir: Path) -> None:
    subprocess.run(
        [
            str(executable), "-i", str(vpk), "-f", ",".join(resources), "-e", "vmdl_c", "-o", str(out_dir), "-d",
            "--gltf_export_format", "gltf", "--gltf_export_animations",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=600,
    )


def load(
    executable: Path, cs2_dir: Path, map_name: str, models: Iterable[str], work_dir: Path
) -> dict[str, np.ndarray]:
    """Corrections keyed by map-export mesh node name for the given `models/...vmdl` paths, cached in work_dir."""
    wanted = sorted({str(m).replace("\\", "/").lower() for m in models if str(m).lower().startswith("models/")})
    cache = work_dir / "bind_corrections.json"
    if cache.is_file():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("models") == wanted:
            return {key: np.array(value) for key, value in cached["corrections"].items()}
    corrections: dict[str, np.ndarray] = {}
    if wanted:
        export_dir = work_dir / "bindpose"
        shutil.rmtree(export_dir, ignore_errors=True)
        resources = [f"{model.removesuffix('.vmdl')}.vmdl_c" for model in wanted]
        csgo = cs2_dir / "game" / "csgo"
        try:
            _export(executable, csgo / "pak01_dir.vpk", resources, export_dir)
            missing = [r for r in resources if not (export_dir / r).with_suffix(".gltf").is_file()]
            if missing:
                _export(executable, csgo / "maps" / f"{map_name}.vpk", missing, export_dir)
        except (subprocess.SubprocessError, OSError) as error:
            print(f"[cs2render] model reference poses unavailable for {map_name}: {error}", flush=True)
            return {}
        for resource in resources:
            path = (export_dir / resource).with_suffix(".gltf")
            if path.is_file():
                corrections.update(corrections_from_gltf(path))
        shutil.rmtree(export_dir, ignore_errors=True)
    cache.write_text(
        json.dumps({"models": wanted, "corrections": {k: v.tolist() for k, v in corrections.items()}}),
        encoding="utf-8",
    )
    return corrections
