"""Minimal GLB reading helpers for Source2Viewer render-world exports."""

from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np

COMPONENT_TYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
COMPONENT_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load(path: Path) -> tuple[dict, bytes | bytearray]:
    """Return the glTF JSON and one binary blob holding every buffer.

    A .gltf file's external buffers are concatenated and its buffer views rebased onto buffer 0, so
    callers read GLB and multi-buffer .gltf exports (needed for worlds of 2 GiB and more) alike."""
    path = Path(path)
    if path.suffix.lower() != ".gltf":
        data = path.read_bytes()
        json_length = struct.unpack_from("<I", data, 12)[0]
        doc = json.loads(data[20 : 20 + json_length])
        return doc, data[20 + json_length + 8 :]
    doc = json.loads(path.read_text(encoding="utf-8"))
    blob = bytearray()
    bases = []
    for buffer in doc.get("buffers", []):
        blob += b"\0" * (-len(blob) % 8)
        bases.append(len(blob))
        data = (path.parent / buffer["uri"]).read_bytes()
        if len(data) < buffer["byteLength"]:
            raise ValueError(f"glTF buffer {buffer['uri']} is truncated")
        blob += data[: buffer["byteLength"]]
    for view in doc.get("bufferViews", []):
        view["byteOffset"] = bases[view["buffer"]] + view.get("byteOffset", 0)
        view["buffer"] = 0
    doc["buffers"] = [{"byteLength": len(blob)}]
    return doc, blob


def accessor(doc: dict, binary: bytes, index: int, *, normalize: bool = True) -> np.ndarray:
    item = doc["accessors"][index]
    view = doc["bufferViews"][item["bufferView"]]
    dtype = np.dtype(COMPONENT_TYPES[item["componentType"]])
    components = COMPONENT_COUNTS[item["type"]]
    offset = view.get("byteOffset", 0) + item.get("byteOffset", 0)
    stride = view.get("byteStride", dtype.itemsize * components)
    count = item["count"]
    raw = np.frombuffer(
        binary,
        dtype=np.uint8,
        count=stride * (count - 1) + dtype.itemsize * components,
        offset=offset,
    )
    out = np.lib.stride_tricks.as_strided(
        raw, shape=(count, dtype.itemsize * components), strides=(stride, 1)
    ).copy()
    out = out.view(dtype).reshape(count, components)
    if normalize and item.get("normalized"):
        out = out.astype(np.float32) / np.iinfo(dtype).max
    return out


def node_matrices(doc: dict) -> dict[int, np.ndarray]:
    matrices: dict[int, np.ndarray] = {}

    def local(node: dict) -> np.ndarray:
        if "matrix" in node:
            return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
        m = np.eye(4)
        if "scale" in node:
            m = np.diag([*node["scale"], 1.0]) @ m
        if "rotation" in node:
            x, y, z, w = node["rotation"]
            r = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            rm = np.eye(4)
            rm[:3, :3] = r
            m = rm @ m
        if "translation" in node:
            t = np.eye(4)
            t[:3, 3] = node["translation"]
            m = t @ m
        return m

    def walk(index: int, parent: np.ndarray) -> None:
        m = parent @ local(doc["nodes"][index])
        matrices[index] = m
        for child in doc["nodes"][index].get("children", []):
            walk(child, m)

    for scene in doc.get("scenes", []):
        for index in scene["nodes"]:
            walk(index, np.eye(4))
    return matrices


def write(path: Path, doc: dict, binary: bytes) -> None:
    js = json.dumps(doc, separators=(",", ":"), sort_keys=True).encode("utf-8")
    js += b" " * (-len(js) % 4)
    binary = binary + b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<III", 0x46546C67, 2, total))
        handle.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        handle.write(struct.pack("<II", len(binary), 0x004E4942) + binary)
