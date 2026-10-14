"""Self-contained colouring inputs for one map.

A bundle holds everything the colouriser needs, so it can run on a machine without CS2:

    bundle.json      map name, layer list and texture table
    collision.glb    the published collision GLB (layers are read back from it)
    world.glb        the Source2Viewer render world reduced to the primitives, attributes and material
                     parameters the albedo evaluation reads
    textures/*.png   level 0 of every texture those materials sample, already reduced to at most
                     albedo.MAX_TEX pixels exactly as albedo.png_level0 does (stored sRGB-encoded for
                     colour textures, so 8-bit storage round-trips losslessly when no reduction was needed)

All files are written deterministically, so the bundle digest identifies the colouriser input and an
unchanged map can reuse its previously published coloured GLB.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .. import glb as collision_glb
from . import albedo as A
from . import gltf as G

FORMAT = "csdemo-color-bundle-v1"
_ATTRIBUTES = ("POSITION", "TEXCOORD_0", "TEXCOORD_1", "COLOR_0", "_TEXCOORD_4")
_NODE_FIELDS = ("mesh", "children", "matrix", "translation", "rotation", "scale")


def pipeline_version() -> str:
    """Hash of the colouring code; any change to it invalidates previously coloured maps."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:16]


def digest(bundle_dir: Path) -> str:
    bundle_dir = Path(bundle_dir)
    total = hashlib.sha256(f"{FORMAT}\n{pipeline_version()}\n".encode())
    for path in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
        total.update(path.relative_to(bundle_dir).as_posix().encode() + b"\0")
        total.update(hashlib.sha256(path.read_bytes()).digest())
    return total.hexdigest()


def create(bundle_dir: Path, map_name: str, collision: bytes, render_dir: Path) -> str:
    """Write the bundle for map_name from its collision GLB and a Source2Viewer render export directory
    (holding world.gltf or world.glb and the exported PNG textures). Returns the bundle digest."""
    bundle_dir = Path(bundle_dir)
    render_dir = Path(render_dir)
    (bundle_dir / "textures").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "collision.glb").write_bytes(collision)
    source = render_dir / "world.gltf"
    if not source.is_file():
        source = render_dir / "world.glb"
    materials = _slim_world(source, bundle_dir / "world.glb")

    requests = sorted({request for material in materials for request in material.texture_requests()})
    textures = []
    for name, srgb in requests:
        level0 = A.png_level0(render_dir, name, srgb)
        if level0 is None:
            continue
        image, base, size = level0
        file = f"textures/{len(textures):05d}.png"
        (bundle_dir / file).write_bytes(_encode_png(image, srgb))
        textures.append({"name": name, "srgb": srgb, "file": file, "base": base, "size": list(size)})

    index = {
        "format": FORMAT,
        "map": map_name,
        "layers": [name for name, *_ in read_layers(collision)],
        "textures": textures,
    }
    (bundle_dir / "bundle.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return digest(bundle_dir)


def load(bundle_dir: Path, log=print):
    """Returns (map name, collision layers [(name, tags, P, F)], zero-argument RenderWorld loader).

    The render world is by far the largest input, so callers open it only once they need it."""
    bundle_dir = Path(bundle_dir)
    index = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    if index.get("format") != FORMAT:
        raise ValueError(f"unsupported colour bundle format: {index.get('format')!r}")
    table = {(t["name"], t["srgb"]): t for t in index["textures"]}

    def loader(name, srgb):
        entry = table.get((name, srgb))
        if entry is None:
            return None
        with Image.open(bundle_dir / entry["file"]) as image:
            a = np.asarray(image.convert("RGBA"), np.float32) / 255.0
        if srgb:
            a[..., :3] = A.s2l(a[..., :3]).astype(np.float32)
        return a, entry["base"], tuple(entry["size"])

    layers = read_layers((bundle_dir / "collision.glb").read_bytes())
    return index["map"], layers, lambda: A.RenderWorld(bundle_dir / "world.glb", A.TextureCache(loader), log)


def read_layers(data: bytes):
    """Collision layers of a csdemo collision GLB as [(name, tags, positions float64, triangles int64)]."""
    document, binary = collision_glb._parse(data)
    layers = []
    for node in document["nodes"]:
        primitive = document["meshes"][node["mesh"]]["primitives"][0]
        positions = _decode(document, binary, primitive["attributes"]["POSITION"], "<f4", 3)
        indices = _decode(document, binary, primitive["indices"], None, 1)
        layers.append(
            (
                node["name"],
                list(node.get("extras", {}).get("collision_tags", [])),
                positions.astype(np.float64),
                indices.reshape(-1, 3).astype(np.int64),
            )
        )
    return layers


def _decode(document, binary, accessor_index, dtype, components):
    accessor = document["accessors"][accessor_index]
    dtype = dtype or {5123: "<u2", 5125: "<u4"}[accessor["componentType"]]
    data = collision_glb._view_data(binary, document["bufferViews"][accessor["bufferView"]])
    return np.frombuffer(
        data, dtype, accessor["count"] * components, accessor.get("byteOffset", 0)
    ).reshape(accessor["count"], components)


def _encode_png(image: np.ndarray, srgb: bool) -> bytes:
    a = image.astype(np.float64)
    if srgb:
        a[..., :3] = A.l2s(a[..., :3])
    pixels = np.round(np.clip(a, 0, 1) * 255).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(out, "PNG")
    return out.getvalue()


def _slim_world(source: Path, target: Path):
    """Copy the render world keeping only what RenderWorld reads for non-skipped triangle primitives.

    Primitive order and per-primitive triangle order are preserved, so triangle ids match the full export.
    Returns the Material objects of the materials that are still referenced."""
    doc, binary = G.load(source)
    materials = [A.Material(m, None) for m in doc.get("materials", [])]
    chunks: list[bytes] = []
    views: list[dict] = []
    accessors: list[dict] = []
    remapped: dict[int, int] = {}
    offset = 0

    def copy(index: int) -> int:
        nonlocal offset
        if index in remapped:
            return remapped[index]
        source_accessor = doc["accessors"][index]
        data = np.ascontiguousarray(G.accessor(doc, binary, index, normalize=False)).tobytes()
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data)})
        chunks.append(data + b"\0" * (-len(data) % 4))
        offset += len(chunks[-1])
        accessor = {
            "bufferView": len(views) - 1,
            "componentType": source_accessor["componentType"],
            "count": source_accessor["count"],
            "type": source_accessor["type"],
        }
        if source_accessor.get("normalized"):
            accessor["normalized"] = True
        accessors.append(accessor)
        remapped[index] = len(accessors) - 1
        return remapped[index]

    used_materials: set[int] = set()
    meshes = []
    for mesh in doc.get("meshes", []):
        primitives = []
        for primitive in mesh["primitives"]:
            if primitive.get("mode", 4) != 4 or "material" not in primitive:
                continue
            if materials[primitive["material"]].skip:
                continue
            attributes = {
                name: copy(primitive["attributes"][name])
                for name in _ATTRIBUTES
                if name in primitive["attributes"]
            }
            if "TEXCOORD_2" in primitive["attributes"]:
                # RenderWorld only checks that a third UV set exists; point it at UV1 instead of storing it
                attributes["TEXCOORD_2"] = attributes.get("TEXCOORD_1", copy(primitive["attributes"]["TEXCOORD_2"]))
            slim = {"attributes": attributes, "material": primitive["material"]}
            if "indices" in primitive:
                slim["indices"] = copy(primitive["indices"])
            primitives.append(slim)
            used_materials.add(primitive["material"])
        meshes.append({"primitives": primitives})

    slim_materials = []
    for index, material in enumerate(doc.get("materials", [])):
        if index not in used_materials:
            slim_materials.append({})
            continue
        entry = {"extras": {"vmat": material["extras"]["vmat"]}}
        for key in ("name", "doubleSided"):
            if key in material:
                entry[key] = material[key]
        factor = material.get("pbrMetallicRoughness", {}).get("baseColorFactor")
        if factor is not None:
            entry["pbrMetallicRoughness"] = {"baseColorFactor": factor}
        slim_materials.append(entry)

    nodes = [{key: node[key] for key in _NODE_FIELDS if key in node} for node in doc.get("nodes", [])]
    slim_binary = b"".join(chunks)
    slim_doc = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": scene["nodes"]} for scene in doc.get("scenes", [])],
        "nodes": nodes,
        "meshes": meshes,
        "materials": slim_materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(slim_binary)}],
    }
    G.write(target, slim_doc, slim_binary)
    return [materials[index] for index in sorted(used_materials)]

