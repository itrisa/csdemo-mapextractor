"""Builds a compact CS2-look render package for one map.

Output layout (out_dir):
  map.json                 manifest: lighting, post, probes, texture classes, chunk table
  materials.bin            float32 material table, MATERIAL_TEXELS RGBA texels per material
  lut.bin                  32^3 RGBA8 colour-correction LUT (gamma-space input)
  probes.ctex              BC6H octahedral reflection probe array
  geo/geo_X_Y.bin          meshopt-compressed chunks per spatial cell, each chunk with an optional far LOD
  visibility.bin           CS2 voxel-PVS octree and per-cluster visible-cell bitsets (see visibility.py)
  lightgrid.bin            gzip light grids for dynamic objects: ambient cube + sun visibility (see lightgrid.py)
  tex/                     texture arrays (*.ctex), lightmaps and the sky panorama (sky.ctex, see sky.py)

Every file except map.json is gzip-compressed (manifest "compression": "gzip"); the client inflates them with
DecompressionStream, so any static host serves them at their compressed size.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import source2viewer
from . import bindpose
from . import breakables
from . import geometry as geo
from . import lighting
from . import lightgrid
from . import objects as entity_objects
from . import sky
from . import textures as tx
from . import visibility
from . import worldnodes
from .gltf_world import GltfWorld, Primitive
from .kv3 import entities as parse_entities
from .materials import ALPHATEST, BLEND, TRANSLUCENT, LayerBake, MaterialSpec, classify
from .probes import ProbeVolumes

MATERIAL_TEXELS = 8
MAX_ARRAY_LAYERS = 256
CELL_SIZE = 2048.0
DETAIL_DIAGONAL = 72.0
OVERLAY_SIZE = 128
TEXTURE_DIR = "tex"


@dataclass
class BuildOptions:
    texels_per_inch: float = 2.0
    min_size: int = 64
    max_size: int = 1024
    source_scale: float = 0.5  # cap each texture at this fraction of its source resolution
    detail_scale: float = 0.5  # normal / mask resolution relative to colour
    lightmap_size: int = 2048
    probe_size: int = 128
    probe_levels: int = 6
    workers: int = 8
    lod_error: float = 4.0  # absolute simplification error of each chunk's far LOD, in inches
    lod_attribute_scale: float = 0.25  # weight of UV / lightmap / paint changes in the far LOD's error metric
    compress: bool = True  # gzip every package file except map.json
    geometry_only: bool = False  # reuse an existing package and rebuild only the geometry
    light_grid_only: bool = False  # reuse an existing package and rebuild only the dynamic-object light grid
    sky_only: bool = False  # reuse an existing package and rebuild only the sky panorama
    breaks_only: bool = False  # reuse an existing package and rebuild only the breakables' pieces


def rebuild_sky(
    executable: Path, cs2_dir: Path, entities: list[dict[str, Any]], work_dir: Path, out_dir: Path, texconv: Path
) -> dict[str, Any]:
    manifest = json.loads((out_dir / "map.json").read_text(encoding="utf-8"))
    _log("sky")
    entry = sky.build_sky(executable, cs2_dir, entities, work_dir, out_dir, texconv)
    if entry is None:
        manifest.pop("sky", None)
    else:
        manifest["sky"] = entry
        if manifest.get("compression") == "gzip":
            _compress_file(out_dir / entry["file"])
    geo.dump(out_dir / "map.json", manifest)
    return {"sky": entry}


def _vpk(cs2_dir: Path) -> Path:
    return cs2_dir / "game" / "csgo" / "pak01_dir.vpk"


def rebuild_breaks(
    executable: Path, cs2_dir: Path, entities: list[dict[str, Any]], work_dir: Path, out_dir: Path
) -> dict[str, Any]:
    manifest = json.loads((out_dir / "map.json").read_text(encoding="utf-8"))
    breakables.attach(executable, _vpk(cs2_dir), manifest.get("objects", []), entities, work_dir, out_dir)
    geo.dump(out_dir / "map.json", manifest)
    return {"objects": len(manifest.get("objects", []))}


def rebuild_light_grid(map_dir: Path, entities: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    manifest = json.loads((out_dir / "map.json").read_text(encoding="utf-8"))
    _log("light grid")
    entry = lightgrid.write(
        ProbeVolumes(map_dir / "lightmaps", entities),
        np.array(manifest["bounds"]["min"]),
        np.array(manifest["bounds"]["max"]),
        out_dir / "lightgrid.bin",
    )
    if entry is None:
        manifest.pop("lightGrid", None)
    else:
        manifest["lightGrid"] = {key: value for key, value in entry.items() if key != "bytes"}
    geo.dump(out_dir / "map.json", manifest)
    return {"lightGrid": entry}


def _log(message: str) -> None:
    print(f"[cs2render {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _pow2(value: float) -> int:
    return 1 << max(0, int(round(math.log2(max(value, 1.0)))))


def _srgb(color: Any) -> list[float]:
    values = np.array([float(x) for x in list(color)[:3]], dtype=np.float64) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    return [round(float(x), 5) for x in linear]


# --- Export (cached in work_dir) ---


def export_sources(executable: Path, cs2_dir: Path, map_name: str, work_dir: Path) -> dict[str, Path]:
    export_dir = work_dir / "export"
    map_dir = export_dir / "maps" / map_name
    if not (map_dir / "world.gltf").is_file():
        _log("exporting render world")
        source2viewer.export_render_world(executable, cs2_dir, map_name, export_dir, extras=True)
    for folder in ("lightmaps", "cubemaps"):
        if not (map_dir / folder).is_dir():
            _log(f"exporting {folder}")
            source2viewer.export_map_folder(executable, cs2_dir, map_name, folder, export_dir)
    entities_path = work_dir / "entities.kv3"
    if not entities_path.is_file():
        _log("extracting entities")
        entities_path.write_text(source2viewer.extract_entity_data(executable, cs2_dir, map_name), encoding="utf-8")
    return {"map": map_dir, "entities": entities_path}


def _postprocess_text(executable: Path, cs2_dir: Path, resource: str, work_dir: Path) -> str | None:
    cached = work_dir / "postprocess.kv3"
    if cached.is_file():
        return cached.read_text(encoding="utf-8")
    vpk = cs2_dir / "game" / "csgo" / "pak01_dir.vpk"
    try:
        text = source2viewer.extract_resource_block(executable, vpk, resource + "_c")
    except Exception as error:  # noqa: BLE001 - post-processing is optional
        _log(f"post-processing volume unavailable: {error}")
        return None
    cached.write_text(text, encoding="utf-8")
    return text


# --- Scene parameters ---


def _number_block(text: str, block: str) -> dict[str, float]:
    match = re.search(rf"{block}\s*=\s*\{{(.*?)\}}", text, re.S)
    values: dict[str, float] = {}
    if match:
        for name, value in re.findall(r"(m_\w+)\s*=\s*(-?[\d.]+(?:e-?\d+)?)\s*$", match.group(1), re.M):
            values[name] = float(value)
    return values


# Hand-tuned against in-game captures where the authored minimum looks too bright.
EXPOSURE_OVERRIDES: dict[str, float] = {"de_cache": 0.75}


def daylight_exposure(entity: dict[str, Any] | None) -> float:
    """Exposure CS2's auto-exposure settles on outdoors. It clamps target / scene luminance with min then max
    (VRF PostProcessRenderer); a sunlit scene's raw scalar is below every authored range, so it lands on the
    minimum, inverted ranges included."""
    if not entity:
        return 1.0
    exposure = 1.0
    if entity.get("enableexposure"):
        if "minlogexposure" in entity:
            exposure = 2.0 ** float(entity["minlogexposure"])
        else:
            exposure = float(entity.get("minexposure", 0.25))
    return exposure * 2.0 ** float(entity.get("exposurecompensation", 0.0))


def postprocess_params(text: str | None, entity: dict[str, Any] | None, out_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"exposure": round(daylight_exposure(entity), 4)}
    if not text:
        return result
    tonemap = _number_block(text, "m_toneMapParams")
    if tonemap:
        result["tonemap"] = {
            "exposureBias": tonemap.get("m_flExposureBias", 0.0),
            "shoulderStrength": tonemap.get("m_flShoulderStrength", 0.15),
            "linearStrength": tonemap.get("m_flLinearStrength", 0.5),
            "linearAngle": tonemap.get("m_flLinearAngle", 0.1),
            "toeStrength": tonemap.get("m_flToeStrength", 0.2),
            "toeNumerator": tonemap.get("m_flToeNum", 0.02),
            "toeDenominator": tonemap.get("m_flToeDenom", 0.3),
            "whitePoint": tonemap.get("m_flWhitePoint", 4.0),
        }
    bloom = _number_block(text, "m_bloomParams")
    if bloom:
        result["bloom"] = {
            "strength": bloom.get("m_flBloomStrength", 0.0),
            "threshold": bloom.get("m_flBloomThreshold", 1.0),
            "thresholdWidth": bloom.get("m_flBloomThresholdWidth", 1.0),
        }
    dim_match = re.search(r"m_nColorCorrectionVolumeDim\s*=\s*(\d+)", text)
    data_match = re.search(r"m_colorCorrectionVolumeData\s*=\s*#\[(.*?)\]", text, re.S)
    if dim_match and data_match:
        dim = int(dim_match.group(1))
        lut = bytes.fromhex(re.sub(r"\s+", "", data_match.group(1)))
        if len(lut) == dim**3 * 4:
            (out_dir / "lut.bin").write_bytes(lut)
            result["lut"] = {"file": "lut.bin", "size": dim}
    return result


def scene_params(entities: list[dict[str, Any]]) -> dict[str, Any]:
    def first(classname: str) -> dict[str, Any] | None:
        return next((e for e in entities if e.get("classname") == classname), None)

    scene: dict[str, Any] = {}
    sun = first("light_environment")
    if sun:
        forward = lighting._angles_matrix(tuple(lighting._vec(sun.get("angles"))))[:, 0]
        brightness = float(sun.get("brightness", 1.0)) * float(sun.get("brightnessscale", 1.0))
        scene["sun"] = {
            "direction": [round(float(-x), 6) for x in forward],  # towards the sun
            "color": [round(c * brightness, 5) for c in _srgb(sun.get("color", [255, 255, 255]))],
            "skyColor": [round(c * float(sun.get("skyintensity", 1.0)), 5) for c in _srgb(sun.get("skycolor", [0, 0, 0]))],
        }
    fog = first("env_gradient_fog")
    if fog and not fog.get("StartDisabled"):
        scene["fog"] = {
            "color": _srgb(fog.get("fogcolor", [255, 255, 255])),
            "start": float(fog.get("fogstart", 0.0)),
            "end": float(fog.get("fogend", 10000.0)),
            "falloffExponent": float(fog.get("fogfalloffexponent", 1.0)),
            "maxOpacity": float(fog.get("fogmaxopacity", 1.0)) * float(fog.get("fogstrength", 1.0)),
            "height": bool(fog.get("heightfog", False)),
            "heightStart": float(fog.get("fogstartheight", 0.0)),
            "heightEnd": float(fog.get("fogendheight", 0.0)),
            "verticalExponent": float(fog.get("fogverticalexponent", 1.0)),
        }
    return scene


# --- Texture density and classes ---


def _layer_uvs(primitive: Primitive, spec: MaterialSpec) -> list[np.ndarray]:
    uvs = [geo._apply_uv_scale(geo.layer_uv(primitive, spec.uv_sets[0]), spec.uv_scale_axes, primitive.node_scale)]
    if len(spec.layers) > 1:
        uvs.append(
            geo._apply_uv_scale(geo.layer_uv(primitive, spec.uv_sets[1]), spec.uv2_scale_axes, primitive.node_scale)
        )
    return uvs


def _inches_per_uv(positions: np.ndarray, uv: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = indices.reshape(-1, 3)
    p = positions[triangles]
    t = uv[triangles].astype(np.float64)
    world_area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    uv_area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    valid = (uv_area > 1e-12) & (world_area > 1e-4)
    return np.sqrt(world_area[valid] / uv_area[valid]), world_area[valid]


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, q * cumulative[-1])])


@dataclass
class TextureClass:
    id: str
    size: int  # colour size
    alpha: bool
    keys: list[str]


def plan_classes(
    primitives: list[Primitive],
    specs: list[MaterialSpec | None],
    source: tx.TextureSource,
    options: BuildOptions,
) -> tuple[dict[str, LayerBake], dict[str, TextureClass], dict[str, tuple[str, int]], dict[str, float]]:
    samples: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    layers: dict[str, LayerBake] = {}
    alpha_reference: dict[str, float] = {}
    for spec in specs:
        if spec is None:
            continue
        for layer in spec.layers:
            layers.setdefault(layer.key, layer)
            if layer.needs_alpha:
                alpha_reference.setdefault(layer.key, float(spec.params.get("alphaTestReference", 0.5)))
    for primitive in primitives:
        spec = specs[primitive.material]
        if spec is None:
            continue
        for layer, uv in zip(spec.layers, _layer_uvs(primitive, spec)):
            samples.setdefault(layer.key, []).append(_inches_per_uv(primitive.positions, uv, primitive.indices))

    sizes: dict[str, int] = {}
    for key, layer in layers.items():
        source_size = max(source.size(layer.color)) if layer.color else options.min_size
        chunks = samples.get(key)
        if chunks:
            values = np.concatenate([c[0] for c in chunks])
            weights = np.concatenate([c[1] for c in chunks])
            density = _weighted_quantile(values, weights, 0.9) if len(values) else 0.0
        else:
            density = 0.0
        size = _pow2(density * options.texels_per_inch)
        source_cap = max(int(source_size * options.source_scale), options.min_size)
        sizes[key] = int(min(max(size, options.min_size), options.max_size, source_cap))

    groups: dict[tuple[bool, int], list[str]] = {}
    for key, layer in layers.items():
        groups.setdefault((layer.needs_alpha, sizes[key]), []).append(key)
    classes: dict[str, TextureClass] = {}
    assignment: dict[str, tuple[str, int]] = {}
    for (alpha, size), keys in sorted(groups.items()):
        keys.sort()
        for part in range(0, len(keys), MAX_ARRAY_LAYERS):
            class_id = f"{'a' if alpha else 'o'}{size}" + (f"_{part // MAX_ARRAY_LAYERS}" if part else "")
            members = keys[part : part + MAX_ARRAY_LAYERS]
            classes[class_id] = TextureClass(class_id, size, alpha, members)
            for index, key in enumerate(members):
                assignment[key] = (class_id, index)
    return layers, classes, assignment, alpha_reference


# --- Texture baking ---


def _bake_set(
    job: tuple[LayerBake, str, int, int, float | None],
) -> tuple[str, tuple[list[bytes], list[bytes], list[bytes]]]:
    """Bakes and encodes one texture set's mip chains: (color, normal, mask)."""
    layer, directory, size, detail_size, alpha_reference = job
    source = tx.TextureSource(Path(directory))
    color = tx.mip_chain(tx.bake_color(layer, source, size), "color", alpha_reference if layer.needs_alpha else None)
    normal = tx.mip_chain(tx.bake_normal(layer, source, detail_size), "normal")
    mask = tx.mip_chain(tx.bake_mask(layer, source, detail_size), "mask")
    return layer.key, (
        tx.encode_color_chain(color, tx.FORMATS["color_alpha" if layer.needs_alpha else "color"]),
        tx.encode_normal_chain(normal, tx.FORMATS["normal"]),
        tx.encode_mask_chain(mask, tx.FORMATS["mask"]),
    )


def bake_textures(
    layers: dict[str, LayerBake],
    classes: dict[str, TextureClass],
    alpha_reference: dict[str, float],
    texture_dir: Path,
    out_dir: Path,
    options: BuildOptions,
) -> dict[str, dict[str, Any]]:
    """Bakes and writes every class's colour, normal and mask arrays; returns the manifest's class table."""
    class_files: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=options.workers) as pool:
        for class_id, texture_class in classes.items():
            started = time.time()
            size = texture_class.size
            detail = max(4, int(size * options.detail_scale))
            jobs = [
                (layers[key], str(texture_dir), size, detail, alpha_reference.get(key)) for key in texture_class.keys
            ]
            results = dict(pool.map(_bake_set, jobs, chunksize=2))
            ordered = [results[key] for key in texture_class.keys]
            color_format = tx.FORMATS["color_alpha"] if texture_class.alpha else tx.FORMATS["color"]
            files = {}
            for stream, fmt, base, index in (
                ("color", color_format, size, 0),
                ("normal", tx.FORMATS["normal"], detail, 1),
                ("mask", tx.FORMATS["mask"], detail, 2),
            ):
                name = f"{TEXTURE_DIR}/{class_id}_{stream[0]}.ctex"
                tx.ArrayTexture(fmt, base, base, [entry[index] for entry in ordered]).write(out_dir / name)
                files[stream] = name
            class_files[class_id] = {"size": size, "layers": len(ordered), **files}
            _log(f"textures {class_id}: {len(texture_class.keys)} sets at {size} in {time.time() - started:.1f}s")
    return class_files


def bake_overlays(
    specs: list[MaterialSpec | None], source: tx.TextureSource, out_dir: Path
) -> tuple[dict[str, int], dict[str, Any] | None]:
    names = sorted({spec.overlay for spec in specs if spec is not None and spec.overlay})
    if not names:
        return {}, None
    levels = [tx.mip_chain(source.load(name, OVERLAY_SIZE), "mask") for name in names]
    fmt = tx.FORMATS["overlay"]
    chains = [[tx.encode_level(level, fmt) for level in chain] for chain in levels]
    file = f"{TEXTURE_DIR}/overlay.ctex"
    tx.ArrayTexture(fmt, OVERLAY_SIZE, OVERLAY_SIZE, chains).write(out_dir / file)
    return {name: index for index, name in enumerate(names)}, {"file": file, "layers": len(chains)}


# --- Material table ---

FLAG_BLEND = 1
FLAG_ALPHATEST = 2
FLAG_TRANSLUCENT = 4
FLAG_COMPLEX = 8
FLAG_METAL_TEXTURE = 16
FLAG_SELF_ILLUM = 32
FLAG_PAINT_VERTEX_COLORS = 64
FLAG_GLASS = 128
FLAG_TRANSMISSIVE = 256
FLAG_MODEL_TINT_1 = 512
FLAG_MODEL_TINT_2 = 1024


def material_table(
    specs: list[MaterialSpec | None], assignment: dict[str, tuple[str, int]], overlays: dict[str, int]
) -> np.ndarray:
    table = np.zeros((max(1, len(specs)), MATERIAL_TEXELS, 4), dtype=np.float32)
    for index, spec in enumerate(specs):
        if spec is None:
            continue
        p = spec.params
        complex_shader = spec.layers[0].kind == "complex"
        flags = 0
        flags |= FLAG_BLEND if spec.family == BLEND else 0
        flags |= FLAG_ALPHATEST if spec.family == ALPHATEST else 0
        flags |= FLAG_TRANSLUCENT if spec.family == TRANSLUCENT else 0
        flags |= FLAG_COMPLEX if complex_shader else 0
        flags |= FLAG_METAL_TEXTURE if p.get("metalTexture") else 0
        flags |= FLAG_SELF_ILLUM if any(p.get("selfIllum", [0.0])) else 0
        flags |= FLAG_PAINT_VERTEX_COLORS if p.get("paintVertexColors") else 0
        flags |= FLAG_GLASS if p.get("glass") else 0
        flags |= FLAG_TRANSMISSIVE if p.get("transmissive") else 0
        model_tint_layers = p.get("modelTintLayers", [1.0, 1.0])
        flags |= FLAG_MODEL_TINT_1 if model_tint_layers[0] else 0
        flags |= FLAG_MODEL_TINT_2 if model_tint_layers[1] else 0
        layer2 = assignment[spec.layers[1].key][1] if len(spec.layers) > 1 else 0
        overlay = overlays.get(spec.overlay, -1) if spec.overlay else -1
        table[index, 0] = [assignment[spec.layers[0].key][1], layer2, overlay, flags]
        table[index, 1] = [*p.get("modelTint", [1.0, 1.0, 1.0]), p.get("modelTintAmount", 0.0)]
        table[index, 2] = [*p.get("vertexTint", [1.0, 1.0, 1.0]), p.get("alphaTestReference", 0.5)]
        table[index, 3] = [*p.get("heightScale", [1.0, 1.0]), *p.get("heightZero", [0.5, 0.5])]
        table[index, 4] = [p.get("blendSoftness", 0.01), *p.get("vertexColorMode", [0, 0]), 0.0]
        table[index, 5] = [*p.get("overlayContrast", [1.0, 1.0]), p.get("metalness", 0.0), p.get("opacityScale", 1.0)]
        table[index, 6] = [*p.get("selfIllum", [0.0, 0.0, 0.0]), p.get("selfIllumAlbedo", 0.0)]
        table[index, 7] = [*p.get("tint", [1.0, 1.0, 1.0]), 0.0]
    return table


# --- Build ---


def build(
    executable: Path,
    cs2_dir: Path,
    map_name: str,
    work_dir: Path,
    out_dir: Path,
    texconv: Path,
    options: BuildOptions | None = None,
) -> dict[str, Any]:
    options = options or BuildOptions()
    started = time.time()
    work_dir.mkdir(parents=True, exist_ok=True)
    previous: dict[str, Any] | None = None
    if options.light_grid_only or options.sky_only or options.breaks_only:
        pass
    elif options.geometry_only:
        previous = json.loads((out_dir / "map.json").read_text(encoding="utf-8"))
        shutil.rmtree(out_dir / "geo", ignore_errors=True)
    else:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

    sources = export_sources(executable, cs2_dir, map_name, work_dir)
    map_dir = sources["map"]
    entities = parse_entities(sources["entities"].read_text(encoding="utf-8"))
    _log(f"{len(entities)} entities")
    if options.light_grid_only:
        return rebuild_light_grid(map_dir, entities, out_dir)
    if options.sky_only:
        return rebuild_sky(executable, cs2_dir, entities, work_dir, out_dir, texconv)
    if options.breaks_only:
        return rebuild_breaks(executable, cs2_dir, entities, work_dir, out_dir)

    probe_lit = worldnodes.load(executable, cs2_dir, map_name, work_dir)
    _log(f"{len(probe_lit)} probe-lit aggregate fragments")
    world = GltfWorld(map_dir / "world.gltf", probe_lit)
    specs = [classify(index, material) for index, material in enumerate(world.materials)]
    texture_source = tx.TextureSource(map_dir)
    entity_layout = entity_objects.layout(world, entities)
    object_models = {e["model"] for e in entities if e.get("model") and entity_objects.object_kind(e)}
    corrections = bindpose.load(executable, cs2_dir, map_name, object_models, work_dir)
    posed = world.pose_nodes(corrections, entity_layout.node_objects)
    _log(f"{posed} dynamic object meshes moved to their reference pose")
    primitives = [primitive for primitive in world.primitives() if primitive.node not in entity_layout.hidden_nodes]
    _log(
        f"{len(entity_layout.hidden_nodes)} start-disabled entity meshes skipped, "
        f"{len(entity_layout.objects)} dynamic objects over {len(entity_layout.node_objects)} meshes"
    )
    _log(f"{len(primitives)} primitives, {sum(s is not None for s in specs)}/{len(specs)} materials kept")

    layers, classes, assignment, alpha_reference = plan_classes(primitives, specs, texture_source, options)
    _log(f"{len(layers)} texture sets in {len(classes)} classes: " + ", ".join(f"{c.id}x{len(c.keys)}" for c in classes.values()))

    if previous is None:
        textures: dict[str, Any] = {
            "classes": bake_textures(layers, classes, alpha_reference, map_dir, out_dir, options)
        }
        overlay_index, overlay_info = bake_overlays(specs, texture_source, out_dir)
        if overlay_info is not None:
            textures["overlay"] = overlay_info

        table = material_table(specs, assignment, overlay_index)
        (out_dir / "materials.bin").write_bytes(table.tobytes())

        _log(f"lightmaps at {options.lightmap_size}")
        result = lighting.pack_lightmaps(
            map_dir / "lightmaps", out_dir / TEXTURE_DIR, options.lightmap_size, options.lightmap_size * 2, texconv
        )
        textures["lightmaps"] = {name: f"{TEXTURE_DIR}/{file}" for name, file in result.files.items()}

        _log("reflection probes")
        probe_metadata, _ = lighting.pack_probes(
            map_dir / "cubemaps", entities, out_dir, options.probe_size, options.probe_levels, texconv
        )

        _log("sky")
        sky_entry = sky.build_sky(executable, cs2_dir, entities, work_dir, out_dir, texconv)
    else:
        textures = previous["textures"]
        probe_metadata = previous["probes"]["volumes"]
        sky_entry = previous.get("sky")

    _log("light probe volumes")
    probe_volumes = ProbeVolumes(map_dir / "lightmaps", entities)
    locator = lighting.ProbeLocator(probe_metadata)

    node_boxes: dict[int, list[np.ndarray]] = {}
    for primitive in primitives:
        box = node_boxes.setdefault(primitive.node, [np.full(3, np.inf), np.full(3, -np.inf)])
        box[0] = np.minimum(box[0], primitive.positions.min(axis=0))
        box[1] = np.maximum(box[1], primitive.positions.max(axis=0))
    detail_nodes = {node for node, (lo, hi) in node_boxes.items() if np.linalg.norm(hi - lo) < DETAIL_DIAGONAL}
    world_min = np.min([lo for lo, _ in node_boxes.values()], axis=0)
    world_max = np.max([hi for _, hi in node_boxes.values()], axis=0)
    _log("light grid")
    light_grid = lightgrid.write(probe_volumes, world_min, world_max, out_dir / "lightgrid.bin")

    def material_key(spec: MaterialSpec) -> tuple:
        class1 = assignment[spec.layers[0].key][0]
        class2 = assignment[spec.layers[1].key][0] if len(spec.layers) > 1 else ""
        return (spec.family, spec.double_sided, class1, class2)

    _log("geometry")
    pieces = geo.build_pieces(
        specs, material_key, locator, probe_volumes, CELL_SIZE, world_min, detail_nodes, primitives,
        entity_layout.node_objects,
    )
    del primitives
    geo.add_lod(pieces, options.lod_error, options.lod_attribute_scale)
    cell_table = visibility.CellTable(world_min)
    for piece in pieces:
        cell_table.add(geo.piece_centroids(piece))
    cell_table.finish()
    visibility_entry: dict[str, Any] | None = None
    visibility_report: dict[str, Any] = {"cells": len(cell_table)}
    voxels = visibility.load(executable, cs2_dir, map_name, work_dir)
    if voxels is not None:
        _log(f"visibility: {voxels.clusters} clusters, {len(cell_table)} cells")
        cell_visibility = visibility.sample_cells(
            voxels, cell_table, geo.triangle_batches(pieces, visibility.BATCH_TRIANGLES)
        )
        visibility_entry = visibility.write(voxels, cell_table, cell_visibility, out_dir / "visibility.bin")
        visibility_report.update(
            clusters=voxels.clusters,
            rows=visibility_entry["rows"][1],
            alwaysVisibleCells=int(cell_visibility.always_visible.sum()),
            unassignedTriangles=cell_visibility.unassigned,
            bytes=(out_dir / "visibility.bin").stat().st_size,
        )
        del voxels
    _log(f"visibility: {json.dumps(visibility_report)}")

    key_fields = ["family", "doubleSided", "class1", "class2", "detail", "object"]
    geo_dir = out_dir / "geo"
    records = geo.write_chunks(pieces, geo_dir, cell_table.index)
    del pieces
    geometry = {"dir": "geo", "lodError": options.lod_error, "chunks": geo.records_json(records, key_fields)}
    geometry_report: dict[str, Any] = {
        "chunks": len(records),
        "parts": sum(r.parts[1] for r in records),
        "vertices": sum(r.vertex_count for r in records),
        "triangles": sum(r.index_count for r in records) // 3,
        "lodTriangles": sum(r.lod["indexCount"] if r.lod else r.index_count for r in records) // 3,
        "bytes": sum(path.stat().st_size for path in geo_dir.iterdir()),
    }
    _log(f"geometry: {json.dumps(geometry_report)}")

    post_entity = next((e for e in entities if e.get("classname") == "post_processing_volume" and e.get("master")), None)
    post_text = (
        _postprocess_text(executable, cs2_dir, post_entity["postprocessing"], work_dir)
        if post_entity and post_entity.get("postprocessing")
        else None
    )
    post = postprocess_params(post_text, post_entity, out_dir)
    if map_name in EXPOSURE_OVERRIDES:
        post["exposure"] = EXPOSURE_OVERRIDES[map_name]

    manifest = {
        "format": "cs2render",
        "version": 5,
        "map": map_name,
        "units": "source-inches",
        "bounds": {"min": [round(float(x), 2) for x in world_min], "max": [round(float(x), 2) for x in world_max]},
        "scene": scene_params(entities),
        "post": post,
        "materials": {"file": "materials.bin", "count": len(specs), "texels": MATERIAL_TEXELS},
        "probes": {"file": "probes.ctex", "size": options.probe_size, "levels": options.probe_levels, "volumes": probe_metadata},
        "textures": textures,
        "chunkKey": key_fields,
        "geometry": geometry,
        "objects": entity_layout.objects,
    }
    if sky_entry is not None:
        manifest["sky"] = sky_entry
    if visibility_entry is not None:
        manifest["visibility"] = visibility_entry
    if light_grid is not None:
        manifest["lightGrid"] = {key: value for key, value in light_grid.items() if key != "bytes"}
    if options.compress:
        manifest["compression"] = "gzip"
    breakables.attach(executable, _vpk(cs2_dir), manifest["objects"], entities, work_dir, out_dir)
    geo.dump(out_dir / "map.json", manifest)

    if options.compress:
        _log("compressing package")
        compress_package(out_dir)
    geometry_report["bytes"] = sum(path.stat().st_size for path in geo_dir.iterdir())
    probe_bytes = (out_dir / "probes.ctex").stat().st_size
    texture_bytes = sum((out_dir / f).stat().st_size for f in _texture_files(textures))

    report = {
        "seconds": round(time.time() - started, 1),
        "geometry": geometry_report,
        "visibility": visibility_report,
        "probeBytes": probe_bytes,
        "textureBytes": texture_bytes,
        "totalBytes": sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file()),
    }
    _log(json.dumps(report))
    return report


def _texture_files(textures: dict[str, Any]) -> list[str]:
    files = [entry[key] for entry in textures["classes"].values() for key in ("color", "normal", "mask")]
    if "overlay" in textures:
        files.append(textures["overlay"]["file"])
    files.extend(textures["lightmaps"].values())
    return files


def _compress_file(path: Path) -> None:
    data = path.read_bytes()
    if data[:2] != b"\x1f\x8b":  # already compressed (lightgrid.bin, files kept by --geometry-only)
        path.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))


def compress_package(out_dir: Path) -> None:
    """Gzips every package file except the manifest and the GLBs (meshopt-compressed, loaded as-is), in place."""
    files = [
        path for path in sorted(out_dir.rglob("*")) if path.is_file() and path.name != "map.json" and path.suffix != ".glb"
    ]
    with ProcessPoolExecutor() as pool:
        list(pool.map(_compress_file, files, chunksize=8))
