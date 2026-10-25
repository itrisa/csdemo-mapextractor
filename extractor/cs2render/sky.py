"""Bakes the map's visible env_sky panorama into a compact BC6H latlong (tex/sky.ctex).

The game draws the 2D sky as an HDR cubemap at the far plane:
texel * exp2(BrightnessExposureBias + RenderOnlyExposureBias) * tint_color * brightnessscale,
rotated by the env_sky angles, then tonemapped like the world. The bake folds the scale and the
rotation into the texels, so the client samples the result with the plain world direction:

  u = atan2(d.y, d.x) / 2pi (wrapped), v = acos(d.z) / pi / vMax (clamped)

Only the rows above vMax (zenith down to a little below the horizon) are kept; the client
stretches the last row over the rest. The 3D skybox (skybox_reference prefab geometry) is not
included.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .. import source2viewer
from . import lighting
from . import textures as tx
from .kv3 import entities as parse_entities

SKY_FILE = "tex/sky.ctex"
SKY_WIDTH = 2048
SKY_V_MAX = 0.625  # zenith to 22.5 degrees below the horizon


def select_sky(main: list[dict[str, Any]], prefab: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The env_sky the game shows: an enabled one from the skybox prefab, then from the map, then any."""
    skies = [e for e in prefab + main if e.get("classname") == "env_sky" and e.get("skyname")]
    return next((e for e in skies if not e.get("StartDisabled")), skies[0] if skies else None)


def material_params(text: str) -> tuple[str | None, float]:
    """Sky texture and summed exposure bias from a sky material's DATA block."""
    texture = re.search(r'"g_tSkyTexture"\s*m_pValue\s*=\s*resource:"([^"]+)"', text)
    biases = re.findall(r'm_name\s*=\s*"g_fl(?:Brightness|RenderOnly)ExposureBias"\s*m_flValue\s*=\s*(-?[\d.]+)', text)
    return (texture.group(1) if texture else None), sum(float(b) for b in biases)


def sky_scale(entity: dict[str, Any], exposure_bias: float) -> np.ndarray:
    tint = np.array([float(c) for c in list(entity.get("tint_color", [255, 255, 255]))[:3]]) / 255.0
    brightness = float(entity.get("brightnessscale", 1.0))
    return tint * (brightness if brightness > 0 else 1.0) * 2.0**exposure_bias


def resample(latlong: np.ndarray, angles: Any, scale: np.ndarray, width: int, v_max: float) -> np.ndarray:
    """Rotates and crops a Source2Viewer latlong export into the client's world-direction latlong."""
    factor = max(1, latlong.shape[1] // width)
    if factor > 1:
        h, w, c = latlong.shape
        latlong = latlong[: h // factor * factor, : w // factor * factor]
        latlong = latlong.reshape(h // factor, factor, w // factor, factor, c).mean(axis=(1, 3))
    src_h, src_w = latlong.shape[:2]
    height = int(round(width / 2 * v_max)) // 4 * 4
    azimuth = (np.arange(width) + 0.5) / width * 2 * math.pi
    polar = (np.arange(height) + 0.5) / height * v_max * math.pi
    sin_p = np.sin(polar)[:, None]
    world = np.stack(
        [sin_p * np.cos(azimuth)[None, :], sin_p * np.sin(azimuth)[None, :], np.repeat(np.cos(polar)[:, None], width, axis=1)],
        axis=-1,
    ).reshape(-1, 3)
    rotation = lighting._angles_matrix(tuple(lighting._vec(angles)))
    local = world @ rotation  # rotation.T @ d per row
    # Source2Viewer's latlong direction (sinV cosU, cosV, sinV sinU) is Source (x, z, -y).
    u = np.mod(np.arctan2(-local[:, 1], local[:, 0]), 2 * math.pi) / (2 * math.pi)
    v = np.arccos(np.clip(local[:, 2], -1.0, 1.0)) / math.pi
    x = u * src_w - 0.5
    y = np.clip(v * src_h - 0.5, 0, src_h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x0 %= src_w
    x1 = (x0 + 1) % src_w
    y1 = np.minimum(y0 + 1, src_h - 1)
    rgb = latlong[..., :3]
    top = rgb[y0, x0] * (1 - fx) + rgb[y0, x1] * fx
    bottom = rgb[y1, x0] * (1 - fx) + rgb[y1, x1] * fx
    color = np.clip((top * (1 - fy) + bottom * fy) * scale[None, :], 0.0, None)
    image = np.concatenate([color, np.ones((color.shape[0], 1))], axis=1)
    return image.reshape(height, width, 4).astype(np.float32)


def _prefab_entities(executable: Path, cs2_dir: Path, main: list[dict[str, Any]], work_dir: Path) -> list[dict[str, Any]]:
    reference = next((e for e in main if e.get("classname") == "skybox_reference" and e.get("targetMapName")), None)
    if reference is None:
        return []
    target = str(reference["targetMapName"]).removesuffix(".vmap")
    cached = work_dir / "skybox_entities.kv3"
    if not cached.is_file():
        vpk = cs2_dir / "game" / "csgo" / f"{target}.vpk"
        if not vpk.is_file():
            return []
        cached.write_text(
            source2viewer.extract_resource_block(executable, vpk, f"{target}/entities/default_ents.vents_c"),
            encoding="utf-8",
        )
    return parse_entities(cached.read_text(encoding="utf-8"))


def build_sky(
    executable: Path, cs2_dir: Path, entities: list[dict[str, Any]], work_dir: Path, out_dir: Path, texconv: Path
) -> dict[str, Any] | None:
    """Writes tex/sky.ctex; returns the manifest entry, or None when the map has no usable sky."""
    sky_dir = work_dir / "sky"
    sky_dir.mkdir(parents=True, exist_ok=True)
    try:
        entity = select_sky(entities, _prefab_entities(executable, cs2_dir, entities, work_dir))
    except Exception as error:  # noqa: BLE001 - the skybox prefab is optional
        print(f"[cs2render] skybox prefab unavailable: {error}", flush=True)
        entity = select_sky(entities, [])
    if entity is None:
        return None
    pak = cs2_dir / "game" / "csgo" / "pak01_dir.vpk"
    material = str(entity["skyname"])
    material_cache = sky_dir / (Path(material).stem + ".kv3")
    if not material_cache.is_file():
        material_cache.write_text(source2viewer.extract_resource_block(executable, pak, material + "_c"), encoding="utf-8")
    texture, exposure_bias = material_params(material_cache.read_text(encoding="utf-8"))
    if texture is None:
        return None
    exports = sky_dir / "export"
    found = list(exports.rglob(Path(texture).stem + "*.exr")) if exports.is_dir() else []
    if not found:
        source2viewer.export_resource(executable, pak, texture + "_c", exports)
        found = list(exports.rglob(Path(texture).stem + "*.exr"))
    if not found:
        raise RuntimeError(f"Source2Viewer exported no EXR for sky texture {texture}")
    image = resample(
        lighting.read_exr(found[0]), entity.get("angles"), sky_scale(entity, exposure_bias), SKY_WIDTH, SKY_V_MAX
    )
    height, width = image.shape[:2]
    blocks = lighting.encode_bc6h([image], texconv)
    tx.write_single(out_dir / SKY_FILE, tx.FORMATS["hdr"], width, height, [blocks[0]])
    return {"file": SKY_FILE, "width": width, "height": height, "vMax": SKY_V_MAX, "material": material}
