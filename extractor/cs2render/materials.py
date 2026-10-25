"""Classifies CS2 materials and derives offline texture bakes plus compact runtime parameters.

Most per-material maths from the CS2 shaders (colour correction, tint-mask contrast, roughness
remaps, AO levels, normal contrast) is constant per texel, so it is baked into the textures here.
Only per-instance or cross-layer terms stay in the runtime shader.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

SKIPPED_SHADERS = {"csgo_effects.vfx", "csgo_black_unlit.vfx", "csgo_water_fancy.vfx", "generic.vfx"}

# Families drive draw passes and shader variants.
OPAQUE = "opaque"
BLEND = "blend"
ALPHATEST = "alphatest"
TRANSLUCENT = "translucent"

# How a material reads vertex paint (COLOR_0); see geometry.vertex_paint.
PAINT_NONE = "none"
PAINT_RAW = "raw"  # csgo_environment(_blend): mix(1, rgb, a)
PAINT_SRGB = "srgb"  # complex: rgb multiplied after sRGB decode
PAINT_OVERLAY = "overlay"  # static overlay: raw rgb, all-zero means unpainted

COMPLEX_SHADERS = {
    "csgo_complex.vfx",
    "csgo_simple.vfx",
    "csgo_vertexlitgeneric.vfx",
    "csgo_foliage.vfx",
    "csgo_static_overlay.vfx",
    "csgo_simple_3layer_parallax.vfx",
    "csgo_lightmappedgeneric.vfx",
    "csgo_glass.vfx",
}


@dataclass(frozen=True)
class LayerBake:
    """Everything needed to build one packed texture set (colour, normal, mask)."""

    kind: str  # "environment" or "complex"
    color: str | None
    normal: str | None
    height: str | None = None
    ao: str | None = None
    tint_mask: str | None = None
    metal: str | None = None
    self_illum: str | None = None
    alpha: str = "none"  # none | color.a (opacity)
    ao_source: str = "none"  # none | color.a | height.b | ao.r
    mask_alpha: str = "none"  # none | height.r | metal.g | selfillum.r
    color_correct: bool = False
    csb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    color_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    tint_mask_contrast: tuple[float, float] = (1.0, 1.0)
    rough_contrast: tuple[float, float] = (1.0, 1.0)
    normal_contrast: float = 1.0
    ao_levels: tuple[float, float, float] = (0.0, 0.5, 1.0)
    constant_color: tuple[float, float, float] | None = None
    # csgo_glass: opacity = mix(low, high, colour alpha) (g_flTranslucencyRemap).
    alpha_remap: tuple[float, float] = (0.0, 1.0)

    @property
    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    @property
    def needs_alpha(self) -> bool:
        return self.alpha != "none"


@dataclass
class MaterialSpec:
    index: int
    name: str
    shader: str
    family: str
    layers: list[LayerBake]
    overlay: str | None = None
    double_sided: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    # Texture coordinate scale axes (Source x/y/z = 1/2/3) for layer 1 and layer 2.
    uv_scale_axes: tuple[int, int] = (0, 0)
    uv2_scale_axes: tuple[int, int] = (0, 0)
    # Which mesh uv each layer and the shared colour overlay sample: 1 = TEXCOORD_0, 2 = the secondary uv.
    uv_sets: tuple[int, int] = (1, 1)
    overlay_uv_set: int = 2


def _f(values: dict[str, Any], key: str, default: float) -> float:
    value = values.get(key, default)
    return float(value) if value is not None else default


def _v(values: dict[str, Any], key: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = values.get(key)
    if value is None:
        return default
    return tuple(float(x) for x in value[: len(default)])


def _srgb_to_linear(value: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def _tex(textures: dict[str, str], key: str) -> str | None:
    name = textures.get(key)
    if not name or name.startswith("default_"):
        return None
    return name


def _texture_or_default(textures: dict[str, str], key: str) -> str | None:
    return textures.get(key) or None


def classify(index: int, material: dict[str, Any]) -> MaterialSpec | None:
    vmat = material.get("extras", {}).get("vmat", {})
    shader = vmat.get("ShaderName", "")
    name = vmat.get("Name", material.get("name", ""))
    if shader in SKIPPED_SHADERS or "/tools/" in name or name.startswith("materials/tools"):
        return None
    ints = vmat.get("IntParams", {})
    # Flipbook-animated overlays (Nuke/Anubis water caustics) would freeze on one frame; leave them out.
    if shader == "csgo_static_overlay.vfx" and int(ints.get("F_TEXTURE_ANIMATION", 0)) == 1:
        return None
    floats = vmat.get("FloatParams", {})
    vectors = vmat.get("VectorParams", {})
    textures = vmat.get("TextureParams", {})
    pbr = material.get("pbrMetallicRoughness", {})
    base_factor = tuple(pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]))

    if shader in ("csgo_environment.vfx", "csgo_environment_blend.vfx"):
        return _environment(index, name, shader, ints, floats, vectors, textures, base_factor)
    if shader in COMPLEX_SHADERS:
        return _complex(index, name, shader, ints, floats, vectors, textures, base_factor, material)
    return None


def _environment_layer(
    n: str, ints: dict, floats: dict, vectors: dict, textures: dict, alpha_test: bool
) -> LayerBake:
    color_correct = int(ints.get(f"g_nColorCorrectionMode{n}", 0)) == 1
    tint = _v(vectors, f"g_vTextureColorTint{n}", (1.0, 1.0, 1.0))
    csb = (
        _f(floats, f"g_fTextureColorContrast{n}", 1.0),
        _f(floats, f"g_fTextureColorSaturation{n}", 1.0),
        _f(floats, f"g_fTextureColorBrightness{n}", 1.0),
    )
    return LayerBake(
        kind="environment",
        color=_texture_or_default(textures, f"g_tColor{n}"),
        normal=_texture_or_default(textures, f"g_tNormal{n}"),
        height=_texture_or_default(textures, f"g_tHeight{n}"),
        alpha="color.a" if alpha_test else "none",
        ao_source="height.b" if alpha_test else "color.a",
        mask_alpha="height.r",
        color_correct=color_correct,
        csb=csb,
        color_tint=tuple(tint),
        tint_mask_contrast=(_f(floats, f"g_fTintMaskContrast{n}", 1.0), _f(floats, f"g_fTintMaskBrightness{n}", 1.0)),
        rough_contrast=(
            _f(floats, f"g_fTextureRoughnessContrast{n}", 1.0),
            _f(floats, f"g_fTextureRoughnessBrightness{n}", 1.0),
        ),
        normal_contrast=_f(floats, f"g_fTextureNormalContrast{n}", 1.0),
        ao_levels=_v(vectors, f"g_vAmbientOcclusionLevels{n}", (0.0, 0.5, 1.0)),
    )


def _environment(
    index: int,
    name: str,
    shader: str,
    ints: dict,
    floats: dict,
    vectors: dict,
    textures: dict,
    base_factor: tuple[float, ...],
) -> MaterialSpec:
    blend = shader == "csgo_environment_blend.vfx"
    alpha_test = int(ints.get("F_ALPHA_TEST", 0)) == 1
    layers = [_environment_layer("1", ints, floats, vectors, textures, alpha_test)]
    if blend:
        layers.append(_environment_layer("2", ints, floats, vectors, textures, False))

    color_tint = _v(vectors, "g_vColorTint", (1.0, 1.0, 1.0))
    color_tint_linear = _srgb_to_linear(np.array(color_tint))
    # glTF baseColorFactor = linear(lerp(1, modelTint, amount) * g_vColorTint); recover the instance tint.
    model_tint = np.clip(np.array(base_factor[:3]) / np.maximum(color_tint_linear, 1e-4), 0.0, 1.0)
    tint_amount = _f(floats, "g_flModelTintAmount", 1.0) * (1.0 - float(model_tint.min()))

    params: dict[str, Any] = {
        "modelTint": [float(x) for x in model_tint],
        "modelTintAmount": tint_amount,
        "vertexTint": [float(x) for x in color_tint_linear],
        "modelTintLayers": [
            1.0 if int(ints.get("g_bModelTint1", 1)) else 0.0,
            1.0 if int(ints.get("g_bModelTint2", 1)) else 0.0,
        ],
        "alphaTestReference": _f(floats, "g_flAlphaTestReference1", 0.5),
        "vertexPaint": PAINT_RAW,
    }
    if blend:
        params.update(
            {
                "heightScale": [_f(floats, "g_flHeightMapScale1", 1.0), _f(floats, "g_flHeightMapScale2", 1.0)],
                "heightZero": [_f(floats, "g_flHeightMapZeroPoint1", 0.5), _f(floats, "g_flHeightMapZeroPoint2", 0.5)],
                "blendSoftness": _f(floats, "g_flBlendSoftness2", 0.01),
                "vertexColorMode": [int(ints.get("g_nVertexColorMode1", 0)), int(ints.get("g_nVertexColorMode2", 0))],
                "overlayContrast": [
                    _f(floats, "g_flOverlayBrightnessContrast", 1.0),
                    _f(floats, "g_flOverlayDarknessContrast", 1.0),
                ],
            }
        )
    overlay = None
    if blend and int(ints.get("F_SHARED_COLOR_OVERLAY", 0)) == 1:
        overlay = textures.get("g_tSharedColorOverlay")

    family = BLEND if blend else (ALPHATEST if alpha_test else OPAQUE)
    return MaterialSpec(
        index=index,
        name=name,
        shader=shader,
        family=family,
        layers=layers,
        overlay=overlay,
        double_sided=int(ints.get("F_RENDER_BACKFACES", 0)) == 1,
        params=params,
        uv_scale_axes=(
            int(ints.get("g_nScaleTexCoordUByModelScaleAxis", 0)),
            int(ints.get("g_nScaleTexCoordVByModelScaleAxis", 0)),
        ),
        uv2_scale_axes=(
            int(ints.get("g_nScaleTexCoord2UByModelScaleAxis", 0)),
            int(ints.get("g_nScaleTexCoord2VByModelScaleAxis", 0)),
        ),
        # 0 (biplanar) is not supported and falls back to the first uv.
        uv_sets=(int(ints.get("g_nUVSet1", 1)), int(ints.get("g_nUVSet2", 1))),
        overlay_uv_set=int(ints.get("g_nColorOverlayUVSet", 2)),
    )


def _complex_vertex_paint(shader: str, ints: dict) -> str:
    """How complex shaders read COLOR_0 (VRF complex.vert): foliage stores animation params there, not paint."""
    if shader == "csgo_foliage.vfx":
        return PAINT_NONE
    if not any(int(ints.get(flag, 0)) == 1 for flag in ("F_VERTEX_COLOR", "F_PAINT_VERTEX_COLORS")):
        return PAINT_NONE
    return PAINT_OVERLAY if shader == "csgo_static_overlay.vfx" else PAINT_SRGB


def _complex(
    index: int,
    name: str,
    shader: str,
    ints: dict,
    floats: dict,
    vectors: dict,
    textures: dict,
    base_factor: tuple[float, ...],
    material: dict[str, Any],
) -> MaterialSpec | None:
    alpha_test = int(ints.get("F_ALPHA_TEST", 0)) == 1
    translucent = (
        int(ints.get("F_TRANSLUCENT", 0)) == 1
        or shader in ("csgo_static_overlay.vfx", "csgo_glass.vfx")
        or material.get("alphaMode") == "BLEND"
    )
    self_illum = int(ints.get("F_SELF_ILLUM", 0)) == 1 and _tex(textures, "g_tSelfIllumMask") is not None
    metal_texture = int(ints.get("F_METALNESS_TEXTURE", 0)) == 1 and _tex(textures, "g_tMetalness") is not None
    tint_mask = _tex(textures, "g_tTintMask") if int(ints.get("F_TINT_MASK", 0)) == 1 else None

    color = textures.get("g_tColor")
    constant_color = None
    normal = textures.get("g_tNormal")
    if shader == "csgo_glass.vfx":
        # Like Source2Viewer's MaterialLoader: the dust texture is the glass colour and opacity (alpha).
        # g_tGlassTintColor is a 1x1 constant, not a surface texture.
        color = textures.get("g_tGlassDust") or textures.get("g_tColor") or textures.get("g_tGlassTintColor")
    if shader == "csgo_lightmappedgeneric.vfx":
        normal = textures.get("g_tLayer1NormalRoughness")
        layer_color = vectors.get("TextureLayer1Color")
        if layer_color is not None:
            constant_color = tuple(float(x) for x in layer_color[:3])
    if not color and constant_color is None:
        return None

    mask_alpha = "selfillum.r" if self_illum else ("metal.g" if metal_texture else "none")
    layer = LayerBake(
        kind="complex",
        color=color,
        normal=normal,
        ao=_tex(textures, "g_tAmbientOcclusion") or _tex(textures, "g_tLayer1AmbientOcclusion"),
        tint_mask=tint_mask,
        metal=_tex(textures, "g_tMetalness") if metal_texture else None,
        self_illum=_tex(textures, "g_tSelfIllumMask") if self_illum else None,
        alpha="color.a" if (alpha_test or translucent) else "none",
        ao_source="ao.r",
        mask_alpha=mask_alpha,
        color_correct=shader == "csgo_static_overlay.vfx",
        normal_contrast=_f(floats, "g_flBumpStrength", 1.0),
        constant_color=constant_color,
        alpha_remap=_v(vectors, "g_flTranslucencyRemap", (0.0, 1.0)) if shader == "csgo_glass.vfx" else (0.0, 1.0),
    )

    color_tint_linear = _srgb_to_linear(np.array(_v(vectors, "g_vColorTint", (1.0, 1.0, 1.0))))
    # Complex shaders multiply by the vertex colour tint (g_vColorTint x instance tint), masked by the tint mask.
    tint = np.clip(np.array(base_factor[:3]), 0.0, 4.0)
    self_illum_tint = _srgb_to_linear(np.array(_v(vectors, "g_vSelfIllumTint", (1.0, 1.0, 1.0))))
    self_illum_scale = (
        2.0 ** _f(floats, "g_flSelfIllumBrightness", 0.0) * _f(floats, "g_flSelfIllumScale", 1.0)
        if self_illum
        else 0.0
    )
    params: dict[str, Any] = {
        "tint": [float(x) for x in (tint if base_factor[:3] != (1.0, 1.0, 1.0) else color_tint_linear)],
        "metalness": 0.0 if metal_texture else _f(floats, "g_flMetalness", 0.0),
        "metalTexture": metal_texture,
        "selfIllum": [float(x) * self_illum_scale for x in self_illum_tint],
        "selfIllumAlbedo": _f(floats, "g_flSelfIllumAlbedoFactor", 0.0),
        "alphaTestReference": _f(floats, "g_flAlphaTestReference", 0.5),
        "opacityScale": _f(floats, "g_flOpacityScale", 1.0),
        "paintVertexColors": int(ints.get("F_PAINT_VERTEX_COLORS", 0)) == 1,
        "vertexPaint": _complex_vertex_paint(shader, ints),
        "transmissive": int(ints.get("F_TRANSMISSIVE_BACKFACE_NDOTL", 0)) == 1,
        "glass": shader == "csgo_glass.vfx",
    }
    if translucent:
        family = TRANSLUCENT
    elif alpha_test:
        family = ALPHATEST
    else:
        family = OPAQUE
    return MaterialSpec(
        index=index,
        name=name,
        shader=shader,
        family=family,
        layers=[layer],
        double_sided=int(ints.get("F_RENDER_BACKFACES", 0)) == 1 or material.get("doubleSided", False),
        params=params,
        uv_scale_axes=(
            int(ints.get("g_nScaleTexCoordUByModelScaleAxis", 0)),
            int(ints.get("g_nScaleTexCoordVByModelScaleAxis", 0)),
        ),
    )


# --- Colour correction matrices, ported from VRF's VfxEvalFunctions (MIT). Row-vector convention. ---

_LUMA = np.array([0.2126, 0.7152, 0.0722])
_LUMA_N = _LUMA / np.linalg.norm(_LUMA)


def _scale(x: float, y: float | None = None, z: float | None = None) -> np.ndarray:
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = x, x if y is None else y, x if z is None else z
    return m


def _translate(t: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[3, :3] = t
    return m


def _luma_rotation() -> np.ndarray:
    axis = np.cross(_LUMA_N, [0.0, 0.0, 1.0])
    angle = np.arctan2(np.linalg.norm(axis), _LUMA_N[2])
    x, y, z = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    column = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )
    m = np.eye(4)
    m[:3, :3] = column.T
    return m


def color_correct_matrix(csb: tuple[float, float, float], average: np.ndarray) -> np.ndarray:
    """MatrixColorCorrect2 as a row-vector 4x4 (apply as [r, g, b, a] @ M)."""
    rotation = _luma_rotation()
    m = _translate(-average) @ _scale(csb[0]) @ _translate(average)
    m = m @ _scale(csb[2])
    m = m @ _scale(*_LUMA_N)
    m = m @ rotation
    m = m @ _scale(csb[1], csb[1], 1.0)
    m = m @ rotation.T
    m = m @ _scale(*(1.0 / _LUMA_N))
    return m


def color_tint_matrix(rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """MatrixColorTint2 as a row-vector 4x4."""
    rgb = np.asarray(rgb, dtype=np.float64)
    peak = rgb.max()
    saturation = 0.0 if peak == 0 else (peak - rgb.min()) / peak
    rotation = _luma_rotation()
    lum_scale = _scale(*_LUMA_N)
    gray = (np.append(rgb, 1.0) @ (lum_scale @ rotation))[:3]
    desat = 1.0 - saturation
    sat_factor = 1.0 - saturation * saturation * strength
    m = lum_scale @ rotation
    m = m @ _translate(np.array([-gray[0], -gray[1], 0.0]))
    m = m @ _scale(desat, desat, sat_factor)
    m = m @ _translate(np.array([gray[0], gray[1], (1.0 - sat_factor) * gray[2]]))
    m = m @ rotation.T
    m = m @ _scale(*(1.0 / _LUMA_N))
    return m


def layer_matrices(layer: LayerBake, average: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (g_mTextureAdjust, g_mTextureColorAdjust) for a layer, both row-vector 4x4."""
    cc = color_correct_matrix(layer.csb, average)
    tinted = cc @ color_tint_matrix(np.array(layer.color_tint))
    return cc, tinted
