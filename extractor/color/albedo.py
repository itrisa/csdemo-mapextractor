"""Numpy port of the VRF (Source2Viewer) albedo math for CS2 world materials.

Evaluates the linear albedo (and opacity) of render-world surfaces at arbitrary hit points so the colour can be
transferred onto collision geometry. Lighting, normals, roughness etc. are ignored; only what ends up in
mat.Albedo / mat.Opacity in VRF's complex.frag and csgo_environment.frag is reproduced.
"""
import math
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from . import gltf as G

Image.MAX_IMAGE_PIXELS = None
HU = 0.0254
LUMA = np.array([0.2125, 0.7154, 0.0721])
LUM_N = np.array([0.2126, 0.7152, 0.0722]) / np.linalg.norm([0.2126, 0.7152, 0.0722])
FOOTPRINT = 24.0  # hammer units a vertex colour stands for; picks the texture mip
MAX_TEX = 256

# ---------------------------------------------------------------- colour helpers

def s2l(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((np.maximum(c, 0) + 0.055) / 1.055) ** 2.4)


def l2s(c):
    c = np.clip(np.asarray(c, np.float64), 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def sat(x):
    return np.clip(x, 0.0, 1.0)


def mix(a, b, t):
    return a + (b - a) * t


def luma(c):
    return c @ LUMA


def smoothstep(e0, e1, x):
    t = sat((x - e0) / np.where(e1 - e0 == 0, 1e-9, e1 - e0))
    return t * t * (3 - 2 * t)


def rotate2d(uv, rot_deg, scale, offset, center):
    r = math.radians(rot_deg)
    c, s = math.cos(r), math.sin(r)
    v = (uv - center) * scale
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], 1) + center + offset


# ---------------------------------------------------------------- System.Numerics-style (row vector) matrices

def _S(x, y=None, z=None):
    if y is None:
        y = z = x
    return np.diag([x, y, z, 1.0])


def _T(x, y, z):
    m = np.eye(4)
    m[3, :3] = (x, y, z)
    return m


def _axis_angle(axis, ang):
    x, y, z = axis
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                  [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                  [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])
    m = np.eye(4)
    m[:3, :3] = R.T  # System.Numerics matrices act on row vectors
    return m


_cross = np.cross(LUM_N, [0, 0, 1.0])
_ROT = _axis_angle(_cross / np.linalg.norm(_cross), math.atan2(np.linalg.norm(_cross), LUM_N[2]))


def color_correct2(csb, avg):
    c, s, b = csb
    m = _T(*(-np.asarray(avg))) @ _S(c) @ _T(*avg) @ _S(b) @ _S(*LUM_N) @ _ROT @ _S(s, s, 1) @ _ROT.T @ _S(*(1 / LUM_N))
    return m


def color_tint2(rgb, strength):
    rgb = np.asarray(rgb, np.float64)
    mx, mn = rgb.max(), rgb.min()
    satv = 0.0 if mx == 0 else (mx - mn) / mx
    lum_rot = _S(*LUM_N) @ _ROT
    gray = rgb @ lum_rot[:3, :3]
    sat_f = 1 - satv * satv * strength
    return (lum_rot @ _T(-gray[0], -gray[1], 0) @ _S(1 - satv, 1 - satv, sat_f)
            @ _T(gray[0], gray[1], (1 - sat_f) * gray[2]) @ _ROT.T @ _S(*(1 / LUM_N)))


def apply_matrix(rgba, M):
    """GLSL `color * M` as VRF uploads it: row vector times the System.Numerics matrix."""
    return rgba @ M


# ---------------------------------------------------------------- textures

class Texture:
    __slots__ = ("levels", "base", "size", "mean")


TextureLoader = Callable[[str, bool], "tuple[np.ndarray, int, tuple[int, int]] | None"]


class TextureCache:
    """Linear float mip chains no larger than MAX_TEX, built from level-0 images supplied by `loader`."""

    def __init__(self, loader: TextureLoader):
        self.loader = loader
        self.cache = {}

    def get(self, vtex, srgb):
        key = (vtex, srgb)
        if key not in self.cache:
            level0 = self.loader(vtex, srgb)
            self.cache[key] = None if level0 is None else build_texture(*level0)
        return self.cache[key]


def png_level0(folder, vtex, srgb):
    """Load an exported PNG and box-filter it down to at most MAX_TEX (linear space for sRGB textures).

    Returns (image float32 HxWx4, number of halvings, original size) or None when the PNG is missing."""
    path = Path(folder) / (vtex[:-5] + ".png" if vtex.endswith(".vtex") else vtex + ".png")
    if not path.exists():
        return None
    img = Image.open(path)
    img.draft("RGBA", (MAX_TEX, MAX_TEX))
    img = img.convert("RGBA")
    w0, h0 = img.size
    a = np.asarray(img, np.float32) / 255.0
    if srgb:
        a[..., :3] = s2l(a[..., :3]).astype(np.float32)
    base = 0
    while max(a.shape[0], a.shape[1]) > MAX_TEX and min(a.shape[0], a.shape[1]) >= 2:
        a = _half(a)
        base += 1
    return a, base, (w0, h0)


def build_texture(level0, base, size):
    t = Texture()
    t.size = tuple(size)
    t.base = base
    a = level0
    t.levels = [a]
    while min(a.shape[0], a.shape[1]) > 1:
        a = _half(a)
        t.levels.append(a)
    t.mean = t.levels[-1].reshape(-1, 4).mean(0)
    return t


def _half(a):
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    a = a[:h, :w]
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def sample(t, uv, density, clamp=(False, False)):
    """Bilinear sample with a mip chosen so one texel spans about FOOTPRINT hammer units of surface."""
    n = len(uv)
    out = np.empty((n, 4), np.float32)
    lod = np.log2(np.maximum(1.0, FOOTPRINT * density * max(t.size)))
    lvl = np.clip(np.round(lod).astype(np.int64) - t.base, 0, len(t.levels) - 1)
    for L in np.unique(lvl):
        sel = np.flatnonzero(lvl == L)
        img = t.levels[L]
        h, w = img.shape[:2]
        x = uv[sel, 0] * w - 0.5
        y = uv[sel, 1] * h - 0.5
        x0, y0 = np.floor(x), np.floor(y)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
        x1, y1 = x0 + 1, y0 + 1
        if clamp[0]:
            x0, x1 = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
        else:
            x0, x1 = x0 % w, x1 % w
        if clamp[1]:
            y0, y1 = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)
        else:
            y0, y1 = y0 % h, y1 % h
        out[sel] = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
                    + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    return out


# ---------------------------------------------------------------- materials

SKIP_SHADERS = ("csgo_water_fancy.vfx", "csgo_effects.vfx", "csgo_black_unlit.vfx", "generic.vfx")
COMPLEX_SHADERS = ("csgo_complex.vfx", "csgo_simple.vfx", "csgo_lightmappedgeneric.vfx", "csgo_vertexlitgeneric.vfx",
                   "csgo_static_overlay.vfx", "csgo_unlitgeneric.vfx", "csgo_foliage.vfx", "csgo_glass.vfx",
                   "csgo_simple_2way_blend.vfx", "csgo_simple_3layer_parallax.vfx")
SRGB_TEXTURES = ("g_tColor", "g_tLayer1Color", "g_tLayer2Color", "g_tSharedColorOverlay")

# composite roles
BASE, OVER_ALPHA, OVER_MOD2X, OVER_MUL = 0, 1, 2, 3


class Material:
    def __init__(self, gltf_mat, textures):
        v = (gltf_mat.get("extras") or {}).get("vmat", {})
        self.name = v.get("Name", gltf_mat.get("name", "")).lower()
        self.shader = v.get("ShaderName", "")
        self.I = v.get("IntParams", {})
        self.Fp = v.get("FloatParams", {})
        self.Vp = v.get("VectorParams", {})
        self.Tp = v.get("TextureParams", {})
        self.tex = textures
        self.double_sided = bool(gltf_mat.get("doubleSided"))
        sh = self.shader
        i = self.I
        self.skip = (not v or self.name.startswith("materials/tools/") or "sky" in sh or "sky" in self.name.split("/")[-1][:4]
                     or sh in SKIP_SHADERS or i.get("F_ADDITIVE_BLEND", 0) == 1)
        self.env = sh in ("csgo_environment.vfx", "csgo_environment_blend.vfx")
        self.blend_mode = i.get("F_BLEND_MODE", 0)
        is_overlay = sh == "csgo_static_overlay.vfx"
        unlit_or_overlay = is_overlay or sh == "csgo_unlitgeneric.vfx"
        self.alphatest = i.get("F_ALPHA_TEST", 0) == 1 or (unlit_or_overlay and self.blend_mode == 2)
        self.translucent = (i.get("F_TRANSLUCENT", 0) == 1 or i.get("F_GLASS", 0) == 1 or sh == "csgo_glass.vfx"
                            or (self.blend_mode > 0 and self.blend_mode != 2 and not self.env))
        if is_overlay and self.blend_mode == 4:
            self.skip = True
        if is_overlay:
            self.role = {3: OVER_MOD2X, 5: OVER_MUL, 6: OVER_MOD2X}.get(self.blend_mode, OVER_ALPHA)
        elif self.translucent:
            self.role = OVER_ALPHA
        else:
            self.role = BASE
        if not self.skip and not self.env and sh not in COMPLEX_SHADERS:
            self.skip = True
        # object tint, recovered from the exported base colour (= srgb2lin(lerp(1, tint, amount) * g_vColorTint))
        bcf = np.array(gltf_mat.get("pbrMetallicRoughness", {}).get("baseColorFactor", [1, 1, 1, 1])[:3], np.float64)
        ct = np.array(self.Vp.get("g_vColorTint", [1, 1, 1, 1])[:3], np.float64)
        amt = self.Fp.get("g_flModelTintAmount", 1.0)
        g = l2s(bcf)
        with np.errstate(divide="ignore", invalid="ignore"):
            tint = 1 + (np.where(ct > 1e-4, g / ct, 1.0) - 1) / (amt if abs(amt) > 1e-4 else 1.0)
        self.obj_tint = np.clip(np.nan_to_num(tint, nan=1.0), 0, 1)
        self.color_tint_lin = s2l(ct)
        self._mats = {}

    def F(self, k, d=0.0):
        return float(self.Fp.get(k, d))

    def V(self, k, d=(1.0, 1.0, 1.0, 1.0)):
        return np.array(self.Vp.get(k, d), np.float64)

    def Iv(self, k, d=0):
        return int(self.I.get(k, d))

    @staticmethod
    def is_srgb(param, name):
        return any(param.startswith(p) for p in SRGB_TEXTURES) or (param.startswith("g_tDecal") and "color" in name)

    def texture_requests(self):
        """Every (texture, srgb) pair this material can sample."""
        return {(name, self.is_srgb(param, name)) for param, name in self.Tp.items() if name}

    def texture(self, param):
        name = self.Tp.get(param)
        if not name:
            return None
        return self.tex.get(name, self.is_srgb(param, name))

    def const_texture(self, param, default):
        v = self.Vp.get("Texture" + param[3:])
        return np.array(v[:4], np.float32) if v is not None else np.array(default, np.float32)

    def tex_sample(self, param, uv, density, default=(1, 1, 1, 1)):
        t = self.texture(param)
        if t is None:
            return np.broadcast_to(self.const_texture(param, default), (len(uv), 4)).copy()
        clamp = (self.Iv("g_nTextureAddressModeU") != 0, self.Iv("g_nTextureAddressModeV") != 0)
        return sample(t, uv, density, clamp)

    def tex_mean(self, param):
        t = self.texture(param)
        return t.mean[:3].astype(np.float64) if t is not None else np.ones(3)


# ---------------------------------------------------------------- hit attributes

class Hits:
    """Per-hit attributes (all numpy, N rows). uv2 falls back to uv0 when the mesh has no secondary UV."""
    def __init__(self, n):
        self.n = n
        self.uv0 = np.zeros((n, 2))
        self.uv2 = np.zeros((n, 2))
        self.d0 = np.full(n, 1 / 64)   # uv units per hammer unit
        self.d2 = np.full(n, 1 / 64)
        self.vcol = np.ones((n, 4))
        self.has_vcol = np.zeros(n, bool)
        self.blend = np.zeros((n, 4))
        self.pos = np.zeros((n, 3))
        self.nrm = np.zeros((n, 3))

    def take(self, sel):
        h = Hits.__new__(Hits)
        h.n = len(sel)
        for k in ("uv0", "uv2", "d0", "d2", "vcol", "has_vcol", "blend", "pos", "nrm"):
            setattr(h, k, getattr(self, k)[sel])
        return h


# ---------------------------------------------------------------- complex family (complex.frag)

def abm(f, m, s):
    return smoothstep(np.maximum(0, m - s), np.minimum(1, m + s), f)


def eval_complex(m, h):
    sh = m.shader
    tc = rotate2d(h.uv0, m.F("g_flTexCoordRotation"), m.V("g_vTexCoordScale")[:2], m.V("g_vTexCoordOffset", (0, 0, 0, 0))[:2],
                  m.V("g_vTexCoordCenter", (0.5, 0.5, 0, 0))[:2])
    dens = h.d0 * np.abs(m.V("g_vTexCoordScale")[:2]).max()
    is_overlay = sh == "csgo_static_overlay.vfx"
    generic = sh in ("csgo_lightmappedgeneric.vfx", "csgo_vertexlitgeneric.vfx")
    two_way = sh == "csgo_simple_2way_blend.vfx"
    foliage = sh == "csgo_foliage.vfx"

    vtx = np.ones((h.n, 4))
    if m.Iv("F_NOTINT") == 0:
        vtx[:, :3] = mix(1.0, s2l(m.obj_tint) * m.color_tint_lin, m.F("g_flModelTintAmount", 1.0))
    if not foliage and (m.Iv("F_VERTEX_COLOR") == 1 or m.Iv("F_PAINT_VERTEX_COLORS") == 1):
        if is_overlay:
            painted = np.sign(h.vcol.sum(1))[:, None]
            vtx *= mix(1.0, h.vcol, painted)
        else:
            vtx[:, :3] *= s2l(h.vcol[:, :3])
            vtx[:, 3] *= h.vcol[:, 3]
    tint_applied = False

    if sh == "csgo_simple_3layer_parallax.vfx":
        color = m.tex_sample("g_tColor", tc, dens)
        mask = m.tex_sample("g_tLayer0Mask", tc, dens)[:, :1]
        l1 = m.tex_sample("g_tLayer1Color", tc, dens)
        l2 = m.tex_sample("g_tLayer2Color", tc, dens)[:, :3]
        color[:, :3] = mix(mix(l2, l1[:, :3], l1[:, 3:4]), color[:, :3], mask)
        if m.Iv("F_TINT_MASK") == 1:
            tf = mix(1.0, vtx[:, :3], m.tex_sample("g_tTintMask", tc, dens)[:, :1])
        else:
            tf = vtx[:, :3]
        color[:, :3] *= tf
        tint_applied = True
    else:
        color = m.tex_sample("g_tColor", tc, dens).astype(np.float64)
        if is_overlay and m.blend_mode == 3:
            color[:, :3] = l2s(color[:, :3])
        if is_overlay:
            M = m._mats.get("adjust")
            if M is None:
                csb = (m.F("g_fTextureColorContrast", 1), m.F("g_fTextureColorSaturation", 1), m.F("g_fTextureColorBrightness", 1))
                M = m._mats["adjust"] = color_correct2(csb, m.tex_mean("g_tColor")) @ color_tint2(m.V("g_vTextureColorCorrectionTint")[:3], 1.0)
            color[:, :3] = apply_matrix(np.concatenate([color[:, :3], np.ones((h.n, 1))], 1), M)[:, :3]
        if (generic and m.Iv("F_LAYERS") > 0) or two_way:
            tcb = tc * m.V("g_vTexCoordScale2")[:2]
            densb = dens * np.abs(m.V("g_vTexCoordScale2")[:2]).max()
            color2 = m.tex_sample("g_tLayer2Color", tcb, densb).astype(np.float64)
            bf = h.blend[:, 0]
            fancy = m.Iv("F_FANCY_BLENDING")
            if generic and fancy > 0:
                mod = m.tex_sample("g_tBlendModulation", tcb, densb)
                soft = m.F("g_flBlendSoftness", 0.0)
                if fancy == 1:
                    bf = abm(bf, mod[:, 1], mod[:, 0])
                elif fancy == 2:
                    bf = abm(bf, mod[:, 1], soft)
                elif fancy == 3:
                    bf = abm(bf, mod[:, 3], soft)
            elif two_way:
                bf = abm(bf, m.tex_sample("g_tMask", tcb, densb)[:, 0], h.blend[:, 3])
                if m.Iv("F_ENABLE_TINT_MASKS") == 1:
                    tmk = m.tex_sample("g_tTintMask", tc, dens)
                    color[:, :3] *= 1 - tmk[:, :1] * (1 - vtx[:, :3])
                    color2[:, :3] *= 1 - tmk[:, 1:2] * (1 - vtx[:, :3])
                    tint_applied = True
            color = mix(color, color2, bf[:, None])

    alb = color[:, :3]
    opacity = color[:, 3].copy() if (m.translucent or m.alphatest) else np.ones(h.n)
    if is_overlay and m.Iv("F_PAINT_VERTEX_COLORS") == 1:
        alb = alb * vtx[:, :3]
        opacity = opacity * vtx[:, 3]
        tint_applied = True
    if m.translucent:
        opacity = opacity * m.F("g_flOpacityScale", 1.0)
    if not tint_applied:
        tint = vtx[:, :3]
        if m.Iv("F_TINT_MASK") == 1:
            use2 = (m.Iv("F_SECONDARY_UV") == 1 and m.Iv("g_bUseSecondaryUvForTintMask", 1) == 1) or m.Iv("F_FORCE_UV2") == 1
            tm = m.tex_sample("g_tTintMask", h.uv2 if use2 else tc, h.d2 if use2 else dens)[:, :1]
            tint = 1 - tm * (1 - tint)
        alb = alb * tint
    if m.Iv("F_DECAL_TEXTURE") == 1:
        use2 = (m.Iv("F_SECONDARY_UV") == 1 and m.Iv("g_bUseSecondaryUvForDecal", 0) == 1) or m.Iv("F_FORCE_UV2") == 1
        dec = m.tex_sample("g_tDecal", h.uv2 if use2 else tc, h.d2 if use2 else dens, (1, 1, 1, 0))
        if m.Iv("F_DECAL_BLEND_MODE") == 0:
            alb = mix(alb, dec[:, :3], dec[:, 3:4])
        else:
            alb = alb * dec[:, :3]
    return alb, opacity


# ---------------------------------------------------------------- environment family (csgo_environment*.frag)

def _biplanar(h):
    an = np.abs(h.nrm)
    scaled = h.pos * np.array([-1.0, 1.0, -1.0]) / 64.0
    major = np.argmax(an, 1)
    minor = np.argmin(an, 1)
    # ties: mimic the shader's strict comparisons closely enough
    median = 3 - major - minor
    median = np.where(major == minor, (major + 1) % 3, median)
    uv_axes = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    def uvs(ax):
        a = np.array([uv_axes[k] for k in range(3)])[ax]
        return np.stack([scaled[np.arange(h.n), a[:, 0]], scaled[np.arange(h.n), a[:, 1]]], 1)
    w = sat((np.stack([an[np.arange(h.n), major], an[np.arange(h.n), median]], 1) - 0.5773) * 2.365744)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
    return uvs(major), uvs(median), w


class EnvSampler:
    def __init__(self, m, h):
        self.m, self.h = m, h
        self._bp = None

    def bp(self):
        if self._bp is None:
            self._bp = _biplanar(self.h)
        return self._bp

    def layer_uv(self, L):
        m, h = self.m, self.h
        uvset = m.Iv(f"g_nUVSet{L}", 1)
        src, dens = (h.uv2, h.d2) if uvset == 2 else (h.uv0, h.d0)
        scale = m.V(f"g_vTexCoordScale{L}")[:2]
        tc = rotate2d(src, m.F(f"g_flTexCoordRotation{L}"), scale, m.V(f"g_vTexCoordOffset{L}", (0, 0, 0, 0))[:2],
                      m.V(f"g_vTexCoordCenter{L}", (0.5, 0.5, 0, 0))[:2])
        return uvset, tc, dens * np.abs(scale).max(), scale

    def sample(self, param, uvset, tc, dens, scale, default=(1, 1, 1, 1)):
        m = self.m
        if uvset != 0:
            return m.tex_sample(param, tc, dens, default).astype(np.float64)
        a, b, w = self.bp()
        d = np.full(self.h.n, np.abs(scale).max() / 64.0)
        out = m.tex_sample(param, a * scale, d, default).astype(np.float64) * w[:, :1]
        sel = w[:, 1] > 0
        if sel.any():
            out[sel] += m.tex_sample(param, b[sel] * scale, d[sel], default).astype(np.float64) * w[sel, 1:2]
        return out


def color_tint_colorize(a, t, k):
    tn = np.maximum(t, 0.001)
    tn = tn / np.linalg.norm(tn)
    l = luma(a)
    v = np.minimum(l / luma(tn), 3 * l * t.max())
    return sat(mix(a, tn[None, :] * v[:, None], k[:, None]))


def blend_band_weight(diff, bf, raw, soft, mwh):
    soft = np.maximum(soft, 1e-4)
    q = soft * 0.25
    ed = 0.02 + q
    cf = sat(0.5 + diff / (2 * soft))
    hi = sat((sat(bf + diff * 0.1 + 0.2) - (0.98 - q)) / ed)
    cf = mix(cf, 1.0, hi)
    lo = sat((ed - sat(bf + diff * 0.05 - 0.05)) / ed)
    cf = mix(cf, 0.0, lo)
    return sat(cf * mix(0.5, raw, mwh) * 2)


def blend_layer(below, bf, raw, zp, scale, soft, infl, mwh):
    w_below, sraw_b, scarry_b, under_b = below
    sf = 2 * bf - 1
    hps = (raw - zp) * scale + sf + np.maximum(sraw_b, 0)
    diff = hps - mix(scarry_b, under_b, infl)
    w = blend_band_weight(diff, bf, raw, soft, mwh)
    active = bf > 0
    w = np.where(active, w, 0.0)
    present = active & (w > 0.05)
    sraw = np.where(active, sf, 0.0)
    scarry = np.where(present, sf * w, 0.0)
    under = np.where(present, mix(under_b, np.maximum(under_b, hps), w), under_b)
    return w, (w, sraw, scarry, under)


def get_blend_weights(ht, hs, hz, bf, soft):
    h1 = hs[0] + soft
    height1 = ht[0] * h1
    h2 = hs[1] + soft
    h22 = ht[1] * (hs[1] - soft)
    b1 = (-hz[0] * h1 - ((1 - hz[1]) * h2)) - soft
    b2 = (1 - hz[0]) * h1 - ((-hz[1]) * h2)
    h2x = h22 + mix(b1, b2, bf)
    mx = np.maximum(height1, h2x) - soft
    w1 = np.maximum(height1 - mx, 0) + 0.001
    w2 = np.maximum(h2x - mx, 0)
    s = w1 + w2
    return w1 / s, w2 / s


def eval_env(m, h):
    blend_shader = m.shader == "csgo_environment_blend.vfx"
    S = EnvSampler(m, h)
    tint_lin = s2l(m.obj_tint)
    tint_amt = m.F("g_flModelTintAmount", 1.0) * (1 - tint_lin.min())
    vpaint = mix(1.0, h.vcol[:, :3], h.vcol[:, 3:4])
    vert_color = m.color_tint_lin[None, :] * vpaint
    layer3 = blend_shader and m.Iv("F_ENABLE_LAYER_3") == 1

    def layer(L):
        uvset, tc, dens, scale = S.layer_uv(L)
        col = S.sample(f"g_tColor{L}", uvset, tc, dens, scale)
        hgt = S.sample(f"g_tHeight{L}", uvset, tc, dens, scale, (0.5, 0.5, 0.5, 0))
        tm = sat(((hgt[:, 1] - 0.5) * m.F(f"g_fTintMaskContrast{L}", 1) + 0.5) * m.F(f"g_fTintMaskBrightness{L}", 1))
        key = f"env{L}"
        if key not in m._mats:
            csb = (m.F(f"g_fTextureColorContrast{L}", 1), m.F(f"g_fTextureColorSaturation{L}", 1), m.F(f"g_fTextureColorBrightness{L}", 1))
            ta = color_correct2(csb, m.tex_mean(f"g_tColor{L}"))
            m._mats[key] = (ta, ta @ color_tint2(m.V(f"g_vTextureColorTint{L}")[:3], 1.0))
        ta, tca = m._mats[key]
        cm = apply_matrix(col, ta)[:, :3] if m.Iv(f"g_nColorCorrectionMode{L}") == 1 else col[:, :3]
        cct = apply_matrix(col, tca)[:, :3]
        c = mix(cm, cct, tm[:, None])
        c = color_tint_colorize(c, tint_lin, tint_amt * tm * m.Iv(f"g_bModelTint{L}", 1))
        return c, col[:, 3], hgt, tm

    c1, a1, h1, tm1 = layer(1)
    if not blend_shader:
        c1 = c1 * mix(1.0, vert_color, h1[:, 1:2])
        return c1, (a1 if m.alphatest else np.ones(h.n))

    blend = h.blend.copy()
    if m.Iv("F_BLEND_BY_FACING_DIRECTION_2") > 0:
        fd = m.V("g_vFacingDirection2", (0, 0, 1, 0))[:3].copy()
        if fd[2] == 0:
            fd[2] += 1e-4
        fd /= np.linalg.norm(fd)
        spread, fall = m.F("g_flFacingDirectionMaskSpread2", 0.5), m.F("g_vFacingDirectionMaskFalloff2", 0.1)
        mn, mx = max(0.0, 1 - spread - fall), min(1.0, 1 - spread + 0.001 + fall)
        blend[:, 0] *= smoothstep(mn, mx, h.nrm @ fd * 0.5 + 0.5)
    if layer3 and m.Iv("F_BLEND_BY_FACING_DIRECTION_3") == 1:
        fd = m.V("g_vFacingDirection3", (0, 0, 1, 0))[:3].copy()
        if fd[2] == 0:
            fd[2] += 1e-4
        fd /= np.linalg.norm(fd)
        spread, fall = m.F("g_flFacingDirectionMaskSpread3", 0.5), m.F("g_vFacingDirectionMaskFalloff3", 0.1)
        mn, mx = max(0.0, 1 - spread - fall), min(1.0, 1 - spread + 0.001 + fall)
        blend[:, 1] *= smoothstep(mn, mx, h.nrm @ fd * 0.5 + 0.5)
    soft = np.full(h.n, m.F("g_flBlendSoftness2", 0.01))
    if layer3:
        s3 = m.F("g_flBlendSoftness3", 0.01)
        soft = np.where(blend[:, 0] < 0.001, s3, soft)
        soft = mix(soft, s3, blend[:, 1])
    blend[:, 3] = np.clip(blend[:, 3] + soft, 0.001, 1)

    c2, a2, h2, tm2 = layer(2)
    zp1, zp2 = m.F("g_flHeightMapZeroPoint1", 0.5), m.F("g_flHeightMapZeroPoint2", 0.5)
    sc1, sc2 = m.F("g_flHeightMapScale1", 1), m.F("g_flHeightMapScale2", 1)
    base1, base2 = h1[:, 0] - zp1, h2[:, 0] - zp2
    bf2 = sat(blend[:, 0] * 1.1 - 0.05)
    bf3 = sat(blend[:, 1] * 1.1 - 0.05)
    soft2 = blend[:, 3]
    new = m.Iv("F_USE_NEW_BLENDING") == 1
    carry2 = None
    if new:
        carry1 = (np.zeros(h.n), np.zeros(h.n), np.zeros(h.n), base1 * sc1)
        w2, carry2 = blend_layer(carry1, bf2, h2[:, 0], zp2, sc2, soft2,
                                 m.F("g_flUnderlyingHeightMapInfluence2", 1), m.F("g_flMaskWithHeight2", 0))
    else:
        _, w2 = get_blend_weights((base1, base2), (sc1, sc2), (zp1, zp2), blend[:, 0], soft2)
    c = mix(c1 * mix(1.0, l2s(sat(c2)) * 2, (w2 * m.F("g_flColorOverlay2", 0))[:, None]) if m.F("g_flColorOverlay2", 0) > 0 else c1,
            c2, (w2 * m.F("g_flColorReplace2", 1))[:, None])
    w3 = np.zeros(h.n)
    tm3 = np.zeros(h.n)
    if layer3:
        c3, a3, h3, tm3 = layer(3)
        zp3, sc3 = m.F("g_flHeightMapZeroPoint3", 0.5), m.F("g_flHeightMapScale3", 1)
        soft3 = blend[:, 3]
        if new:
            w3, _ = blend_layer(carry2, bf3, h3[:, 0], zp3, sc3, soft3,
                                m.F("g_flUnderlyingHeightMapInfluence3", 1), m.F("g_flMaskWithHeight3", 0))
        else:
            hb = base1 * (1 - w2) + base2 * w2
            _, w3 = get_blend_weights((hb, h3[:, 0] - zp3), (max(sc1, sc2), sc3), (max(zp1, zp2), zp3), bf3, soft3)
        ov3 = m.F("g_flColorOverlay3", 0)
        c = mix(c * mix(1.0, l2s(sat(c3)) * 2, (w3 * ov3)[:, None]) if ov3 > 0 else c, c3, (w3 * m.F("g_flColorReplace3", 1))[:, None])
    share2 = w2 * (1 - w3)
    share1 = sat(1 - share2 - w3)

    if m.Iv("F_SHARED_COLOR_OVERLAY") == 1:
        ms = m.V("g_vColorOverlayTintMaskStrengths", (0, 0, 0, 0))
        ls = m.V("g_vColorOverlayLayerStrengths", (1, 1, 1, 1))
        def amount(tm, k, share):
            mask = tm if ms[k] > 0 else 1 - tm
            return mix(1.0, mask, abs(ms[k])) * ls[k] * share
        amt = amount(tm1, 0, share1) + amount(tm2, 1, share2)
        if layer3:
            amt = amt + amount(tm3, 2, w3)
        if (amt > 0).any():
            uvset = m.Iv("g_nColorOverlayUVSet", 2)
            src, dens = (h.uv2, h.d2) if uvset == 2 else (h.uv0, h.d0)
            oscale = m.V("g_vOverlayTexCoordScale")[:2]
            tc = rotate2d(src, m.F("g_flOverlayTexCoordRotation"), oscale, m.V("g_vOverlayTexCoordOffset", (0, 0, 0, 0))[:2],
                          m.V("g_vOverlayTexCoordCenter", (0.5, 0.5, 0, 0))[:2])
            o = S.sample("g_tSharedColorOverlay", uvset, tc, dens * np.abs(oscale).max(), oscale, (0.5, 0.5, 0.5, 1))[:, :3] * 2 - 1
            bc, dc = m.F("g_flOverlayBrightnessContrast", 1), m.F("g_flOverlayDarknessContrast", 1)
            fac = np.maximum(0, 1 + (1 - np.power(np.maximum(1 - np.maximum(0, o), 0), bc)) * bc
                             + (np.power(np.maximum(1 + np.minimum(0, o), 0), dc) - 1) * dc)
            c = c * mix(1.0, fac, amt[:, None])

    vm = [m.Iv(f"g_nVertexColorMode{L}", 0) for L in (1, 2, 3)]
    masked = tm1 * share1 + tm2 * share2
    unmasked = (vm[0] != 0) * share1 + (vm[1] != 0) * share2
    enabled = (vm[0] != 2) * share1 + (vm[1] != 2) * share2
    if layer3:
        masked = masked + tm3 * w3
        unmasked = unmasked + (vm[2] != 0) * w3
    vmask = sat(unmasked + masked) * enabled
    if layer3:
        vmask = vmask + (vm[2] != 2) * w3
    c = c * mix(1.0, vert_color, sat(vmask)[:, None])
    return c, (a1 if m.alphatest else np.ones(h.n))


def evaluate(m, h):
    if m.env:
        rgb, a = eval_env(m, h)
    else:
        rgb, a = eval_complex(m, h)
    return np.clip(np.nan_to_num(rgb), 0, 1), np.clip(np.nan_to_num(a, nan=1.0), 0, 1)


# ---------------------------------------------------------------- render world

def _uv_density(uvt, wt):
    """uv units per hammer unit for triangles given uv (k,3,2) and world (k,3,3)."""
    e1, e2 = uvt[:, 1] - uvt[:, 0], uvt[:, 2] - uvt[:, 0]
    a_uv = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    a_w = np.linalg.norm(np.cross(wt[:, 1] - wt[:, 0], wt[:, 2] - wt[:, 0]), axis=1)
    return np.sqrt(a_uv / np.maximum(a_w, 1e-9))


class RenderWorld:
    def __init__(self, world_glb, textures, log=print):
        self.doc, self.bin = G.load(world_glb)
        doc = self.doc
        self.textures = textures
        self.materials = [Material(m, self.textures) for m in doc["materials"]]
        pos, tris, inst = [], [], []
        voff = toff = 0
        pos_cache = {}
        used_cache = {}
        self.attr_cache = {}
        self.idx_cache = {}
        skipped = 0
        for ni, M in G.node_matrices(doc).items():
            node = doc["nodes"][ni]
            if "mesh" not in node:
                continue
            flip = np.linalg.det(M[:3, :3]) < 0
            for pi, pr in enumerate(doc["meshes"][node["mesh"]]["primitives"]):
                if pr.get("mode", 4) != 4 or "material" not in pr:
                    continue
                mat = self.materials[pr["material"]]
                if mat.skip:
                    skipped += 1
                    continue
                pa = pr["attributes"]["POSITION"]
                p = pos_cache.get(pa)
                if p is None:
                    p = pos_cache[pa] = G.accessor(doc, self.bin, pa).astype(np.float64)
                idx = self._indices(pr, len(p))
                # S2V primitives often index a small slice of a vertex buffer shared
                # by the whole mesh; only transform the vertices this primitive uses.
                ukey = (pa, pr.get("indices"))
                used = used_cache.get(ukey)
                if used is None:
                    u, local = np.unique(idx, return_inverse=True)
                    used = used_cache[ukey] = (u, local.reshape(idx.shape).astype(np.uint32))
                u, local = used
                w = p[u] @ M[:3, :3].T + M[:3, 3]
                w = w[:, [2, 0, 1]] / HU
                if flip:
                    local = local[:, ::-1]
                pos.append(w.astype(np.float32))
                tris.append(local + np.uint32(voff))
                inst.append((node["mesh"], pi, pr["material"], toff, len(idx)))
                voff += len(u)
                toff += len(idx)
        del pos_cache, used_cache
        self.positions = np.concatenate(pos)
        self.triangles = np.concatenate(tris)
        self.inst = inst
        self.inst_start = np.array([i[3] for i in inst], np.int64)
        self.inst_mat = np.array([i[2] for i in inst], np.int64)
        log(f"  render world: {len(self.triangles):,} triangles in {len(inst):,} instances ({skipped:,} skipped prims)")

    def _indices(self, pr, nverts):
        key = pr.get("indices", ("n", nverts))
        idx = self.idx_cache.get(key)
        if idx is None:
            idx = (G.accessor(self.doc, self.bin, pr["indices"]).ravel() if "indices" in pr else np.arange(nverts)).reshape(-1, 3).astype(np.int64)
            self.idx_cache[key] = idx
        return idx

    def _attr(self, pr, name):
        a = pr["attributes"].get(name)
        if a is None:
            return None
        v = self.attr_cache.get(a)
        if v is None:
            v = self.attr_cache[a] = G.accessor(self.doc, self.bin, a).astype(np.float32)
        return v

    def _uv2_slot(self, pr, idx, start):
        at = pr["attributes"]
        if "TEXCOORD_2" in at:
            return "TEXCOORD_1"
        if "TEXCOORD_1" not in at:
            return None
        key = ("uv2slot", at["TEXCOORD_1"], pr.get("indices"))
        r = self.attr_cache.get(key, 0)
        if r == 0:
            tri_world = self.positions[self.triangles[start:start + len(idx)]].astype(np.float64)
            uv = self._attr(pr, "TEXCOORD_1")[idx]
            e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
            a_uv = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
            a_w = np.linalg.norm(np.cross(tri_world[:, 1] - tri_world[:, 0], tri_world[:, 2] - tri_world[:, 0]), axis=1).sum() * HU * HU
            r = self.attr_cache[key] = "TEXCOORD_1" if a_uv / max(a_w, 1e-9) > 1e-3 else None
        return r

    def hit_attributes(self, tri, bu, bv):
        """tri: global triangle ids, (bu, bv): barycentrics of vertices 1 and 2."""
        n = len(tri)
        H = Hits(n)
        inst = np.searchsorted(self.inst_start, tri, side="right") - 1
        wt = self.positions[self.triangles[tri]].astype(np.float64)
        b = np.stack([1 - bu - bv, bu, bv], 1)
        H.pos = np.einsum("ij,ijk->ik", b, wt)
        nn = np.cross(wt[:, 1] - wt[:, 0], wt[:, 2] - wt[:, 0])
        H.nrm = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        order = np.argsort(inst, kind="stable")
        bounds = np.flatnonzero(np.diff(inst[order])) + 1
        for sel in np.split(order, bounds):
            if len(sel) == 0:
                continue
            mesh, pi, _, start, _ = self.inst[inst[sel[0]]]
            pr = self.doc["meshes"][mesh]["primitives"][pi]
            idx_all = self._indices(pr, None if "indices" in pr else len(self._attr(pr, "POSITION")))
            local = idx_all[tri[sel] - start]
            bs = b[sel]
            def interp(name):
                a = self._attr(pr, name)
                if a is None:
                    return None, None
                t = a[local]
                return np.einsum("ij,ijk->ik", bs, t), t
            uv0, uvt = interp("TEXCOORD_0")
            if uv0 is not None:
                H.uv0[sel] = uv0
                H.d0[sel] = _uv_density(uvt.astype(np.float64), wt[sel])
            slot = self._uv2_slot(pr, idx_all, start)
            if slot:
                uv2, uvt2 = interp(slot)
                H.uv2[sel] = uv2
                H.d2[sel] = _uv_density(uvt2.astype(np.float64), wt[sel])
            else:
                H.uv2[sel] = H.uv0[sel]
                H.d2[sel] = H.d0[sel]
            vc, _ = interp("COLOR_0")
            if vc is not None:
                H.vcol[sel, :vc.shape[1]] = vc
                H.has_vcol[sel] = True
            bl, _ = interp("_TEXCOORD_4")
            if bl is not None:
                H.blend[sel, :bl.shape[1]] = bl
        return H, inst

    def evaluate(self, tri, bu, bv):
        """Returns (rgb linear (N,3), opacity (N,), role (N,), double_sided (N,), render normal (N,3))."""
        H, inst = self.hit_attributes(tri, bu, bv)
        mat_ids = self.inst_mat[inst]
        rgb = np.zeros((len(tri), 3))
        alpha = np.ones(len(tri))
        role = np.zeros(len(tri), np.int8)
        alphatest = np.zeros(len(tri), bool)
        ds = np.zeros(len(tri), bool)
        for mid in np.unique(mat_ids):
            sel = np.flatnonzero(mat_ids == mid)
            m = self.materials[mid]
            c, a = evaluate(m, H.take(sel))
            rgb[sel] = c
            alpha[sel] = a
            role[sel] = m.role
            alphatest[sel] = m.alphatest
            ds[sel] = m.double_sided or m.shader == "csgo_foliage.vfx"
        return rgb, alpha, role, alphatest, ds, H.nrm
