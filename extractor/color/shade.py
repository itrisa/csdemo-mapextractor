"""Transfer render-world albedo onto remeshed collision layers as per-vertex colours.

Every collision triangle is sampled at three interior points; rays cast along the collision normal find the
render surface it represents, whose linear albedo comes from the VRF shader math in albedo.py. Decals and
translucent sheets just above that surface are composited on top.
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from . import albedo as A

PROBE = 48.0          # ray starts this far in front of the collision surface
FALLBACK_PROBE = 160.0
NEAR = 12.0           # render surfaces within this distance count as "the" surface
OVERLAY_BEHIND, OVERLAY_FRONT = 1.0, 4.0
FACING = 0.2
CHUNK = 150_000
BARY = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])


def gather_hits(scene, origins, dirs):
    rays = o3d.core.Tensor(np.concatenate([origins, dirs], 1).astype(np.float32))
    r = scene.list_intersections(rays)
    splits = r["ray_splits"].numpy().astype(np.int64)
    ray = np.repeat(np.arange(len(origins)), np.diff(splits))
    t = r["t_hit"].numpy().astype(np.float64)
    prim = r["primitive_ids"].numpy().astype(np.int64)
    uv = r["primitive_uvs"].numpy().astype(np.float64)
    return ray, t, prim, uv


def shade_samples(world, scene, S, N, probe=PROBE, any_side=False):
    """S: sample points (n,3), N: facing normals (n,3). Returns linear rgb (n,3) and found mask (n,)."""
    rgb_out = np.zeros((len(S), 3))
    found = np.zeros(len(S), bool)
    for c0 in range(0, len(S), CHUNK):
        s, n = S[c0:c0 + CHUNK], N[c0:c0 + CHUNK]
        ray, t, prim, uv = gather_hits(scene, s + n * probe, -n)
        d = probe - t
        keep = (d >= -probe) & (d <= probe)
        ray, d, prim, uv = ray[keep], d[keep], prim[keep], uv[keep]
        if len(ray) == 0:
            continue
        rgb, alpha, role, alphatest, ds, rn = world.evaluate(prim, uv[:, 0], uv[:, 1])
        facing = np.einsum("ij,ij->i", rn, n[ray])
        ok = (facing > FACING) | ((ds | any_side) & (facing < -FACING))
        ref = np.array([world.materials[m].F("g_flAlphaTestReference", 0.5) for m in world.inst_mat[
            np.searchsorted(world.inst_start, prim, side="right") - 1]]) if alphatest.any() else 0.5
        is_base = ok & (role == A.BASE) & (~alphatest | (alpha >= ref))
        is_over = ok & (role != A.BASE) & (alpha > 0.004)

        # base: outermost surface within NEAR, else the closest one
        nr = len(s)
        score = np.where(np.abs(d) <= NEAR, 1000 + d, -np.abs(d))
        score = np.where(is_base, score, -np.inf)
        order = np.lexsort((-score, ray))
        first = np.ones(len(order), bool)
        first[1:] = ray[order][1:] != ray[order][:-1]
        cand = order[first]
        cand = cand[np.isfinite(score[cand])]
        base_idx = np.full(nr, -1)
        base_idx[ray[cand]] = cand
        # glass/translucent-only spots: use the nearest alpha overlay as the base
        lone = (base_idx < 0)
        alt = is_over & (role == A.OVER_ALPHA) & lone[ray]
        if alt.any():
            sc = np.where(alt, -np.abs(d), -np.inf)
            o2 = np.lexsort((-sc, ray))
            f2 = np.ones(len(o2), bool)
            f2[1:] = ray[o2][1:] != ray[o2][:-1]
            c2 = o2[f2]
            c2 = c2[np.isfinite(sc[c2])]
            base_idx[ray[c2]] = c2
            is_over[c2] = False
            alpha[c2] = 1.0
        has = base_idx >= 0
        col = np.zeros((nr, 3))
        col[has] = rgb[base_idx[has]]
        bd = np.full(nr, np.nan)
        bd[has] = d[base_idx[has]]

        # overlays (decals, translucent sheets) just above the base, composited far to near
        ov = is_over & has[ray]
        ov &= (d >= bd[ray] - OVERLAY_BEHIND) & (d <= bd[ray] + OVERLAY_FRONT)
        oi = np.flatnonzero(ov)
        oi = oi[np.lexsort((d[oi], ray[oi]))]
        for k in oi:
            r_, a_, c_ = ray[k], alpha[k], rgb[k]
            if role[k] == A.OVER_ALPHA:
                col[r_] = col[r_] + (c_ - col[r_]) * a_
            elif role[k] == A.OVER_MOD2X:
                col[r_] = col[r_] * (1 + (2 * A.l2s(c_) - 1) * a_)
            else:
                col[r_] = col[r_] * (1 + (c_ - 1) * a_)
        rgb_out[c0:c0 + nr] = np.clip(col, 0, 1)
        found[c0:c0 + nr] = has
    return rgb_out, found


def color_layer(world, scene, V, T, N):
    """V, T: remeshed layer, N: per-triangle plane normal. Returns (V, T, rgba uint8, sample hit rate)."""
    tri = V[T]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    S = np.einsum("bk,tkc->tbc", BARY, tri).reshape(-1, 3)
    rgb, found = shade_samples(world, scene, S, np.repeat(N, 3, 0))
    miss = np.flatnonzero(~found)
    if len(miss):
        # second chance for surfaces whose render counterpart is further away or single-sided the other way
        r2, f2 = shade_samples(world, scene, S[miss], np.repeat(N, 3, 0)[miss], probe=FALLBACK_PROBE, any_side=True)
        rgb[miss[f2]] = r2[f2]
        found[miss[f2]] = True
    rgb, found = rgb.reshape(-1, 3, 3), found.reshape(-1, 3)
    acc = np.zeros((len(V), 3))
    wsum = np.zeros(len(V))
    for b in range(3):
        w = BARY[b][None, :] * area[:, None] * found[:, b:b + 1]
        for k in range(3):
            np.add.at(acc, T[:, k], rgb[:, b] * w[:, k:k + 1])
            np.add.at(wsum, T[:, k], w[:, k])
    have = wsum > 0
    col = np.zeros((len(V), 3))
    col[have] = acc[have] / wsum[have, None]
    # fill gaps from neighbours within the same group (groups are not connected to each other)
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    for _ in range(64):
        miss = ~have
        if not miss.any():
            break
        src = have[e[:, 0]] & miss[e[:, 1]]
        if not src.any():
            break
        s_acc = np.zeros((len(V), 3))
        s_n = np.zeros(len(V))
        np.add.at(s_acc, e[src, 1], col[e[src, 0]])
        np.add.at(s_n, e[src, 1], 1)
        new = s_n > 0
        col[new] = s_acc[new] / s_n[new, None]
        have |= new
    if (~have).any() and have.any():
        dist, j = cKDTree(V[have]).query(V[~have], distance_upper_bound=64.0)
        ok = np.isfinite(dist)
        idx = np.flatnonzero(~have)[ok]
        col[idx] = col[have][j[ok]]
        have[idx] = True
    rgba = np.zeros((len(V), 4), np.uint8)
    rgba[:, :3] = np.round(np.clip(col, 0, 1) * 255)
    rgba[:, 3] = np.where(have, 255, 0)
    return V, T, rgba, found.mean()
