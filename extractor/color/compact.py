"""Shrink a vertex-coloured collision mesh without visible change.

1. weld: vertices sharing a position (and layer) whose sRGB colours differ by at most WELD_TOL/255 per channel
   become one vertex with the averaged colour. Larger colour steps stay split, i.e. real colour edges stay sharp.
2. simplify: colour-aware edge collapse (meshoptimizer quadrics) over all layers at once. A collapse is only taken
   while the surface stays within GEOM_TOL units of the input and the colour within ~COLOR_TOL/255, so large
   flat-coloured areas become few triangles and detail is kept only where the colour actually changes. Remaining
   colour splits and layer boundaries are seams, which the simplifier keeps watertight.
3. report: every input vertex is projected onto the result to measure the real geometric and colour deviation.
"""
import ctypes
import time

import numpy as np
from meshoptimizer._loader import lib as _lib
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

WELD_TOL = 4.0
COLOR_TOL = 3.0
GEOM_TOL = 0.5
TILE = 256.0

# the bundled binding declares no ctypes signatures (floats fail, size_t would be truncated)
_size, _fp, _up = ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint)
_lib.meshopt_simplifyWithAttributes.restype = _size
_lib.meshopt_simplifyWithAttributes.argtypes = [_up, _up, _size, _fp, _size, _size, _fp, _size, _fp, _size,
                                                ctypes.POINTER(ctypes.c_ubyte), _size, ctypes.c_float,
                                                ctypes.c_uint, _fp]


def _ptr(a, t):
    return a.ctypes.data_as(ctypes.POINTER(t))


def l2s(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def weld(V, C, L, tol):
    """Merge same-position, same-layer vertices with sRGB colours within tol (/255, per channel)."""
    key = np.concatenate([V.astype(np.float32).view(np.uint32), L[:, None].astype(np.uint32)], 1)
    _, pid = np.unique(key, axis=0, return_inverse=True)
    pid = pid.ravel()
    srgb = l2s(C[:, :3] / 255.0) * 255.0
    feat = np.concatenate([pid[:, None] * 1e4, srgb, C[:, 3:4] * 1e2], 1)
    pairs = cKDTree(feat).query_pairs(tol, p=np.inf, output_type="ndarray")
    n = len(V)
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    k, lab = connected_components(g, directed=False)
    cnt = np.bincount(lab, minlength=k).astype(np.float64)
    col = np.zeros((k, 3))
    for c in range(3):
        col[:, c] = np.bincount(lab, C[:, c].astype(np.float64), k) / cnt
    first = np.empty(k, np.int64)
    first[lab[::-1]] = np.arange(n)[::-1]
    Cw = np.empty((k, 4), np.uint8)
    Cw[:, :3] = np.round(col)
    Cw[:, 3] = C[first, 3]
    return V[first], Cw, L[first], lab


def _simplify_one(P, attrs, I, lock, color_tol, geom_tol):
    lo = P.min(0)
    Pl = np.ascontiguousarray(P - lo, np.float32)
    scale = float(max((Pl.max(0)).max(), 1e-3))
    err = geom_tol / scale
    # attributes live in the same (normalised) error space as positions: weight w makes a colour step of
    # color_tol/255 cost as much as a geometric deviation of geom_tol
    w = err / (color_tol / 255.0)
    weights = np.array([w, w, w, w * 10], np.float32)
    I = np.ascontiguousarray(I, np.uint32)
    out = np.empty_like(I)
    res = ctypes.c_float(0)
    lk = np.ascontiguousarray(lock, np.uint8)
    n = _lib.meshopt_simplifyWithAttributes(
        _ptr(out, ctypes.c_uint), _ptr(I, ctypes.c_uint), len(I), _ptr(Pl, ctypes.c_float), len(Pl), 12,
        _ptr(attrs, ctypes.c_float), attrs.shape[1] * 4, _ptr(weights, ctypes.c_float), len(weights),
        _ptr(lk, ctypes.c_ubyte), 0, err, 0, ctypes.byref(res))
    return out[:n], res.value * scale


def _far_tris(V, T, idx, out_tris, limit):
    """Input triangles (subset idx of T) whose centroid or edge midpoints are farther than limit from the output."""
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(out_tris.astype(np.uint32)))
    tri = V[T[idx]]
    probes = np.stack([tri.mean(1), (tri[:, 0] + tri[:, 1]) / 2, (tri[:, 1] + tri[:, 2]) / 2,
                       (tri[:, 2] + tri[:, 0]) / 2], 1)
    d = sc.compute_distance(o3d.core.Tensor(probes.reshape(-1, 3).astype(np.float32))).numpy().reshape(-1, 4)
    return idx[(d > limit).any(1)]


def simplify(V, C, L, T, color_tol, geom_tol, tile=TILE, rounds=4, log=print):
    """Tiled so each simplifier call works at a local scale (float32 quadrics over a whole map are too coarse
    for sub-unit tolerances); vertices whose position is used by more than one tile are locked so tiles
    stay watertight against each other.

    Quadric error measures distance to the (infinite) planes of the input faces, so a small face whose
    vertices lie on its neighbours' extended planes can collapse away at "zero" error. After each pass every
    input triangle is checked against the output; vertices of triangles that ended up farther than
    2 * geom_tol are locked and their tiles redone (falling back to the unsimplified tile if that persists)."""
    attrs = np.ascontiguousarray(np.concatenate([l2s(C[:, :3] / 255.0), C[:, 3:4] / 255.0], 1), np.float32)
    cen = V[T].mean(1)
    cell = np.floor((cen - V.min(0)) / tile).astype(np.int64)
    _, tid = np.unique(cell, axis=0, return_inverse=True)
    tid = tid.ravel()
    _, pid = np.unique(V.astype(np.float32).view(np.uint32).reshape(-1, 3), axis=0, return_inverse=True)
    pid = pid.ravel()
    pt = np.unique(np.stack([pid[T].ravel(), np.repeat(tid, 3)], 1), axis=0)
    shared = np.bincount(pt[:, 0], minlength=pid.max() + 1) > 1
    plock = shared.copy()
    order = np.argsort(tid, kind="stable")
    bounds = np.flatnonzero(np.diff(tid[order])) + 1
    tiles = np.split(order, bounds)
    outs = [None] * len(tiles)
    worst = 0.0

    def run(k):
        nonlocal worst
        used, inv = np.unique(T[tiles[k]], return_inverse=True)
        I, e = _simplify_one(V[used], attrs[used], inv.ravel(), plock[pid[used]], color_tol, geom_tol)
        worst = max(worst, e)
        outs[k] = used[I.astype(np.int64)].reshape(-1, 3)

    dirty = np.arange(len(tiles))
    for rnd in range(rounds + 1):
        for k in dirty:
            run(k)
        check = np.concatenate([tiles[k] for k in dirty])
        far = _far_tris(V, T, check, np.concatenate(outs), 2 * geom_tol)
        log(f"    simplify round {rnd}: {len(dirty):,} tiles, {len(far):,} input tris beyond {2 * geom_tol:g} units")
        if not len(far):
            break
        dirty = np.unique(tid[far])
        if rnd == rounds:
            for k in dirty:
                outs[k] = T[tiles[k]]
            log(f"    {len(dirty)} tiles left unsimplified")
            break
        plock[pid[T[far]].ravel()] = True
    T2 = np.concatenate(outs)
    used, inv = np.unique(T2, return_inverse=True)
    return V[used], C[used], L[used], inv.reshape(-1, 3), worst


def deviation(Vsrc, Csrc, Tsrc, V, C, T, sample=400_000, seed=0):
    """Area-weighted random points on the input surface, ray-cast along their normal onto the result.
    Returns geometric offset (units) and sRGB colour difference (/255, max channel) of what a viewer would see."""
    import open3d as o3d
    rng = np.random.default_rng(seed)
    tri = Vsrc[Tsrc]
    nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(nn, axis=1)
    k = rng.choice(len(Tsrc), sample, p=area / area.sum())
    r1, r2 = rng.random(sample), rng.random(sample)
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    b = np.stack([1 - r1 - r2, r1, r2], 1)
    p = np.einsum("ij,ijk->ik", b, tri[k])
    n = nn[k] / area[k, None]
    src = np.einsum("ij,ijk->ik", b, Csrc[Tsrc[k]].astype(np.float64) / 255.0)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(T.astype(np.uint32)))
    probe = 1.0
    rays = np.concatenate([p + n * probe, -n], 1).astype(np.float32)
    q = scene.cast_rays(o3d.core.Tensor(rays))
    t = q["t_hit"].numpy().astype(np.float64)
    prim = q["primitive_ids"].numpy().astype(np.int64)
    uv = q["primitive_uvs"].numpy().astype(np.float64)
    hit = np.isfinite(t) & (t < 2 * probe)
    off = np.full(sample, np.inf)
    off[hit] = np.abs(t[hit] - probe)
    bb = np.stack([1 - uv[hit, 0] - uv[hit, 1], uv[hit, 0], uv[hit, 1]], 1)
    col = np.einsum("ij,ijk->ik", bb, C[T[prim[hit]]].astype(np.float64) / 255.0)
    ok = (src[hit, 3] > 0.5) & (col[:, 3] > 0.5)
    dc = np.abs(l2s(col[ok, :3]) - l2s(src[hit][ok, :3])).max(1) * 255
    return off, dc


def compact_layers(layers, weld_tol=WELD_TOL, color_tol=COLOR_TOL, geom_tol=GEOM_TOL, log=print, report=True):
    """layers: [(name, tags, V, T, rgba)]. Returns the same structure, welded and simplified."""
    t0 = time.time()
    Vs, Cs, Ls, Ts, off = [], [], [], [], 0
    for li, (_, _, V, T, C) in enumerate(layers):
        Vs.append(np.asarray(V, np.float64))
        Cs.append(np.asarray(C, np.uint8))
        Ls.append(np.full(len(V), li, np.int64))
        Ts.append(np.asarray(T, np.int64) + off)
        off += len(V)
    V0, C0, L0, T0 = np.concatenate(Vs), np.concatenate(Cs), np.concatenate(Ls), np.concatenate(Ts)
    V, C, L, lab = weld(V0, C0, L0, weld_tol)
    T = lab[T0]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
    log(f"  weld: {len(V0):,} -> {len(V):,} vertices ({time.time() - t0:.0f}s)")
    V2, C2, L2, T2, rerr = simplify(V, C, L, T, color_tol, geom_tol, log=log)
    log(f"  simplify: {len(T0):,} -> {len(T2):,} tris, {len(V):,} -> {len(V2):,} vertices, "
        f"reported error {rerr:.3f} ({time.time() - t0:.0f}s)")
    tl = L2[T2]
    mixed = int((tl[:, 0] != tl[:, 1]).sum() + (tl[:, 0] != tl[:, 2]).sum())
    if mixed:
        log(f"  warning: {mixed} triangles span layers")
    if report:
        off, dc = deviation(V0, C0, T0, V2, C2, T2)
        log(f"  deviation: surface missed {np.isinf(off).mean() * 100:.3f}%, offset p99.9 "
            f"{np.quantile(off[np.isfinite(off)], 0.999):.3f} units; colour mean {dc.mean():.2f} "
            f"p95 {np.quantile(dc, 0.95):.1f} p99 {np.quantile(dc, 0.99):.1f} /255")
    out = []
    for li, (name, tags, *_rest) in enumerate(layers):
        sel = tl[:, 0] == li
        if not sel.any():
            continue
        used, inv = np.unique(T2[sel], return_inverse=True)
        out.append((name, tags, V2[used], inv.reshape(-1, 3), C2[used]))
    return out
