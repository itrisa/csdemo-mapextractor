"""Crack- and overlap-free refinement of collision layers for per-vertex coloring.

Pipeline (all layers share one boundary registry so cracks between layers are repaired too):
  1. weld near-identical vertices and drop degenerate triangles
  2. group coplanar same-facing triangles of a layer into plane groups
  3. per group: overlap-free union of its triangles (GEOS polygon union)
  4. globally split every group boundary edge at any other boundary vertex lying on it
     (removes T-junctions) and then into equal pieces no longer than max_edge
  5. per group: conforming quality refinement (Triangle) that never adds boundary points
"""
import multiprocessing
import time
from types import SimpleNamespace

import numpy as np
import shapely
import triangle as tr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

WELD_EPS = 0.01
ON_EDGE_TOL = 0.05
NORMAL_CELL = 1 / 500
PLANE_GAP = 0.25
GRID = 1e-4
REFINE_FLAGS = "pcYYQa{area:.3f}"


def merge_points(P, eps):
    """Union points closer than eps. Returns (representative positions, label per input point).

    Components are labelled in order of their first input point, which is also their representative."""
    n = len(P)
    # exact duplicates are common (shared vertices), so build the eps graph over distinct points only
    Q, inv = np.unique(P, axis=0, return_inverse=True)
    inv = inv.ravel()
    pairs = cKDTree(Q).query_pairs(eps, output_type="ndarray")
    if len(pairs) == 0:
        k, lab_q = len(Q), np.arange(len(Q))
    else:
        g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(Q), len(Q)))
        k, lab_q = connected_components(g, directed=False)
    lab = lab_q[inv]
    first = np.full(k, n, np.int64)
    np.minimum.at(first, lab, np.arange(n))
    order = np.argsort(first)
    rank = np.empty(k, np.int64)
    rank[order] = np.arange(k)
    return P[first[order]], rank[lab]


def weld(P, F):
    Pw, lab = merge_points(np.asarray(P, np.float64), WELD_EPS)
    F = lab[np.asarray(F, np.int64)]
    T = Pw[F]
    area2 = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2]) & (area2 > 1e-6)
    return Pw, F[ok]


def plane_groups(P, F):
    """Cluster triangles into planes (facing-independent). Returns (plane id, side +-1, canonical normal)."""
    T = P[F]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    ax = np.argmax(np.abs(n), 1)
    side = np.sign(n[np.arange(len(n)), ax])
    un = n * side[:, None]
    d = np.einsum("ij,ij->i", un, T.mean(1))
    _, nid = np.unique(np.round(un / NORMAL_CELL).astype(np.int64), axis=0, return_inverse=True)
    nid = nid.ravel()
    order = np.lexsort((d, nid))
    ds = d[order]
    brk = np.ones(len(order), bool)
    brk[1:] = (nid[order][1:] != nid[order][:-1]) | (np.diff(ds) > PLANE_GAP)
    # gap chaining alone can link surfaces at many heights (floor, curb, step...) into one "plane" spanning
    # tens of units, which the 2D union then flattens; split such runs at their largest gaps until each
    # spans at most PLANE_GAP
    starts = np.flatnonzero(brk)
    ends = np.append(starts[1:], len(order))
    stack = [(s, e) for s, e in zip(starts, ends) if ds[e - 1] - ds[s] > PLANE_GAP]
    while stack:
        s, e = stack.pop()
        k = s + 1 + int(np.argmax(np.diff(ds[s:e])))
        brk[k] = True
        for a, b in ((s, k), (k, e)):
            if ds[b - 1] - ds[a] > PLANE_GAP:
                stack.append((a, b))
    pid = np.empty(len(order), np.int64)
    pid[order] = np.cumsum(brk) - 1
    return pid, side, un


def basis(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(n, e1)])


def _rings(geom):
    for poly in shapely.get_parts(geom):
        if shapely.get_type_id(poly) != 3 or poly.area < 1e-6:
            continue
        yield poly, True, np.asarray(poly.exterior.coords)[:-1]
        for ring in poly.interiors:
            yield poly, False, np.asarray(ring.coords)[:-1]


def _boundary(origin, B, V3, V2, U):
    """Directed boundary edges (A -> B) of U in 3D, snapped back onto the input vertices they came from."""
    tree = cKDTree(V2)
    A, Bv = [], []
    for poly, outer, ring in _rings(U):
        if len(ring) < 3:
            continue
        d, j = tree.query(ring, distance_upper_bound=10 * GRID)
        hit = np.isfinite(d)
        r3 = origin + ring @ B
        r3[hit] = V3[j[hit]]
        A.append(r3)
        Bv.append(np.roll(r3, -1, axis=0))
    if not A:
        return None
    return np.concatenate(A), np.concatenate(Bv)


def _plane_input(P, F, idx):
    vid, Floc = np.unique(F[idx], return_inverse=True)
    return P[vid], Floc.reshape(-1, 3)


def union_pass(P, F, idx, side, un):
    """Overlap-free union per facing of one plane; where both facings overlap only the front (+) side is kept,
    since the client renders double-sided and coincident opposite faces would z-fight.

    Returns (origin, 2D basis, [(facing sign, union polygon, boundary (A, B))])."""
    V3, Floc = _plane_input(P, F, idx)
    origin = V3.mean(0)
    B = basis(un)
    V2 = (V3 - origin) @ B.T
    polys = shapely.polygons(V2[Floc])
    out = []
    U_pos = None
    for s in (1, -1):
        sel = side == s
        if not sel.any():
            continue
        U = shapely.union_all(polys[sel], grid_size=GRID) if sel.sum() > 1 else shapely.set_precision(polys[sel][0], GRID)
        if s == 1:
            U_pos = U
        elif U_pos is not None:
            U = shapely.difference(U, U_pos, grid_size=GRID)
        if U.is_empty or U.area < 1e-6:
            continue
        boundary = _boundary(origin, B, V3, V2, U)
        if boundary is not None:
            out.append((s, U, boundary))
    return origin, B, out


def _on_edge_points(tree, R, edges, probe_budget=1 << 20):
    """Registry points lying on the interior of each edge, as (edge id, point id, t) sorted by edge then t."""
    a, b = R[edges[:, 0]], R[edges[:, 1]]
    d = b - a
    L = np.linalg.norm(d, axis=1)
    # cover each edge with probe spheres around k equal sub-segments, each just large enough to contain
    # every point within ON_EDGE_TOL of its sub-segment, so dense surroundings aren't pulled in
    rad = 8.0
    k = np.maximum(1, np.ceil(L / rad)).astype(np.int64)
    radius = L / (2 * k) + 2 * ON_EDGE_TOL
    # bounded chunks keep the candidate lists (Python objects) small on dense maps
    cuts = np.searchsorted(np.cumsum(k), np.arange(probe_budget, k.sum(), probe_budget), side="right")
    E, P, T = [], [], []
    for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(edges)]):
        if lo == hi:
            continue
        kc = k[lo:hi]
        se = np.repeat(np.arange(lo, hi), kc)
        st = (np.arange(kc.sum()) - np.repeat(np.cumsum(kc) - kc, kc) + 0.5) / k[se]
        probes = a[se] + st[:, None] * d[se]
        cand = tree.query_ball_point(probes, radius[se], return_sorted=False)
        counts = np.fromiter((len(c) for c in cand), np.int64, len(cand))
        eid = np.repeat(se, counts)
        pid = np.fromiter((p for c in cand for p in c), np.int64, counts.sum())
        del cand
        pair = np.unique(np.stack([eid, pid], 1), axis=0)
        eid, pid = pair[:, 0], pair[:, 1]
        t = np.einsum("ij,ij->i", R[pid] - a[eid], d[eid]) / (L[eid] ** 2)
        off = np.linalg.norm(a[eid] + t[:, None] * d[eid] - R[pid], axis=1)
        on = (t * L[eid] > ON_EDGE_TOL) & ((1 - t) * L[eid] > ON_EDGE_TOL) & (off < ON_EDGE_TOL)
        E.append(eid[on])
        P.append(pid[on])
        T.append(t[on])
    if not E:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    eid, pid, t = np.concatenate(E), np.concatenate(P), np.concatenate(T)
    order = np.lexsort((t, eid))
    return eid[order], pid[order], t[order]


def split_edges(R, edges, max_edge):
    """Split unique undirected registry edges at the registry points on them, then into pieces <= max_edge.

    Returns flat (points (m,3), params t (m,), offsets (len(edges)+1,)); edge e's polyline along a->b is
    points[offsets[e]:offsets[e+1]] with t running from 0 to 1."""
    n = len(edges)
    eid, pid, t = _on_edge_points(cKDTree(R), R, edges)
    # chain nodes per edge: a, on-edge points by t, b
    per = np.bincount(eid, minlength=n) + 2
    noff = np.r_[0, np.cumsum(per)]
    NP = np.empty((noff[-1], 3))
    NT = np.empty(noff[-1])
    NP[noff[:-1]], NT[noff[:-1]] = R[edges[:, 0]], 0.0
    NP[noff[1:] - 1], NT[noff[1:] - 1] = R[edges[:, 1]], 1.0
    slot = noff[eid] + 1 + np.arange(len(eid)) - np.searchsorted(eid, eid)
    NP[slot], NT[slot] = R[pid], t
    del eid, pid, t, slot
    # each chain segment i -> i+1 becomes k equal pieces: k points from its start, then the edge's end node
    last = np.zeros(noff[-1], bool)
    last[noff[1:] - 1] = True
    s0 = np.flatnonzero(~last)
    seg_edge = np.repeat(np.arange(n), per - 1)
    P0, T0 = NP[s0], NT[s0]
    dP, dT = NP[s0 + 1] - P0, NT[s0 + 1] - T0
    k = np.maximum(1, np.ceil(np.linalg.norm(dP, axis=1) / max_edge)).astype(np.int64)
    kcum = np.cumsum(k) - k
    rep = np.repeat(np.arange(len(s0)), k)
    j = np.arange(k.sum()) - kcum[rep]
    frac = j / k[rep]
    m = k.sum() + n
    pts = np.empty((m, 3))
    ts = np.empty(m)
    dst = kcum[rep] + seg_edge[rep] + j
    pts[dst] = P0[rep] + dP[rep] * frac[:, None]
    ts[dst] = T0[rep] + dT[rep] * frac
    first = j == 0
    pts[dst[first]], ts[dst[first]] = P0, T0
    out_off = np.r_[0, np.cumsum(np.add.reduceat(k, np.r_[0, np.cumsum(per - 1)[:-1]]) + 1)]
    pts[out_off[1:] - 1], ts[out_off[1:] - 1] = NP[noff[1:] - 1], 1.0
    return pts, ts, out_off


def refine_pass(g, polylines, max_edge):
    """Conforming refinement given split boundary polylines [(pts3 (n,3), t (n,))] oriented as the group edges."""
    P3, P2, seg = [], [], []
    off = 0
    for pts, ts in polylines:
        e0 = (pts[0] - g.origin) @ g.B.T
        e1 = (pts[-1] - g.origin) @ g.B.T
        P3.append(pts)
        # interior split points sit exactly on the 2D edge so Triangle never intersects segments
        P2.append(e0 + ts[:, None] * (e1 - e0))
        n = len(pts)
        seg.append(np.stack([np.arange(off, off + n - 1), np.arange(off + 1, off + n)], 1))
        off += n
    P3, P2, seg = np.concatenate(P3), np.concatenate(P2), np.concatenate(seg)
    _, lab = merge_points(P2, 1e-6)
    keep_first = np.unique(lab, return_index=True)[1]
    P2, P3 = P2[keep_first], P3[keep_first]
    seg = lab[seg]
    seg = seg[seg[:, 0] != seg[:, 1]]
    seg = np.unique(np.sort(seg, 1), axis=0)
    area = max_edge * max_edge * 0.5
    data = {"vertices": P2, "segments": seg}
    try:
        r = tr.triangulate(data, REFINE_FLAGS.format(area=area))
    except Exception:
        return None, None
    if "triangles" not in r or len(r["triangles"]) == 0:
        return None, None
    V, T = r["vertices"], r["triangles"]
    # the hull is triangulated without hole carving (which leaks through any gap); keep what lies in the union
    c = V[T].mean(1)
    T = T[shapely.contains_xy(g.U, c[:, 0], c[:, 1])]
    if len(T) == 0:
        return None, None
    out3 = np.empty((len(V), 3))
    out3[: len(P3)] = P3
    out3[len(P3):] = g.origin + V[len(P3):] @ g.B
    used, T = np.unique(T, return_inverse=True)
    T = T.reshape(-1, 3)
    out3 = out3[used]
    tri = out3[T]
    nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = nn @ g.normal < 0
    T[flip] = T[flip][:, ::-1]
    return out3, T[np.linalg.norm(nn, axis=1) > 1e-7]


_LAYERS = None
UNION_BATCH = 256
FLUSH_GROUPS = 1 << 16


def _init_union(layers):
    global _LAYERS
    _LAYERS = layers


def _union_task(task):
    """Union a run of planes; results come back as a few flat arrays rather than per-group objects, which
    keeps the parent small on maps with millions of (mostly single-triangle) planes."""
    li, lo, hi = task
    Pw, Fw, order, starts, side, un = _LAYERS[li]
    keys, meta, wkb, As, Bs = [], [], [], [], []
    for p in range(lo, hi):
        idx = order[starts[p]:starts[p + 1]]
        n = un[idx].mean(0)
        n /= np.linalg.norm(n)
        origin, B, gs = union_pass(Pw, Fw, idx, side[idx], n)
        for s, U, (A, Bv) in gs:
            keys.append((li, p, s))
            meta.append(np.concatenate([n * s, origin, B.ravel()]))
            wkb.append(shapely.to_wkb(U))
            As.append(A)
            Bs.append(Bv)
    if not keys:
        return np.zeros((0, 3), np.int64), np.zeros((0, 12)), [], np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 3))
    return (np.array(keys, np.int64), np.stack(meta), wkb, np.array([len(a) for a in As], np.int64),
            np.concatenate(As), np.concatenate(Bs))


def _refine_task(task):
    li, p, s, meta, wkb, pls, max_edge = task
    U = shapely.from_wkb(wkb)
    shapely.prepare(U)
    g = SimpleNamespace(normal=meta[:3], origin=meta[3:6], B=meta[6:].reshape(2, 3), U=U)
    t_g = time.perf_counter()
    V, T = refine_pass(g, pls, max_edge) if pls else (None, None)
    dt = time.perf_counter() - t_g
    failed = V is None
    if failed:
        Pw, Fw, order, starts, side, _ = _LAYERS[li]
        idx = order[starts[p]:starts[p + 1]]
        V, Floc = _plane_input(Pw, Fw, idx)
        T = Floc[side[idx] == s]
    return V, T, dt, failed


def _pool_map(pool, fn, tasks, chunksize):
    if pool is None:
        return map(fn, tasks)
    return pool.imap(fn, tasks, chunksize=chunksize)


class _Accumulator:
    """Concatenates a stream of small arrays in bounded batches."""

    def __init__(self):
        self.chunks, self.pending = [], []

    def add(self, a):
        self.pending.append(a)
        if len(self.pending) >= FLUSH_GROUPS:
            self.flush()

    def flush(self):
        if self.pending:
            self.chunks.append(np.concatenate(self.pending))
            self.pending = []

    def result(self):
        self.flush()
        return np.concatenate(self.chunks) if len(self.chunks) != 1 else self.chunks[0]


def remesh_layers(layers, max_edge, log=print, workers=1):
    """layers: list of (P, F). Returns per layer (positions (k,3), tris (m,3), per-tri plane normal (m,3)),
    or None for a layer without surface. Groups are concatenated but not connected to each other.

    Per-group work runs in `workers` processes; results are identical to a serial run."""
    shared = []
    union_tasks = []
    for li, (P, F) in enumerate(layers):
        Pw, Fw = weld(P, F)
        gid, side, un = plane_groups(Pw, Fw)
        order = np.argsort(gid, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(gid[order])) + 1, len(order)]
        planes = len(starts) - 1
        shared.append((Pw, Fw, order, starts, side, un))
        union_tasks += [(li, lo, min(lo + UNION_BATCH, planes)) for lo in range(0, planes, UNION_BATCH)]
        log(f"  layer {li}: {len(Fw):,} welded tris, {planes:,} planes")
    pool = None
    if workers > 1:
        pool = multiprocessing.get_context().Pool(workers, initializer=_init_union, initargs=(shared,))
    else:
        _init_union(shared)
    try:
        keys, meta, counts, A, Bp = (_Accumulator() for _ in range(5))
        wkb = []
        for k, m, w, c, a, b in _pool_map(pool, _union_task, union_tasks, 1):
            keys.add(k)
            meta.add(m)
            wkb += w
            counts.add(c)
            A.add(a)
            Bp.add(b)
        keys, meta, counts, A, Bp = keys.result(), meta.result(), counts.result(), A.result(), Bp.result()
        R, lab = merge_points(np.concatenate([A, Bp]), 1e-3)
        ea, eb = lab[: len(A)], lab[len(A):]
        ok = ea != eb
        key = np.sort(np.stack([ea, eb], 1), 1)
        uniq, inv = np.unique(key[ok], axis=0, return_inverse=True)
        log(f"  registry {len(R):,} boundary vertices, {len(uniq):,} unique boundary edges")
        pl_pts, pl_t, pl_off = split_edges(R, uniq, max_edge)
        log(f"  split edges into {len(pl_t) - len(uniq):,} segments")
        edge_of = np.full(len(key), -1)
        edge_of[ok] = inv.ravel()
        del A, Bp, R, lab, eb, ok, key, inv
        group_off = np.r_[0, np.cumsum(counts)]

        def refine_tasks():
            for gi in range(len(keys)):
                pls = []
                for k in range(group_off[gi], group_off[gi + 1]):
                    e = edge_of[k]
                    if e < 0:
                        continue
                    pts, ts = pl_pts[pl_off[e]:pl_off[e + 1]], pl_t[pl_off[e]:pl_off[e + 1]]
                    pls.append((pts, ts) if ea[k] == uniq[e, 0] else (pts[::-1], 1 - ts[::-1]))
                li, p, s = keys[gi]
                yield li, p, s, meta[gi], wkb[gi], pls, max_edge

        out = [(_Accumulator(), _Accumulator(), _Accumulator(), [0]) for _ in layers]
        failed = 0
        for gi, (V, T, dt, fell_back) in enumerate(_pool_map(pool, _refine_task, refine_tasks(), 64)):
            li = keys[gi, 0]
            if dt > 2:
                log(f"    slow group layer {li}: {0 if fell_back else len(T)} tris, {dt:.1f}s")
            if (gi + 1) % 50000 == 0:
                log(f"    {gi + 1:,}/{len(keys):,} groups")
            failed += fell_back
            Vs, Ts, Ns, voff = out[li]
            Vs.add(V)
            Ts.add(T + voff[0])
            Ns.add(np.repeat(meta[gi, :3][None], len(T), 0))
            voff[0] += len(V)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _init_union(None)
    log(f"  refined {len(keys):,} groups ({failed} fell back to the welded input)")
    return [(Vs.result(), Ts.result(), Ns.result()) if Vs.chunks or Vs.pending else None for Vs, Ts, Ns, _ in out]
