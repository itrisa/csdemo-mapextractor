"""Colour a map bundle into the GLB csdemo-client renders."""

from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np

from . import bundle as B
from .compact import compact_layers
from .encode import encode
from .remesh import remesh_layers

MAX_EDGE = 64.0


def colorize(bundle_dir: Path, *, workers: int | None = None, log=print) -> bytes:
    import open3d as o3d

    from .shade import color_layer

    workers = workers or os.cpu_count() or 1
    t0 = time.time()
    map_name, layers, open_world = B.load(bundle_dir, log)
    log(f"{map_name}: {len(layers)} layers, {sum(len(F) for *_, F in layers):,} collision tris")
    remeshed = remesh_layers([(P, F) for _, _, P, F in layers], MAX_EDGE, log, workers=workers)
    log(f"remeshed ({time.time() - t0:.0f}s)")
    world = open_world()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(world.positions), o3d.core.Tensor(world.triangles))
    log(f"render world ready ({time.time() - t0:.0f}s)")
    colored = []
    for li, (name, tags, _, _) in enumerate(layers):
        mesh, remeshed[li] = remeshed[li], None
        if mesh is None:
            continue
        V, T, rgba, hit = color_layer(world, scene, *mesh)
        del mesh
        log(
            f"  {name}: {len(T):,} tris, {hit * 100:.1f}% samples hit, "
            f"{(rgba[:, 3] == 0).mean() * 100:.1f}% verts uncoloured ({time.time() - t0:.0f}s)"
        )
        # positions pass through float32 between stages, as the coloured GLB stores them
        colored.append((name, tags, V.astype(np.float32).astype(np.float64), T, rgba))
    del world, scene, remeshed
    compacted = compact_layers(colored, log=log, report=False)
    data = encode(compacted, map_name)
    log(f"encoded {len(data) / 1e6:.2f} MB ({time.time() - t0:.0f}s)")
    return data
