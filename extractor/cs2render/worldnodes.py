"""Baked lighting mode of the world's aggregate fragments, read from the map's world nodes (.vwnod_c).

CS2 lights some aggregate fragments from light probes instead of the lightmap (their draw calls carry no
lightmap uv). The glTF export doesn't say which, and their other uv sets can lie in [0, 1], so without this
they would sample the lightmap at unrelated coordinates."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from .. import source2viewer
from . import kv3


def probe_lit_meshes(text: str) -> set[str]:
    """glTF mesh names (`<model>_fragment<n>`, as Source2Viewer names them) of probe-lit aggregate fragments."""
    root = kv3.loads(text)
    names: set[str] = set()
    for scene_object in root.get("m_aggregateSceneObjects") or []:
        model = scene_object.get("m_renderableModel")
        if not model:
            continue
        stem = PurePosixPath(str(model)).stem
        fragments: list[dict[str, Any]] = scene_object.get("m_aggregateMeshes") or []
        for number, fragment in enumerate(fragments, start=1):
            if int(fragment.get("m_nLightProbeVolumePrecomputedHandshake") or 0) != 0:
                names.add(f"{stem}_fragment{number}")
    return names


def _world_node_resources(executable: Path, vpk: Path, map_name: str) -> list[str]:
    completed = subprocess.run(
        [str(executable), "-i", str(vpk), "-l", "-f", f"maps/{map_name}/worldnodes/", "-e", "vwnod_c"],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return sorted(
        line.partition(" CRC:")[0].strip().replace("\\", "/")
        for line in completed.stdout.splitlines()
        if " CRC:" in line and ".vwnod_c" in line
    )


def load(executable: Path, cs2_dir: Path, map_name: str, work_dir: Path) -> frozenset[str]:
    """Probe-lit fragment mesh names, cached in work_dir; empty when the world nodes can't be read."""
    cache = work_dir / "probe_lit_meshes.json"
    if cache.is_file():
        return frozenset(json.loads(cache.read_text(encoding="utf-8")))
    vpk = (cs2_dir / "game" / "csgo" / "maps" / f"{map_name}.vpk").resolve()
    names: set[str] = set()
    try:
        for resource in _world_node_resources(executable, vpk, map_name):
            names |= probe_lit_meshes(source2viewer.extract_resource_block(executable, vpk, resource))
    except Exception as error:  # noqa: BLE001 - fall back to the uv-based lightmap detection
        print(f"[cs2render] world nodes unavailable for {map_name}: {error}", flush=True)
        return frozenset()
    cache.write_text(json.dumps(sorted(names)), encoding="utf-8")
    return frozenset(names)
