"""Stable adapter around ValveResourceFormat's Source2Viewer-CLI."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


_MAP_NAME = re.compile(r"[A-Za-z0-9_]+")


def resolve(executable: Path | None = None) -> Path:
    candidates: list[Path] = []
    if executable is not None:
        candidates.append(executable)
    if configured := os.environ.get("SOURCE2VIEWER_CLI"):
        candidates.append(Path(configured))
    candidates.extend((Path.cwd() / "tools" / "Source2Viewer-CLI.exe", Path.cwd() / "tools" / "Source2Viewer-CLI"))
    if located := shutil.which("Source2Viewer-CLI"):
        candidates.append(Path(located))
    if located := shutil.which("Source2Viewer-CLI.exe"):
        candidates.append(Path(located))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Source2Viewer-CLI was not found; pass --source2viewer, set SOURCE2VIEWER_CLI, "
        "install it under tools/, or add it to PATH"
    )


def version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = completed.stdout.strip() or completed.stderr.strip()
    if not output:
        raise RuntimeError("Source2Viewer-CLI returned an empty version")
    first_line = output.splitlines()[0]
    prefix = "Version:"
    return first_line[len(prefix) :].strip() if first_line.startswith(prefix) else first_line


def extract_world_physics(
    executable: Path, cs2_dir: Path, map_name: str, *, materials: bool = True
) -> bytes:
    if _MAP_NAME.fullmatch(map_name) is None:
        raise ValueError("map name may contain only letters, numbers, and underscores")
    map_vpk = cs2_dir / "game" / "csgo" / "maps" / f"{map_name}.vpk"
    if not map_vpk.is_file():
        raise FileNotFoundError(f"map VPK was not found: {map_vpk}")
    return _extract_glb(
        executable,
        map_vpk,
        f"maps/{map_name}/world_physics.vmdl_c",
        "world_physics_physics.glb",
        materials=materials,
    )


def extract_entity_physics(executable: Path, cs2_dir: Path, map_name: str) -> bytes:
    if _MAP_NAME.fullmatch(map_name) is None:
        raise ValueError("map name may contain only letters, numbers, and underscores")
    map_vpk = cs2_dir / "game" / "csgo" / "maps" / f"{map_name}.vpk"
    if not map_vpk.is_file():
        raise FileNotFoundError(f"map VPK was not found: {map_vpk}")
    return _extract_glb(
        executable,
        map_vpk,
        f"maps/{map_name}/entities/default_ents.vents_c",
        "default_ents_physics.glb",
        materials=False,
    )


def extract_entity_data(executable: Path, cs2_dir: Path, map_name: str) -> str:
    if _MAP_NAME.fullmatch(map_name) is None:
        raise ValueError("map name may contain only letters, numbers, and underscores")
    map_vpk = cs2_dir / "game" / "csgo" / "maps" / f"{map_name}.vpk"
    if not map_vpk.is_file():
        raise FileNotFoundError(f"map VPK was not found: {map_vpk}")
    completed = subprocess.run(
        [
            str(executable),
            "-i",
            str(map_vpk),
            "-f",
            f"maps/{map_name}/entities/default_ents.vents_c",
            "--block",
            "DATA",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    if "m_entityKeyValues" not in completed.stdout:
        raise RuntimeError("Source2Viewer entity DATA output is missing m_entityKeyValues")
    return completed.stdout


def _extract_glb(
    executable: Path,
    vpk: Path,
    internal_path: str,
    preferred_output: str,
    *,
    materials: bool,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="csdemo-mapextractor-") as temporary:
        output_dir = Path(temporary)
        command = [
                str(executable),
                "-i",
                str(vpk),
                "-f",
                internal_path,
                "-o",
                str(output_dir),
                "-d",
                "--gltf_export_format",
                "glb",
            ]
        if materials:
            command.append("--gltf_export_materials")
        subprocess.run(
            command,
            check=True,
            timeout=600,
        )
        outputs = list(output_dir.rglob("*.glb"))
        physics_outputs = [path for path in outputs if path.name == preferred_output]
        if len(physics_outputs) == 1:
            return physics_outputs[0].read_bytes()
        if len(outputs) == 1:
            return outputs[0].read_bytes()
        raise RuntimeError(
            "expected one Source2Viewer physics GLB output, "
            f"found {len(physics_outputs)} among {len(outputs)} GLB files"
        )


def cs2_client_version(cs2_dir: Path) -> str | None:
    steam_inf = cs2_dir / "game" / "csgo" / "steam.inf"
    if not steam_inf.is_file():
        return None
    for line in steam_inf.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "ClientVersion":
            return value.strip() or None
    return None