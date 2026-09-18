"""Stable adapter around ValveResourceFormat's Source2Viewer-CLI."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass


_MAP_NAME = re.compile(r"[A-Za-z0-9_]+")
_QUOTED_KEY = re.compile(r'"([^"]+)"\s*\{')


@dataclass(frozen=True, slots=True)
class MapAssets:
    overview: bytes | None
    logo: bytes | None
    radars: dict[str, bytes]
    missing: tuple[str, ...] = ()


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


def list_map_asset_resources(executable: Path, cs2_dir: Path) -> frozenset[str]:
    vpk = cs2_dir / "game" / "csgo" / "pak01_dir.vpk"
    if not vpk.is_file():
        raise FileNotFoundError(f"CS2 VPK was not found: {vpk}")
    completed = subprocess.run(
        [
            str(executable),
            "-i",
            str(vpk),
            "-l",
            "-f",
            ",".join(
                (
                    "resource/overviews/",
                    "panorama/images/overheadmaps/",
                    "panorama/images/map_icons/",
                )
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return frozenset(
        line.partition(" CRC:")[0].strip().replace("\\", "/")
        for line in completed.stdout.splitlines()
        if " CRC:" in line
    )


def extract_map_assets(
    executable: Path,
    cs2_dir: Path,
    map_name: str,
    *,
    available_resources: frozenset[str] | None = None,
) -> MapAssets:
    _validate_map_name(map_name)
    csgo_dir = cs2_dir / "game" / "csgo"
    vpk = csgo_dir / "pak01_dir.vpk"
    if not vpk.is_file():
        raise FileNotFoundError(f"CS2 VPK was not found: {vpk}")
    available = (
        available_resources
        if available_resources is not None
        else list_map_asset_resources(executable, cs2_dir)
    )
    missing = []

    overview_path = f"resource/overviews/{map_name}.txt"
    if overview_path in available:
        overview = _extract_resource(
            executable,
            vpk,
            overview_path,
            Path(overview_path),
            decompile=False,
        )
        sections = _vertical_sections(overview.decode("utf-8", errors="replace"))
    else:
        overview = None
        sections = ()
        missing.append("overview")
    radar_names = ("default", *(section for section in sections if section != "default"))
    radars = {}
    for section in radar_names:
        suffix = "" if section == "default" else f"_{section}"
        internal_path = (
            f"panorama/images/overheadmaps/{map_name}{suffix}_radar_psd.vtex_c"
        )
        if internal_path in available:
            radars[section] = _extract_resource(
                executable,
                vpk,
                internal_path,
                Path(internal_path.removesuffix(".vtex_c") + ".png"),
                decompile=True,
            )
        else:
            missing.append(f"radar:{section}")

    logo_path = f"panorama/images/map_icons/map_icon_{map_name}.vsvg_c"
    logo = (
        _extract_resource(
            executable,
            vpk,
            logo_path,
            Path(logo_path.removesuffix(".vsvg_c") + ".svg"),
            decompile=True,
        )
        if logo_path in available
        else None
    )
    if logo is None:
        missing.append("logo")
    return MapAssets(
        overview=overview,
        logo=logo,
        radars=radars,
        missing=tuple(missing),
    )


def _vertical_sections(overview: str) -> tuple[str, ...]:
    content = re.sub(r"//.*$", "", overview, flags=re.MULTILINE)
    marker = re.search(r'"verticalsections"\s*\{', content, flags=re.IGNORECASE)
    if marker is None:
        return ()
    start = content.find("{", marker.start())
    depth = 0
    sections = []
    position = start
    while position < len(content):
        character = content[position]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                break
        elif character == '"' and depth == 1:
            match = _QUOTED_KEY.match(content, position)
            if match is not None:
                section = match.group(1).lower()
                if _MAP_NAME.fullmatch(section) is None:
                    raise RuntimeError(
                        f"invalid vertical radar section name: {section}"
                    )
                sections.append(section)
                position = match.end() - 1
                continue
        position += 1
    return tuple(sections)


def _extract_resource(
    executable: Path,
    vpk: Path,
    internal_path: str,
    output_relative: Path,
    *,
    decompile: bool,
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
        ]
        if decompile:
            command.append("-d")
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = output_dir / output_relative
        if not output.is_file():
            raise RuntimeError(
                f"Source2Viewer did not produce {output_relative} from {internal_path}"
            )
        return output.read_bytes()


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


def _validate_map_name(map_name: str) -> None:
    if _MAP_NAME.fullmatch(map_name) is None:
        raise ValueError("map name may contain only letters, numbers, and underscores")