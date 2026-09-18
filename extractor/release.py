"""Build versioned, browser-ready map asset releases."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable

from .entity_data import parse_states
from .layered import CollisionLayer, dumps as dumps_layered, entity_layers, from_vrf
from . import source2viewer


_EXCLUDED_MAP_MARKERS = ("_preview", "_vanity", "lobby_", "graphics_")
_SAFE_VERSION = re.compile(r"[A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class ReleaseResult:
    root_index: dict[str, object]
    version_manifest: dict[str, object]
    version_directory: Path


def discover_maps(cs2_dir: Path) -> tuple[str, ...]:
    maps_dir = cs2_dir / "game" / "csgo" / "maps"
    if not maps_dir.is_dir():
        raise FileNotFoundError(f"CS2 maps directory was not found: {maps_dir}")
    return tuple(
        sorted(
            path.stem
            for path in maps_dir.glob("*.vpk")
            if not any(marker in path.stem.lower() for marker in _EXCLUDED_MAP_MARKERS)
        )
    )


def steam_versions(cs2_dir: Path) -> dict[str, str | None]:
    steam_inf = cs2_dir / "game" / "csgo" / "steam.inf"
    if not steam_inf.is_file():
        raise FileNotFoundError(f"steam.inf was not found: {steam_inf}")
    values: dict[str, str] = {}
    for line in steam_inf.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return {
        "client_version": values.get("ClientVersion"),
        "server_version": values.get("ServerVersion"),
        "patch_version": values.get("PatchVersion"),
        "source_revision": values.get("SourceRevision"),
        "version_date": values.get("VersionDate"),
        "version_time": values.get("VersionTime"),
    }


def build_release(
    cs2_dir: Path,
    executable: Path,
    output_dir: Path,
    *,
    maps: Iterable[str] | None = None,
    steam_build_id: str | None = None,
    all_layers: bool = False,
    force: bool = False,
) -> ReleaseResult:
    versions = steam_versions(cs2_dir)
    client_version = versions["client_version"]
    if client_version is None or _SAFE_VERSION.fullmatch(client_version) is None:
        raise ValueError(f"invalid CS2 ClientVersion: {client_version!r}")
    map_names = tuple(dict.fromkeys(maps if maps is not None else discover_maps(cs2_dir)))
    if not map_names:
        raise ValueError("no maps were selected for the release")

    release_key = (
        f"{client_version}-{steam_build_id}" if steam_build_id else client_version
    )
    if _SAFE_VERSION.fullmatch(release_key) is None:
        raise ValueError(f"invalid release key: {release_key!r}")
    version_relative = Path("v1") / release_key
    version_directory = output_dir / version_relative
    if version_directory.exists() and not force:
        raise FileExistsError(
            f"release {version_directory} already exists; pass --force to replace it"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".csdemo-release-", dir=output_dir) as temporary:
        staged_version = Path(temporary) / client_version
        staged_version.mkdir()
        available_resources = source2viewer.list_map_asset_resources(
            executable, cs2_dir
        )
        map_records = [
            _build_map(
                cs2_dir,
                executable,
                staged_version,
                map_name,
                available_resources=available_resources,
                all_layers=all_layers,
            )
            for map_name in map_names
        ]
        selection_profile = (
            "all-physics-layered-v1"
            if all_layers
            else "default-plus-enabled-func-brush-v1"
        )
        version_manifest: dict[str, object] = {
            "format": "csdemo-map-release-v1",
            "compression": "EXT_meshopt_compression",
            "coordinate_system": "source2-hammer-z-up",
            "selection_profile": selection_profile,
            "source2viewer_version": source2viewer.version(executable),
            "steam_build_id": steam_build_id,
            **versions,
            "maps": map_records,
        }
        _write_json(staged_version / "index.json", version_manifest)

        version_directory.parent.mkdir(parents=True, exist_ok=True)
        if version_directory.exists():
            shutil.rmtree(version_directory)
        staged_version.replace(version_directory)

    root_index: dict[str, object] = {
        "format": "csdemo-map-host-v1",
        "current": release_key,
        "client_version": client_version,
        "steam_build_id": steam_build_id,
        "manifest": (version_relative / "index.json").as_posix(),
    }
    _write_json_atomic(output_dir / "index.json", root_index)
    return ReleaseResult(root_index, version_manifest, version_directory)


def _build_map(
    cs2_dir: Path,
    executable: Path,
    version_directory: Path,
    map_name: str,
    *,
    available_resources: frozenset[str],
    all_layers: bool,
) -> dict[str, object]:
    source_glb = source2viewer.extract_world_physics(
        executable, cs2_dir, map_name, materials=False
    )
    layers = list(from_vrf(source_glb))
    entity_glb = source2viewer.extract_entity_physics(executable, cs2_dir, map_name)
    entity_data = source2viewer.extract_entity_data(executable, cs2_dir, map_name)
    layers.extend(
        entity_layers(
            entity_glb,
            "func_brush",
            states=parse_states(entity_data, "func_brush"),
        )
    )
    selected_layers = _select_layers(tuple(layers), all_layers=all_layers)
    selection_profile = (
        "all-physics-layered-v1"
        if all_layers
        else "default-plus-enabled-func-brush-v1"
    )
    collision = dumps_layered(
        selected_layers,
        map_name=map_name,
        selection_profile=selection_profile,
    )
    assets = source2viewer.extract_map_assets(
        executable,
        cs2_dir,
        map_name,
        available_resources=available_resources,
    )
    _validate_assets(map_name, assets)

    map_directory = version_directory / "maps" / map_name
    files: dict[str, dict[str, object]] = {}
    files["collision"] = _write_file(
        version_directory, map_directory / "collision.glb", collision, "model/gltf-binary"
    )
    if assets.overview is not None:
        files["overview"] = _write_file(
            version_directory,
            map_directory / "overview.txt",
            assets.overview,
            "text/plain; charset=utf-8",
        )
    if assets.logo is not None:
        files["logo"] = _write_file(
            version_directory,
            map_directory / "logo.svg",
            assets.logo,
            "image/svg+xml",
        )
    radars = []
    for section, data in assets.radars.items():
        filename = "radar.png" if section == "default" else f"radar_{section}.png"
        radars.append(
            {
                "section": section,
                **_write_file(
                    version_directory,
                    map_directory / filename,
                    data,
                    "image/png",
                ),
            }
        )
    return {
        "name": map_name,
        "files": files,
        "radars": radars,
        "missing_assets": list(assets.missing),
        "vertex_count": sum(len(layer.geometry.positions) for layer in selected_layers),
        "triangle_count": sum(len(layer.geometry.triangles) for layer in selected_layers),
        "layers": [_layer_record(layer) for layer in selected_layers],
    }


def _select_layers(
    layers: tuple[CollisionLayer, ...], *, all_layers: bool
) -> tuple[CollisionLayer, ...]:
    if all_layers:
        return layers
    return tuple(
        layer for layer in layers if layer.name in {"default", "entity:func_brush"}
    )


def _layer_record(layer: CollisionLayer) -> dict[str, object]:
    return {
        "name": layer.name,
        "tags": list(layer.tags),
        "surface_properties": list(layer.surface_properties),
        "vertex_count": len(layer.geometry.positions),
        "triangle_count": len(layer.geometry.triangles),
    }


def _validate_assets(map_name: str, assets: source2viewer.MapAssets) -> None:
    if assets.overview is not None and not assets.overview.strip():
        raise ValueError(f"{map_name} overview is empty")
    if assets.logo is not None and not assets.logo.lstrip().startswith((b"<svg", b"<?xml")):
        raise ValueError(f"{map_name} logo is not SVG")
    for section, radar in assets.radars.items():
        if not radar.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"{map_name} radar section {section!r} is not PNG")


def _write_file(
    version_directory: Path,
    path: Path,
    data: bytes,
    content_type: str,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": path.relative_to(version_directory).as_posix(),
        "content_type": content_type,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, value)
    temporary.replace(path)
