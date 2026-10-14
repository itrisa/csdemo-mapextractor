"""Build versioned, browser-ready map asset releases."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Iterable

from .entity_data import parse_states
from .layered import CollisionLayer, dumps as dumps_layered, entity_layers, from_vrf
from . import source2viewer


_MAP_PREFIX = "de_"
_EXCLUDED_MAP_MARKERS = ("_preview", "_vanity", "lobby_", "graphics_")
_SAFE_VERSION = re.compile(r"[A-Za-z0-9._-]+")
# published with their collision GLB only: colouring them takes too long / too much memory for CI runners
COLOR_EXCLUDED_MAPS = frozenset({"de_boulder"})
OBJECTS_DIRECTORY = "objects"


@dataclass(frozen=True, slots=True)
class ReleaseResult:
    root_index: dict[str, object]
    version_manifest: dict[str, object]
    manifest_path: Path


def discover_maps(cs2_dir: Path) -> tuple[str, ...]:
    maps_dir = cs2_dir / "game" / "csgo" / "maps"
    if not maps_dir.is_dir():
        raise FileNotFoundError(f"CS2 maps directory was not found: {maps_dir}")
    return tuple(
        sorted(
            path.stem
            for path in maps_dir.glob("*.vpk")
            if path.stem.lower().startswith(_MAP_PREFIX)
            and not any(marker in path.stem.lower() for marker in _EXCLUDED_MAP_MARKERS)
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
    color_bundles: Path | None = None,
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
    release_directory = output_dir / "v1"
    existing = sorted(release_directory.glob(f"release-{release_key}-*.json"))
    if existing and not force:
        raise FileExistsError(
            f"release {existing[0]} already exists; pass --force to replace it"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".csdemo-release-", dir=output_dir) as temporary:
        staged_objects = Path(temporary) / OBJECTS_DIRECTORY
        staged_objects.mkdir()
        available_resources = source2viewer.list_map_asset_resources(
            executable, cs2_dir
        )
        map_records = [
            _build_map(
                cs2_dir,
                executable,
                staged_objects,
                map_name,
                available_resources=available_resources,
                all_layers=all_layers,
                color_bundles=color_bundles,
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
            "release": release_key,
            "compression": "EXT_meshopt_compression",
            "coordinate_system": "source2-hammer-z-up",
            "selection_profile": selection_profile,
            "source2viewer_version": source2viewer.version(executable),
            "steam_build_id": steam_build_id,
            **versions,
            "maps": map_records,
        }
        objects_directory = release_directory / OBJECTS_DIRECTORY
        objects_directory.mkdir(parents=True, exist_ok=True)
        for staged in staged_objects.iterdir():
            target = objects_directory / staged.name
            if not target.exists():
                staged.replace(target)
        return write_manifest(output_dir, version_manifest)


def load_release(output_dir: Path) -> tuple[dict[str, object], Path, dict[str, object]]:
    """Returns (root index, manifest path, manifest) of a built release directory."""
    root_index = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
    manifest_path = output_dir / Path(str(root_index["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return root_index, manifest_path, manifest


def write_manifest(output_dir: Path, version_manifest: dict[str, object]) -> ReleaseResult:
    """Write the manifest under its content-derived name and point the root index at it."""
    release_key = str(version_manifest["release"])
    release_directory = output_dir / "v1"
    manifest_bytes = _json_bytes(version_manifest)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()[:12]
    manifest_path = release_directory / f"release-{release_key}-{manifest_digest}.json"
    for previous in release_directory.glob(f"release-{release_key}-*.json"):
        if previous != manifest_path:
            previous.unlink()
    manifest_path.write_bytes(manifest_bytes)

    root_index: dict[str, object] = {
        "format": "csdemo-map-host-v1",
        "current": release_key,
        "client_version": version_manifest.get("client_version"),
        "steam_build_id": version_manifest.get("steam_build_id"),
        "manifest": manifest_path.relative_to(output_dir).as_posix(),
    }
    _write_json_atomic(output_dir / "index.json", root_index)
    return ReleaseResult(root_index, version_manifest, manifest_path)


def write_object(objects_directory: Path, suffix: str, data: bytes, content_type: str) -> dict[str, object]:
    objects_directory.mkdir(parents=True, exist_ok=True)
    return _write_object(objects_directory, suffix, data, content_type)


def _build_map(
    cs2_dir: Path,
    executable: Path,
    objects_directory: Path,
    map_name: str,
    *,
    available_resources: frozenset[str],
    all_layers: bool,
    color_bundles: Path | None = None,
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

    files: dict[str, dict[str, object]] = {}
    files["collision"] = _write_object(
        objects_directory, ".glb", collision, "model/gltf-binary"
    )
    if assets.overview is not None:
        files["overview"] = _write_object(
            objects_directory,
            ".txt",
            assets.overview,
            "text/plain; charset=utf-8",
        )
    if assets.logo is not None:
        files["logo"] = _write_object(
            objects_directory,
            ".svg",
            assets.logo,
            "image/svg+xml",
        )
    radars = []
    for section, data in assets.radars.items():
        radars.append(
            {
                "section": section,
                **_write_object(
                    objects_directory,
                    ".png",
                    data,
                    "image/png",
                ),
            }
        )
    record: dict[str, object] = {
        "name": map_name,
        "files": files,
        "radars": radars,
        "missing_assets": list(assets.missing),
        "vertex_count": sum(len(layer.geometry.positions) for layer in selected_layers),
        "triangle_count": sum(len(layer.geometry.triangles) for layer in selected_layers),
        "layers": [_layer_record(layer) for layer in selected_layers],
    }
    if color_bundles is not None and map_name not in COLOR_EXCLUDED_MAPS:
        from .color import bundle as _  # noqa: F401 - missing dependencies must fail the build, not skip colour

        try:
            record["color_source_sha256"] = _write_color_bundle(
                cs2_dir, executable, color_bundles, map_name, collision
            )
        except Exception as error:  # noqa: BLE001 - colouring is optional; the map still ships uncoloured
            shutil.rmtree(color_bundles / map_name, ignore_errors=True)
            print(f"warning: no colour bundle for {map_name}: {error}", file=sys.stderr)
    return record


def _write_color_bundle(
    cs2_dir: Path, executable: Path, color_bundles: Path, map_name: str, collision: bytes
) -> str:
    from .color import bundle

    bundle_dir = color_bundles / map_name
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    with tempfile.TemporaryDirectory(prefix=f".render-{map_name}-") as temporary:
        render_dir = source2viewer.export_render_world(
            executable, cs2_dir, map_name, Path(temporary)
        )
        return bundle.create(bundle_dir, map_name, collision, render_dir)


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


def _write_object(
    objects_directory: Path,
    suffix: str,
    data: bytes,
    content_type: str,
) -> dict[str, object]:
    digest = hashlib.sha256(data).hexdigest()
    name = f"{digest}{suffix}"
    path = objects_directory / name
    if not path.exists():
        path.write_bytes(data)
    return {
        "path": f"{OBJECTS_DIRECTORY}/{name}",
        "content_type": content_type,
        "bytes": len(data),
        "sha256": digest,
    }


def _json_bytes(value: dict[str, object]) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_bytes(_json_bytes(value))


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, value)
    temporary.replace(path)
