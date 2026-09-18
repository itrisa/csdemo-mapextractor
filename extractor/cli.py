"""Command-line interface for artifact conversion and validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Sequence

from . import tri
from . import glb
from .comparison import compare
from .geometry import CollisionGeometry
from .entity_data import parse_states
from .layered import CollisionLayer, dumps as dumps_layered, entity_layers, from_vrf
from .manifest import create as create_manifest
from .profiles import ALL_PHYSICS, AWPY_DEFAULT, VISUAL_OCCLUDERS
from . import r2
from . import release
from . import source2viewer
from . import steam


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="csdemo-mapextractor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--cs2-dir", type=Path)
    doctor_parser.add_argument("--source2viewer", type=Path)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--cs2-dir", type=Path, required=True)
    extract_parser.add_argument("--map", required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument(
        "--profile",
        choices=(AWPY_DEFAULT, VISUAL_OCCLUDERS, ALL_PHYSICS),
        default=VISUAL_OCCLUDERS,
    )
    extract_parser.add_argument("--source2viewer", type=Path)
    extract_parser.add_argument("--meshopt", action="store_true")
    extract_parser.add_argument("--force", action="store_true")

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("input", type=Path)
    convert_parser.add_argument("output", type=Path)
    convert_parser.add_argument("--map")
    convert_parser.add_argument("--profile", default=VISUAL_OCCLUDERS)
    convert_parser.add_argument("--meshopt", action="store_true")
    convert_parser.add_argument("--force", action="store_true")

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("artifact", type=Path)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("artifact", type=Path)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("first", type=Path)
    compare_parser.add_argument("second", type=Path)

    cache_parser = subparsers.add_parser("cache-maps")
    cache_parser.add_argument("--cs2-dir", type=Path, required=True)
    cache_parser.add_argument("--maps", nargs="+", required=True)
    cache_parser.add_argument("--output-dir", type=Path, required=True)
    cache_parser.add_argument("--source2viewer", type=Path)
    cache_parser.add_argument("--all-layers", action="store_true")
    cache_parser.add_argument("--force", action="store_true")

    assets_parser = subparsers.add_parser("extract-assets")
    assets_parser.add_argument("--cs2-dir", type=Path, required=True)
    assets_parser.add_argument("--maps", nargs="+", required=True)
    assets_parser.add_argument("--output-dir", type=Path, default=Path("out"))
    assets_parser.add_argument("--source2viewer", type=Path)
    assets_parser.add_argument("--force", action="store_true")

    discover_parser = subparsers.add_parser("discover-maps")
    discover_parser.add_argument("--cs2-dir", type=Path, required=True)

    steam_parser = subparsers.add_parser("steam-build")
    steam_parser.add_argument("--steamcmd")
    steam_parser.add_argument("--key", action="store_true")

    release_parser = subparsers.add_parser("build-release")
    release_parser.add_argument("--cs2-dir", type=Path, required=True)
    release_parser.add_argument("--maps", nargs="+")
    release_parser.add_argument("--output-dir", type=Path, default=Path("out") / "release")
    release_parser.add_argument("--source2viewer", type=Path)
    release_parser.add_argument("--steam-build-id")
    release_parser.add_argument("--all-layers", action="store_true")
    release_parser.add_argument("--force", action="store_true")

    publish_parser = subparsers.add_parser("publish-r2")
    publish_parser.add_argument("--release-dir", type=Path, default=Path("out") / "release")
    publish_parser.add_argument("--bucket", required=True)
    publish_parser.add_argument("--endpoint-url", required=True)
    publish_parser.add_argument("--aws", default="aws")
    publish_parser.add_argument("--profile")
    publish_parser.add_argument("--force", action="store_true")

    current_parser = subparsers.add_parser("r2-current-build")
    current_parser.add_argument("--bucket", required=True)
    current_parser.add_argument("--endpoint-url", required=True)
    current_parser.add_argument("--aws", default="aws")
    current_parser.add_argument("--profile")

    arguments = parser.parse_args(argv)
    if arguments.command == "doctor":
        return _doctor(arguments)
    if arguments.command == "extract":
        return _extract(arguments)
    if arguments.command == "convert":
        return _convert(arguments)
    if arguments.command == "compare":
        print(json.dumps(compare(_read_geometry(arguments.first), _read_geometry(arguments.second)), indent=2))
        return 0
    if arguments.command == "cache-maps":
        return _cache_maps(arguments)
    if arguments.command == "extract-assets":
        return _extract_assets(arguments)
    if arguments.command == "discover-maps":
        print("\n".join(release.discover_maps(arguments.cs2_dir)))
        return 0
    if arguments.command == "steam-build":
        build_id, updated_at = steam.public_build(arguments.steamcmd)
        print(build_id if arguments.key else json.dumps(
            {"buildid": build_id, "timeupdated": updated_at}, indent=2
        ))
        return 0
    if arguments.command == "build-release":
        return _build_release(arguments)
    if arguments.command == "publish-r2":
        result = r2.publish(
            arguments.release_dir,
            bucket=arguments.bucket,
            endpoint_url=arguments.endpoint_url,
            aws=arguments.aws,
            profile=arguments.profile,
            force=arguments.force,
        )
        print(json.dumps(result, indent=2))
        return 0
    if arguments.command == "r2-current-build":
        build_id = r2.current_build(
            bucket=arguments.bucket,
            endpoint_url=arguments.endpoint_url,
            aws=arguments.aws,
            profile=arguments.profile,
        )
        print(build_id or "")
        return 0
    geometry = _read_geometry(arguments.artifact)
    details = _describe(geometry, arguments.artifact.suffix.lower())
    details["valid"] = True
    print(json.dumps(details, indent=2))
    return 0


def _doctor(arguments: argparse.Namespace) -> int:
    details: dict[str, object] = {}
    try:
        executable = source2viewer.resolve(arguments.source2viewer)
        details["source2viewer"] = {"path": str(executable), "version": source2viewer.version(executable)}
    except (FileNotFoundError, RuntimeError, subprocess.SubprocessError) as error:
        details["source2viewer"] = {"error": str(error)}
    if arguments.cs2_dir is not None:
        csgo_dir = arguments.cs2_dir / "game" / "csgo"
        details["cs2"] = {
            "path": str(arguments.cs2_dir.resolve()),
            "valid": csgo_dir.is_dir(),
            "client_version": source2viewer.cs2_client_version(arguments.cs2_dir),
        }
    print(json.dumps(details, indent=2))
    return 0 if all("error" not in value and value.get("valid", True) for value in details.values()) else 1


def _extract(arguments: argparse.Namespace) -> int:
    _ensure_writable(arguments.output, arguments.force)
    executable = source2viewer.resolve(arguments.source2viewer)
    source_glb = source2viewer.extract_world_physics(executable, arguments.cs2_dir, arguments.map)
    geometry = glb.loads(source_glb, selection_profile=arguments.profile)
    artifact, output_format = _encode_geometry(geometry, arguments.output, arguments.profile, arguments.meshopt)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(artifact)
    manifest = create_manifest(
        geometry,
        artifact,
        map_name=arguments.map,
        selection_profile=arguments.profile,
        output_format=output_format,
        compression="EXT_meshopt_compression" if arguments.meshopt else None,
    )
    manifest["source2viewer_version"] = source2viewer.version(executable)
    manifest["cs2_client_version"] = source2viewer.cs2_client_version(arguments.cs2_dir)
    _write_manifest(arguments.output, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def _convert(arguments: argparse.Namespace) -> int:
    _ensure_writable(arguments.output, arguments.force)
    geometry = _read_geometry(arguments.input)
    artifact, output_format = _encode_geometry(geometry, arguments.output, arguments.profile, arguments.meshopt)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(artifact)
    manifest = create_manifest(
        geometry,
        artifact,
        map_name=arguments.map,
        selection_profile=arguments.profile,
        output_format=output_format,
        compression="EXT_meshopt_compression" if arguments.meshopt else None,
    )
    _write_manifest(arguments.output, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def _cache_maps(arguments: argparse.Namespace) -> int:
    executable = source2viewer.resolve(arguments.source2viewer)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    maps = []
    for map_name in arguments.maps:
        output = arguments.output_dir / f"{map_name}.glb"
        if output.exists() and not arguments.force:
            raise SystemExit(f"refusing to overwrite {output}; pass --force")
        source_glb = source2viewer.extract_world_physics(
            executable, arguments.cs2_dir, map_name, materials=False
        )
        layers = list(from_vrf(source_glb))
        entity_glb = source2viewer.extract_entity_physics(executable, arguments.cs2_dir, map_name)
        entity_data = source2viewer.extract_entity_data(executable, arguments.cs2_dir, map_name)
        layers.extend(
            entity_layers(
                entity_glb,
                "func_brush",
                states=parse_states(entity_data, "func_brush"),
            )
        )
        layers_tuple = _select_cache_layers(tuple(layers), all_layers=arguments.all_layers)
        selection_profile = (
            "all-physics-layered-v1"
            if arguments.all_layers
            else "default-plus-enabled-func-brush-v1"
        )
        artifact = dumps_layered(
            layers_tuple,
            map_name=map_name,
            selection_profile=selection_profile,
        )
        temporary = output.with_suffix(".glb.tmp")
        temporary.write_bytes(artifact)
        temporary.replace(output)
        maps.append(
            {
                "name": map_name,
                "file": output.name,
                "bytes": len(artifact),
                "sha256": hashlib.sha256(artifact).hexdigest(),
                "vertex_count": sum(len(layer.geometry.positions) for layer in layers_tuple),
                "triangle_count": sum(len(layer.geometry.triangles) for layer in layers_tuple),
                "layers": [
                    {
                        "name": layer.name,
                        "tags": list(layer.tags),
                        "surface_properties": list(layer.surface_properties),
                        "vertex_count": len(layer.geometry.positions),
                        "triangle_count": len(layer.geometry.triangles),
                    }
                    for layer in layers_tuple
                ],
            }
        )
    index = {
        "format": "csdemo-layered-maps-v1",
        "compression": "EXT_meshopt_compression",
        "coordinate_system": "source2-hammer-z-up",
        "selection_profile": selection_profile,
        "source2viewer_version": source2viewer.version(executable),
        "cs2_client_version": source2viewer.cs2_client_version(arguments.cs2_dir),
        "maps": maps,
    }
    index_path = arguments.output_dir / "index.json"
    temporary_index = index_path.with_suffix(".json.tmp")
    temporary_index.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    temporary_index.replace(index_path)
    print(json.dumps(index, indent=2))
    return 0


def _extract_assets(arguments: argparse.Namespace) -> int:
    executable = source2viewer.resolve(arguments.source2viewer)
    available_resources = source2viewer.list_map_asset_resources(
        executable, arguments.cs2_dir
    )
    outputs_by_map: list[tuple[str, Path, dict[str, Path], tuple[str, ...]]] = []
    for map_name in arguments.maps:
        map_dir = arguments.output_dir / map_name
        paths = {
            "overview": map_dir / "overview.txt",
            "logo": map_dir / "logo.svg",
        }
        existing = [
            path
            for path in (*paths.values(), *map_dir.glob("radar*.png"))
            if path.exists()
        ]
        if existing and not arguments.force:
            raise SystemExit(
                f"refusing to overwrite {existing[0]}; pass --force"
            )
        assets = source2viewer.extract_map_assets(
            executable,
            arguments.cs2_dir,
            map_name,
            available_resources=available_resources,
        )
        for section in assets.radars:
            filename = "radar.png" if section == "default" else f"radar_{section}.png"
            paths[f"radar:{section}"] = map_dir / filename
        if arguments.force:
            expected = set(paths.values())
            for stale in map_dir.glob("radar*.png"):
                if stale not in expected:
                    stale.unlink()
        payloads = {
            **{
                paths[f"radar:{section}"]: data
                for section, data in assets.radars.items()
            },
        }
        if assets.overview is not None:
            payloads[paths["overview"]] = assets.overview
        else:
            if arguments.force and paths["overview"].exists():
                paths["overview"].unlink()
            del paths["overview"]
        if assets.logo is not None:
            payloads[paths["logo"]] = assets.logo
        else:
            if arguments.force and paths["logo"].exists():
                paths["logo"].unlink()
            del paths["logo"]
        map_dir.mkdir(parents=True, exist_ok=True)
        for output, data in payloads.items():
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(output)
        outputs_by_map.append((map_name, map_dir, paths, assets.missing))

    result = {
        "format": "cs2-map-assets-v1",
        "source2viewer_version": source2viewer.version(executable),
        "cs2_client_version": source2viewer.cs2_client_version(arguments.cs2_dir),
        "maps": [
            {
                "name": map_name,
                "directory": str(map_dir),
                "missing_assets": list(missing),
                "files": {
                    name: {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for name, path in paths.items()
                },
            }
            for map_name, map_dir, paths, missing in outputs_by_map
        ],
    }
    print(json.dumps(result, indent=2))
    return 0


def _build_release(arguments: argparse.Namespace) -> int:
    executable = source2viewer.resolve(arguments.source2viewer)
    result = release.build_release(
        arguments.cs2_dir,
        executable,
        arguments.output_dir,
        maps=arguments.maps,
        steam_build_id=arguments.steam_build_id,
        all_layers=arguments.all_layers,
        force=arguments.force,
    )
    print(json.dumps({
        "directory": str(result.version_directory),
        **result.root_index,
    }, indent=2))
    return 0


def _select_cache_layers(
    layers: tuple[CollisionLayer, ...], *, all_layers: bool
) -> tuple[CollisionLayer, ...]:
    if all_layers:
        return layers
    return tuple(
        layer for layer in layers if layer.name in {"default", "entity:func_brush"}
    )


def _read_geometry(path: Path) -> CollisionGeometry:
    data = path.read_bytes()
    if path.suffix.lower() == ".tri":
        return tri.loads(data)
    if path.suffix.lower() == ".glb":
        return glb.loads(data)
    raise SystemExit("artifact extension must be .tri or .glb")


def _describe(geometry: CollisionGeometry, suffix: str) -> dict[str, object]:
    return {
        "format": "glTF 2.0" if suffix == ".glb" else "tri-v1",
        "vertex_count": len(geometry.positions),
        "triangle_count": len(geometry.triangles),
    }


def _encode_geometry(
    geometry: CollisionGeometry, output: Path, profile: str, meshopt: bool = False
) -> tuple[bytes, str]:
    if output.suffix.lower() == ".tri":
        if meshopt:
            raise SystemExit("--meshopt is only valid for GLB output")
        return tri.dumps(geometry), "tri-v1"
    if output.suffix.lower() == ".glb":
        return glb.dumps(geometry, selection_profile=profile, meshopt=meshopt), "glTF 2.0"
    raise SystemExit("output extension must be .tri or .glb")


def _ensure_writable(output: Path, force: bool) -> None:
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    if not force and (output.exists() or manifest.exists()):
        raise SystemExit(f"refusing to overwrite {output} or its manifest; pass --force")


def _write_manifest(output: Path, manifest: dict[str, object]) -> None:
    path = output.with_suffix(output.suffix + ".manifest.json")
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())