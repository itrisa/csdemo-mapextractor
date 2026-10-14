"""Plan and attach per-vertex coloured collision GLBs to a built release.

Colouring is expensive, so a map is only recoloured when its colour bundle digest (recorded per map as
``color_source_sha256`` by ``build-release --color-bundles``) differs from the one the currently published
release was coloured from; otherwise the published coloured GLB is reused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from .release import COLOR_EXCLUDED_MAPS, OBJECTS_DIRECTORY, ReleaseResult, load_release, write_manifest, write_object

COLORED_CONTENT_TYPE = "model/gltf-binary"
_TIMEOUT = 60


def published_colored(index_url: str | None) -> tuple[str | None, dict[str, dict[str, Any]]]:
    """Returns (manifest URL, {map: colored file record}) of the published release; empty if unavailable."""
    if not index_url:
        return None, {}
    try:
        root_index = _get_json(index_url)
        manifest_url = urllib.parse.urljoin(index_url, str(root_index["manifest"]))
        manifest = _get_json(manifest_url)
    except (urllib.error.URLError, OSError, KeyError, ValueError) as error:
        print(f"No published release available ({error}); every map will be coloured.", file=sys.stderr)
        return None, {}
    records = {}
    for map_record in manifest.get("maps", []):
        colored = map_record.get("files", {}).get("colored")
        if colored and colored.get("source_sha256"):
            records[map_record["name"]] = colored
    return manifest_url, records


def plan(release_dir: Path, index_url: str | None) -> list[str]:
    """Maps of the release whose colour bundle has no published coloured GLB yet."""
    _, _, manifest = load_release(release_dir)
    _, published = published_colored(index_url)
    pending = []
    for map_record in manifest["maps"]:
        source = map_record.get("color_source_sha256")
        if not source:
            continue
        if published.get(map_record["name"], {}).get("source_sha256") != source:
            pending.append(map_record["name"])
    return pending


def attach(release_dir: Path, colored_dir: Path | None, index_url: str | None) -> tuple[ReleaseResult, dict[str, str]]:
    """Add ``files.colored`` to every map with a matching coloured GLB, built (colored_dir/<map>.glb plus
    <map>.json naming its source digest) or reused from the published release. Returns the rewritten release
    and how each map was handled."""
    _, _, manifest = load_release(release_dir)
    objects_directory = release_dir / "v1" / OBJECTS_DIRECTORY
    manifest_url, published = published_colored(index_url)
    outcome: dict[str, str] = {}
    for map_record in manifest["maps"]:
        name = map_record["name"]
        source = map_record.get("color_source_sha256")
        files = map_record["files"]
        files.pop("colored", None)
        if not source:
            outcome[name] = "excluded" if name in COLOR_EXCLUDED_MAPS else "no-bundle"
            continue
        data = _built(colored_dir, name, source)
        if data is not None:
            outcome[name] = "built"
        else:
            record = published.get(name)
            if record is not None and record.get("source_sha256") == source and manifest_url:
                data = _download(urllib.parse.urljoin(manifest_url, record["path"]), record)
                outcome[name] = "reused"
        if data is None:
            outcome[name] = "missing"
            continue
        files["colored"] = {
            **write_object(objects_directory, ".glb", data, COLORED_CONTENT_TYPE),
            "source_sha256": source,
        }
    return write_manifest(release_dir, manifest), outcome


def _built(colored_dir: Path | None, name: str, source: str) -> bytes | None:
    if colored_dir is None:
        return None
    glb, sidecar = colored_dir / f"{name}.glb", colored_dir / f"{name}.json"
    if not glb.is_file() or not sidecar.is_file():
        return None
    if json.loads(sidecar.read_text(encoding="utf-8")).get("source_sha256") != source:
        return None
    return glb.read_bytes()


def _download(url: str, record: dict[str, Any]) -> bytes | None:
    try:
        with urllib.request.urlopen(_request(url), timeout=_TIMEOUT) as response:
            data = response.read()
    except (urllib.error.URLError, OSError) as error:
        print(f"Could not download {url}: {error}", file=sys.stderr)
        return None
    if len(data) != record.get("bytes") or hashlib.sha256(data).hexdigest() != record.get("sha256"):
        print(f"Published object {url} does not match its manifest record", file=sys.stderr)
        return None
    return data


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(_request(url), timeout=_TIMEOUT) as response:
        return json.loads(response.read())


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url, headers={"Cache-Control": "no-cache", "User-Agent": "csdemo-mapextractor"}
    )
