"""Publish the CS2 render packages (maps, weapons, sounds) to R2, keeping one version of each.

Bucket layout:

    cs2/index.json                  the current packages; revalidated on every request
    cs2/maps/<map>/<digest>/...     one cs2render package per map
    cs2/weapons/<digest>/...        the cs2weapons package
    cs2/sounds/<digest>/...         the cs2sounds package

<digest> is derived from a package's files, so an unchanged package is not uploaded again and every
file under it can be cached as immutable. Packages are uploaded first, then the index, then every
other key under cs2/ is deleted, so the bucket holds a single version of each package.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Protocol, Sequence


PREFIX = "cs2/"
INDEX_KEY = PREFIX + "index.json"
FORMAT = "csdemo-cs2-assets-v1"
CONTENT_TYPES = {
    ".bin": "application/octet-stream",
    ".ctex": "application/octet-stream",
    ".glb": "model/gltf-binary",
    ".json": "application/json",
}
IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
# deleted versions must not be served from a cached index
INDEX_CACHE = "no-cache"
_MAP_NAME = re.compile(r"[a-z0-9_]+")
_DIGEST_LENGTH = 20


@dataclass(frozen=True)
class Package:
    kind: str  # "maps/<map>", "weapons" or "sounds"
    files: dict[str, tuple[Path, str]]  # relative path -> (file, sha256)
    digest: str

    @property
    def path(self) -> str:
        return f"{self.kind}/{self.digest}/"

    @property
    def bytes(self) -> int:
        return sum(path.stat().st_size for path, _ in self.files.values())

    def entry(self) -> dict[str, Any]:
        return {"path": self.path, "bytes": self.bytes, "files": len(self.files)}


def load_package(kind: str, root: Path, marker: str) -> Package:
    if not (root / marker).is_file():
        raise FileNotFoundError(f"{root} is not a {kind} package: {marker} is missing")
    files: dict[str, tuple[Path, str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() not in CONTENT_TYPES:
            raise ValueError(f"unsupported package file type: {path}")
        files[relative] = (path, hashlib.sha256(path.read_bytes()).hexdigest())
    combined = hashlib.sha256()
    for relative, (_, digest) in files.items():
        combined.update(f"{relative}\0{digest}\n".encode("utf-8"))
    return Package(kind, files, combined.hexdigest()[:_DIGEST_LENGTH])


def collect(
    maps_dir: Path | None, weapons_dir: Path | None, sounds_dir: Path | None
) -> tuple[dict[str, Package], Package | None, Package | None]:
    """Loads the packages a bake produced; a directory that does not exist counts as not rebuilt."""
    maps: dict[str, Package] = {}
    if maps_dir is not None and maps_dir.is_dir():
        for root in sorted(p for p in maps_dir.iterdir() if p.is_dir()):
            if not _MAP_NAME.fullmatch(root.name):
                raise ValueError(f"unexpected map package name: {root.name}")
            maps[root.name] = load_package(f"maps/{root.name}", root, "map.json")
    weapons = load_package("weapons", weapons_dir, "manifest.json") if weapons_dir and weapons_dir.is_dir() else None
    sounds = load_package("sounds", sounds_dir, "sounds.json") if sounds_dir and sounds_dir.is_dir() else None
    return maps, weapons, sounds


class Bucket(Protocol):
    def list(self, prefix: str) -> dict[str, int]: ...
    def get_json(self, key: str) -> dict[str, Any] | None: ...
    def put(self, key: str, path: Path, content_type: str, cache_control: str, sha256: str) -> None: ...
    def delete(self, key: str) -> None: ...


class AwsBucket:
    """R2 through the aws CLI's s3api, like r2.py."""

    def __init__(self, bucket: str, endpoint_url: str, aws: str = "aws", profile: str | None = None):
        self.bucket = bucket
        self.aws = aws
        self.common = ["--endpoint-url", endpoint_url] + (["--profile", profile] if profile else [])

    def _run(self, *args: str) -> str:
        completed = subprocess.run(
            [self.aws, "s3api", *args, "--bucket", self.bucket, *self.common],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    def list(self, prefix: str) -> dict[str, int]:
        listing = json.loads(self._run("list-objects-v2", "--prefix", prefix) or "{}") or {}
        return {item["Key"]: int(item.get("Size", 0)) for item in listing.get("Contents") or [] if "Key" in item}

    def get_json(self, key: str) -> dict[str, Any] | None:
        if key not in self.list(key):
            return None
        with tempfile.TemporaryDirectory(prefix="csdemo-cs2-index-") as temporary:
            output = Path(temporary) / "object.json"
            self._run("get-object", "--key", key, str(output))
            value = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {key}")
        return value

    def put(self, key: str, path: Path, content_type: str, cache_control: str, sha256: str) -> None:
        self._run(
            "put-object",
            "--key", key,
            "--body", str(path),
            "--content-type", content_type,
            "--cache-control", cache_control,
            "--metadata", f"sha256={sha256}",
        )
        remote = json.loads(self._run("head-object", "--key", key))
        if remote.get("ContentLength") != path.stat().st_size:
            raise RuntimeError(f"uploaded object has an unexpected size: {key}")
        metadata = {str(name).lower(): value for name, value in (remote.get("Metadata") or {}).items()}
        if metadata.get("sha256") != sha256:
            raise RuntimeError(f"uploaded object has an unexpected checksum: {key}")

    def delete(self, key: str) -> None:
        # single deletes: R2 rejects the checksum headers newer CLIs send on DeleteObjects
        self._run("delete-object", "--key", key)


def current_key(bucket: Bucket) -> str:
    index = bucket.get_json(INDEX_KEY)
    return str(index.get("bake") or "") if index else ""


def publish(
    bucket: Bucket,
    maps: dict[str, Package],
    weapons: Package | None,
    sounds: Package | None,
    *,
    steam_build_id: str,
    bake_key: str,
    expected_maps: Sequence[str] = (),
    workers: int = 8,
) -> dict[str, Any]:
    """Uploads changed packages, replaces the index, then deletes everything else under cs2/.

    A package missing from this build (a map that failed to bake) keeps its published version.
    """
    previous = bucket.get_json(INDEX_KEY) or {}
    listing = bucket.list(PREFIX)

    def still_published(entry: Any) -> bool:
        """Whether a previous index entry's package is still complete in the bucket."""
        path = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(path, str) or not re.fullmatch(r"[a-z0-9_]+(/[a-z0-9_]+)*/[0-9a-f]+/", path):
            return False
        sizes = [size for key, size in listing.items() if key.startswith(PREFIX + path)]
        return bool(sizes) and len(sizes) == entry.get("files") and sum(sizes) == entry.get("bytes")

    carried: list[str] = []
    missing: list[str] = []
    map_entries = {name: package.entry() for name, package in maps.items()}
    for name in expected_maps:
        if name in map_entries:
            continue
        entry = (previous.get("maps") or {}).get(name)
        if still_published(entry):
            map_entries[name] = entry
            carried.append(f"maps/{name}")
        else:
            missing.append(f"maps/{name}")
    singles: dict[str, Any] = {}
    for kind, package in (("weapons", weapons), ("sounds", sounds)):
        if package is not None:
            singles[kind] = package.entry()
        elif still_published(previous.get(kind)):
            singles[kind] = previous[kind]
            carried.append(kind)
        else:
            missing.append(kind)
    if not map_entries:
        raise RuntimeError("refusing to publish an index without any map")

    uploads = []
    unchanged = []
    for package in [*maps.values(), *(p for p in (weapons, sounds) if p is not None)]:
        pending = [
            (PREFIX + package.path + relative, path, digest)
            for relative, (path, digest) in package.files.items()
            if listing.get(PREFIX + package.path + relative) != path.stat().st_size
        ]
        if pending:
            uploads.extend(pending)
        else:
            unchanged.append(package.kind)

    def upload(item: tuple[str, Path, str]) -> None:
        key, path, digest = item
        bucket.put(key, path, CONTENT_TYPES[path.suffix.lower()], IMMUTABLE_CACHE, digest)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(upload, uploads))

    complete = not (carried or missing)
    index = {
        "format": FORMAT,
        "steam_build_id": steam_build_id,
        # stays partial until old versions are deleted, so a failed cleanup or bake is retried next run
        "bake": f"{bake_key}+partial",
        "published_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "maps": dict(sorted(map_entries.items())),
        **singles,
    }
    _put_index(bucket, index)

    kept = [PREFIX + entry["path"] for entry in [*map_entries.values(), *singles.values()]]
    stale = sorted(
        key for key in bucket.list(PREFIX) if key != INDEX_KEY and not any(key.startswith(prefix) for prefix in kept)
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(bucket.delete, stale))
    if complete:
        index["bake"] = bake_key
        _put_index(bucket, index)

    return {
        "index": index,
        "uploaded": len(uploads),
        "uploaded_bytes": sum(path.stat().st_size for _, path, _ in uploads),
        "unchanged": unchanged,
        "carried_over": carried,
        "missing": missing,
        "deleted": len(stale),
    }


def _put_index(bucket: Bucket, index: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="csdemo-cs2-index-") as temporary:
        path = Path(temporary) / "index.json"
        path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        bucket.put(INDEX_KEY, path, CONTENT_TYPES[".json"], INDEX_CACHE, hashlib.sha256(path.read_bytes()).hexdigest())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    key_parser = subparsers.add_parser("current-key", help="print the bake key of the published index")
    publish_parser = subparsers.add_parser("publish", help="publish baked packages and delete older versions")
    for command_parser in (key_parser, publish_parser):
        command_parser.add_argument("--bucket", required=True)
        command_parser.add_argument("--endpoint-url", required=True)
        command_parser.add_argument("--aws", default="aws")
        command_parser.add_argument("--profile")
    publish_parser.add_argument("--maps-dir", type=Path, help="directory of cs2render packages, one per map")
    publish_parser.add_argument("--weapons-dir", type=Path)
    publish_parser.add_argument("--sounds-dir", type=Path)
    publish_parser.add_argument("--expected-maps", nargs="*", default=[], help="maps to keep published if not rebuilt")
    publish_parser.add_argument("--steam-build-id", required=True)
    publish_parser.add_argument("--bake-key", required=True)
    publish_parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args(argv)

    bucket = AwsBucket(arguments.bucket, arguments.endpoint_url, arguments.aws, arguments.profile)
    if arguments.command == "current-key":
        print(current_key(bucket))
        return 0
    maps, weapons, sounds = collect(arguments.maps_dir, arguments.weapons_dir, arguments.sounds_dir)
    report = publish(
        bucket,
        maps,
        weapons,
        sounds,
        steam_build_id=arguments.steam_build_id,
        bake_key=arguments.bake_key,
        expected_maps=arguments.expected_maps,
        workers=arguments.workers,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
