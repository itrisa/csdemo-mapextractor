"""Publish a validated release directory through the R2 S3-compatible API."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any, Sequence


_CONTENT_TYPES = {
    ".glb": "model/gltf-binary",
    ".json": "application/json",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
}
_IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
_PREFIX = "v1/"
_MANIFEST_KEY = re.compile(r"v1/release-[A-Za-z0-9._-]+\.json")
_OBJECT_PATH = re.compile(r"objects/[0-9a-f]{64}\.(?:glb|png|svg|txt)")
DEFAULT_KEEP = 2
_ROOT_CACHE = "public, max-age=300, must-revalidate"


def current_build(
    *,
    bucket: str,
    endpoint_url: str,
    aws: str = "aws",
    profile: str | None = None,
) -> str | None:
    common = ["--endpoint-url", endpoint_url]
    if profile is not None:
        common.extend(("--profile", profile))
    if _object_metadata(aws, common, bucket, "index.json") is None:
        return None
    with tempfile.TemporaryDirectory(prefix="csdemo-r2-index-") as temporary:
        output = Path(temporary) / "index.json"
        subprocess.run(
            [
                aws,
                "s3api",
                "get-object",
                "--bucket",
                bucket,
                "--key",
                "index.json",
                str(output),
                *common,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        index = _read_json(output)
    build_id = index.get("steam_build_id")
    if build_id is None:
        return None
    if not isinstance(build_id, str) or not build_id.isdecimal():
        raise ValueError("root index contains an invalid steam_build_id")
    return build_id


def publish(
    release_dir: Path,
    *,
    bucket: str,
    endpoint_url: str,
    aws: str = "aws",
    profile: str | None = None,
    keep: int = DEFAULT_KEEP,
) -> dict[str, object]:
    """Upload missing objects, then the manifest, then the root index, then prune.

    Objects are content-addressed, so an object already present with the expected
    size is identical and is never uploaded again.
    """
    if keep < 1:
        raise ValueError("keep must be at least 1")
    root_index_path = release_dir / "index.json"
    root_index = _read_json(root_index_path)
    manifest_key = _required_string(root_index, "manifest")
    if _MANIFEST_KEY.fullmatch(manifest_key) is None:
        raise ValueError(f"unexpected release manifest key: {manifest_key}")
    manifest_path = release_dir / Path(manifest_key)
    manifest = _read_json(manifest_path)
    objects = _validate_release(release_dir, manifest_key, manifest)

    common = ["--endpoint-url", endpoint_url]
    if profile is not None:
        common.extend(("--profile", profile))
    remote = {item["Key"]: item for item in _list_objects(aws, common, bucket, _PREFIX)}

    uploaded = []
    skipped = []
    for key, path in objects.items():
        if remote.get(key, {}).get("Size") == path.stat().st_size:
            skipped.append(key)
            continue
        uploaded.append(
            _put_and_verify(
                aws, common, bucket, key, path, cache_control=_IMMUTABLE_CACHE
            )
        )
    uploaded.append(
        _put_and_verify(
            aws,
            common,
            bucket,
            manifest_key,
            manifest_path,
            cache_control=_IMMUTABLE_CACHE,
        )
    )
    root_record = _put_and_verify(
        aws,
        common,
        bucket,
        "index.json",
        root_index_path,
        cache_control=_ROOT_CACHE,
    )
    deleted = prune(
        aws=aws,
        common=common,
        bucket=bucket,
        current_manifest_key=manifest_key,
        current_manifest=manifest,
        keep=keep,
    )
    return {
        "format": "csdemo-r2-publish-v2",
        "bucket": bucket,
        "manifest": manifest_key,
        "uploaded": uploaded,
        "skipped": skipped,
        "deleted": deleted,
        "root_index": root_record,
    }


def prune(
    *,
    aws: str,
    common: list[str],
    bucket: str,
    current_manifest_key: str,
    current_manifest: dict[str, Any],
    keep: int,
) -> list[str]:
    """Delete everything under v1/ not referenced by the newest `keep` releases."""
    listing = _list_objects(aws, common, bucket, _PREFIX)
    manifests = sorted(
        (
            item
            for item in listing
            if _MANIFEST_KEY.fullmatch(item["Key"])
            and item["Key"] != current_manifest_key
        ),
        key=lambda item: str(item.get("LastModified", "")),
        reverse=True,
    )
    kept = {current_manifest_key: current_manifest}
    for item in manifests[: keep - 1]:
        kept[item["Key"]] = _get_json(aws, common, bucket, item["Key"])
    referenced = set(kept)
    for key, manifest in kept.items():
        referenced.update(_manifest_object_keys(key, manifest))
    stale = sorted(
        item["Key"]
        for item in listing
        if item["Key"].startswith(_PREFIX) and item["Key"] not in referenced
    )
    # Single-object deletes: R2 rejects the checksum headers newer CLIs send on DeleteObjects.
    for key in stale:
        subprocess.run(
            [aws, "s3api", "delete-object", "--bucket", bucket, "--key", key, *common],
            check=True,
            capture_output=True,
            text=True,
        )
    return stale


def _validate_release(
    release_dir: Path, manifest_key: str, manifest: dict[str, Any]
) -> dict[str, Path]:
    objects: dict[str, Path] = {}
    for key, record in zip(
        _manifest_object_keys(manifest_key, manifest), _manifest_records(manifest)
    ):
        path = release_dir / Path(key)
        if not path.is_file():
            raise FileNotFoundError(f"release asset was not found: {path}")
        data = path.read_bytes()
        if len(data) != record.get("bytes"):
            raise ValueError(f"release asset size does not match manifest: {path}")
        digest = hashlib.sha256(data).hexdigest()
        if digest != record.get("sha256") or not path.name.startswith(digest + "."):
            raise ValueError(f"release asset checksum does not match manifest: {path}")
        objects[key] = path
    return objects


def _manifest_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for map_record in manifest.get("maps", []):
        records.extend(map_record.get("files", {}).values())
        records.extend(map_record.get("radars", []))
    return records


def _manifest_object_keys(manifest_key: str, manifest: dict[str, Any]) -> list[str]:
    base = PurePosixPath(manifest_key).parent
    keys = []
    for record in _manifest_records(manifest):
        relative = _required_string(record, "path")
        if _OBJECT_PATH.fullmatch(relative) is None:
            raise ValueError(f"unexpected release object path: {relative}")
        keys.append((base / relative).as_posix())
    return keys


def _list_objects(
    aws: str, common: list[str], bucket: str, prefix: str
) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            aws,
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            *common,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    listing = json.loads(completed.stdout or "{}") or {}
    return [item for item in listing.get("Contents") or [] if "Key" in item]


def _get_json(aws: str, common: list[str], bucket: str, key: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="csdemo-r2-get-") as temporary:
        output = Path(temporary) / "object.json"
        subprocess.run(
            [aws, "s3api", "get-object", "--bucket", bucket, "--key", key, str(output), *common],
            check=True,
            capture_output=True,
            text=True,
        )
        return _read_json(output)


def _put_and_verify(
    aws: str,
    common: list[str],
    bucket: str,
    key: str,
    path: Path,
    *,
    cache_control: str,
) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    content_type = _CONTENT_TYPES.get(path.suffix.lower())
    if content_type is None:
        raise ValueError(f"unsupported release file type: {path}")
    subprocess.run(
        [
            aws,
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            str(path),
            "--content-type",
            content_type,
            "--cache-control",
            cache_control,
            "--metadata",
            f"sha256={digest}",
            *common,
        ],
        check=True,
    )
    completed = subprocess.run(
        [
            aws,
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            *common,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    remote = json.loads(completed.stdout)
    if remote.get("ContentLength") != path.stat().st_size:
        raise RuntimeError(f"uploaded object has unexpected size: s3://{bucket}/{key}")
    metadata = {str(name).lower(): value for name, value in remote.get("Metadata", {}).items()}
    if metadata.get("sha256") != digest:
        raise RuntimeError(f"uploaded object has unexpected checksum: s3://{bucket}/{key}")
    return {"key": key, "bytes": path.stat().st_size, "sha256": digest}


def _object_metadata(
    aws: str, common: list[str], bucket: str, key: str
) -> dict[str, Any] | None:
    completed = subprocess.run(
        [
            aws,
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            key,
            "--max-items",
            "1",
            *common,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    listing = json.loads(completed.stdout)
    if not any(item.get("Key") == key for item in listing.get("Contents", [])):
        return None
    completed = subprocess.run(
        [
            aws,
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            *common,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError(f"unexpected metadata for s3://{bucket}/{key}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"release manifest was not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _required_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"required string field {key!r} is missing")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    current_parser = subparsers.add_parser("current-build")
    publish_parser = subparsers.add_parser("publish")
    for command_parser in (current_parser, publish_parser):
        command_parser.add_argument("--bucket", required=True)
        command_parser.add_argument("--endpoint-url", required=True)
        command_parser.add_argument("--aws", default="aws")
        command_parser.add_argument("--profile")
    publish_parser.add_argument("--release-dir", type=Path, required=True)
    publish_parser.add_argument(
        "--keep",
        type=int,
        default=DEFAULT_KEEP,
        help="number of most recent releases (including the new one) to retain",
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "current-build":
        build_id = current_build(
            bucket=arguments.bucket,
            endpoint_url=arguments.endpoint_url,
            aws=arguments.aws,
            profile=arguments.profile,
        )
        print(build_id or "")
        return 0
    result = publish(
        arguments.release_dir,
        bucket=arguments.bucket,
        endpoint_url=arguments.endpoint_url,
        aws=arguments.aws,
        profile=arguments.profile,
        keep=arguments.keep,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
