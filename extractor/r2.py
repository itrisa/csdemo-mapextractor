"""Publish a validated release directory through the R2 S3-compatible API."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
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
    force: bool = False,
) -> dict[str, object]:
    root_index_path = release_dir / "index.json"
    root_index = _read_json(root_index_path)
    manifest_key = _required_string(root_index, "manifest")
    manifest_path = release_dir / Path(manifest_key)
    manifest = _read_json(manifest_path)
    _validate_release(release_dir, manifest_key, manifest)

    common = ["--endpoint-url", endpoint_url]
    if profile is not None:
        common.extend(("--profile", profile))
    existing_manifest = _object_metadata(aws, common, bucket, manifest_key)
    if existing_manifest is not None and not force:
        local_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        remote_digest = {
            str(name).lower(): value
            for name, value in existing_manifest.get("Metadata", {}).items()
        }.get("sha256")
        if (
            existing_manifest.get("ContentLength") != manifest_path.stat().st_size
            or remote_digest != local_digest
        ):
            raise FileExistsError(
                f"s3://{bucket}/{manifest_key} exists with different content; "
                "pass --force to replace it"
            )

    version_directory = manifest_path.parent
    uploaded = []
    version_files = sorted(
        path
        for path in version_directory.rglob("*")
        if path.is_file() and path != manifest_path
    )
    version_files.append(manifest_path)
    for path in version_files:
        key = path.relative_to(release_dir).as_posix()
        uploaded.append(
            _put_and_verify(
                aws,
                common,
                bucket,
                key,
                path,
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
    return {
        "format": "csdemo-r2-publish-v1",
        "bucket": bucket,
        "manifest": manifest_key,
        "uploaded": uploaded,
        "root_index": root_record,
    }


def _validate_release(
    release_dir: Path, manifest_key: str, manifest: dict[str, Any]
) -> None:
    version_directory = (release_dir / manifest_key).parent
    records = []
    for map_record in manifest.get("maps", []):
        records.extend(map_record.get("files", {}).values())
        records.extend(map_record.get("radars", []))
    for record in records:
        relative = _safe_relative_path(_required_string(record, "path"))
        path = version_directory / relative
        if not path.is_file():
            raise FileNotFoundError(f"release asset was not found: {path}")
        data = path.read_bytes()
        if len(data) != record.get("bytes"):
            raise ValueError(f"release asset size does not match manifest: {path}")
        if hashlib.sha256(data).hexdigest() != record.get("sha256"):
            raise ValueError(f"release asset checksum does not match manifest: {path}")
    expected = {Path("index.json"), *(_safe_relative_path(_required_string(record, "path")) for record in records)}
    actual = {
        path.relative_to(version_directory)
        for path in version_directory.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise ValueError("release directory contains unlisted or missing files")


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


def _safe_relative_path(value: str) -> Path:
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts or "\\" in value:
        raise ValueError(f"unsafe release path: {value}")
    return Path(*posix.parts)


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
    publish_parser.add_argument("--force", action="store_true")
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
        force=arguments.force,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
