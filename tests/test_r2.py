import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from csdemo_mapextractor.r2 import current_build, main, publish


class R2Tests(unittest.TestCase):
    def test_empty_bucket_has_no_current_build(self) -> None:
        completed = subprocess.CompletedProcess(
            ["aws"], 0, stdout="{}"
        )
        with patch(
            "csdemo_mapextractor.r2.subprocess.run",
            return_value=completed,
        ) as run:
            build_id = current_build(
                bucket="maps",
                endpoint_url="https://account.r2.cloudflarestorage.com",
            )

        self.assertIsNone(build_id)
        self.assertEqual("list-objects-v2", run.call_args.args[0][2])

    def test_reads_current_build_from_root_index(self) -> None:
        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            operation = command[2]
            if operation == "list-objects-v2":
                return subprocess.CompletedProcess(
                    command, 0, stdout='{"Contents":[{"Key":"index.json"}]}'
                )
            if operation == "head-object":
                return subprocess.CompletedProcess(command, 0, stdout="{}")
            output = Path(command[command.index("index.json") + 1])
            output.write_text('{"steam_build_id":"25218825"}', encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="{}")

        with patch(
            "csdemo_mapextractor.r2.subprocess.run",
            side_effect=fake_run,
        ):
            self.assertEqual(
                "25218825",
                current_build(
                    bucket="maps",
                    endpoint_url="https://account.r2.cloudflarestorage.com",
                ),
            )

    def _write_release(self, root: Path, payloads: dict[str, bytes]) -> tuple[str, dict]:
        objects = root / "v1" / "objects"
        objects.mkdir(parents=True)
        files = {}
        for name, data in payloads.items():
            digest = hashlib.sha256(data).hexdigest()
            (objects / f"{digest}.glb").write_bytes(data)
            files[name] = {
                "path": f"objects/{digest}.glb",
                "content_type": "model/gltf-binary",
                "bytes": len(data),
                "sha256": digest,
            }
        manifest = {"maps": [{"name": "de_mirage", "files": files, "radars": []}]}
        manifest_key = "v1/release-2000908-25218825-aaaaaaaaaaaa.json"
        (root / manifest_key).write_text(json.dumps(manifest), encoding="utf-8")
        (root / "index.json").write_text(
            json.dumps({"manifest": manifest_key}), encoding="utf-8"
        )
        return manifest_key, manifest

    def test_uploads_missing_objects_then_manifest_then_root_and_prunes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_key, manifest = self._write_release(
                root, {"collision": b"new-glb", "overview": b"shared"}
            )
            shared_key = "v1/" + manifest["maps"][0]["files"]["overview"]["path"]
            new_key = "v1/" + manifest["maps"][0]["files"]["collision"]["path"]
            previous_key = "v1/release-2000907-25000000-bbbbbbbbbbbb.json"
            oldest_key = "v1/release-2000906-24000000-cccccccccccc.json"
            previous_object = "v1/objects/" + "1" * 64 + ".glb"
            oldest_object = "v1/objects/" + "2" * 64 + ".glb"
            remote = {
                shared_key: {"Size": len(b"shared"), "LastModified": "2026-01-01T00:00:00Z"},
                previous_key: {"Size": 10, "LastModified": "2026-02-01T00:00:00Z"},
                oldest_key: {"Size": 10, "LastModified": "2026-01-01T00:00:00Z"},
                previous_object: {"Size": 3, "LastModified": "2026-02-01T00:00:00Z"},
                oldest_object: {"Size": 3, "LastModified": "2026-01-01T00:00:00Z"},
                "v1/2000905/maps/de_mirage/collision.glb": {"Size": 3},
            }
            remote_manifests = {
                previous_key: {
                    "maps": [
                        {
                            "files": {"collision": {"path": "objects/" + "1" * 64 + ".glb"}},
                            "radars": [{"path": shared_key.removeprefix("v1/")}],
                        }
                    ]
                },
            }
            uploads: list[str] = []
            deleted: list[str] = []

            def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
                operation = command[2]
                if operation == "list-objects-v2":
                    prefix = command[command.index("--prefix") + 1]
                    contents = [
                        {"Key": key, **value}
                        for key, value in remote.items()
                        if key.startswith(prefix)
                    ]
                    return subprocess.CompletedProcess(
                        command, 0, stdout=json.dumps({"Contents": contents})
                    )
                if operation == "get-object":
                    key = command[command.index("--key") + 1]
                    Path(command[command.index("--key") + 2]).write_text(
                        json.dumps(remote_manifests[key]), encoding="utf-8"
                    )
                    return subprocess.CompletedProcess(command, 0, stdout="{}")
                if operation == "delete-object":
                    key = command[command.index("--key") + 1]
                    deleted.append(key)
                    remote.pop(key)
                    return subprocess.CompletedProcess(command, 0, stdout="{}")
                key = command[command.index("--key") + 1]
                if operation == "put-object":
                    uploads.append(key)
                    body = root / Path(key)
                    remote[key] = {"Size": body.stat().st_size, "LastModified": "2026-03-01T00:00:00Z"}
                    return subprocess.CompletedProcess(command, 0)
                body = root / Path(key)
                digest = hashlib.sha256(body.read_bytes()).hexdigest()
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {"ContentLength": body.stat().st_size, "Metadata": {"sha256": digest}}
                    ),
                )

            with patch("csdemo_mapextractor.r2.subprocess.run", side_effect=fake_run):
                result = publish(
                    root,
                    bucket="maps",
                    endpoint_url="https://account.r2.cloudflarestorage.com",
                )

            self.assertEqual([new_key, manifest_key, "index.json"], uploads)
            self.assertEqual([shared_key], result["skipped"])
            self.assertEqual(manifest_key, result["manifest"])
            self.assertEqual(
                sorted([oldest_key, oldest_object, "v1/2000905/maps/de_mirage/collision.glb"]),
                sorted(deleted),
            )
            self.assertEqual(
                {shared_key, new_key, manifest_key, previous_key, previous_object, "index.json"},
                set(remote),
            )

    def test_rejects_asset_that_does_not_match_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = self._write_release(root, {"collision": b"glb"})
            path = root / "v1" / manifest["maps"][0]["files"]["collision"]["path"]
            path.write_bytes(b"tampered")
            with patch("csdemo_mapextractor.r2.subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "does not match manifest"):
                    publish(
                        root,
                        bucket="maps",
                        endpoint_url="https://account.r2.cloudflarestorage.com",
                    )
            run.assert_not_called()

    def test_standalone_current_build_output(self) -> None:
        output = StringIO()
        with (
            patch(
                "csdemo_mapextractor.r2.current_build",
                return_value="25218825",
            ),
            redirect_stdout(output),
        ):
            self.assertEqual(
                0,
                main(
                    [
                        "current-build",
                        "--bucket",
                        "maps",
                        "--endpoint-url",
                        "https://account.r2.cloudflarestorage.com",
                    ]
                ),
            )

        self.assertEqual("25218825\n", output.getvalue())

    def test_standalone_publish_output(self) -> None:
        output = StringIO()
        with (
            patch(
                "csdemo_mapextractor.r2.publish",
                return_value={"manifest": "v1/2000908/index.json"},
            ) as publish_release,
            redirect_stdout(output),
        ):
            self.assertEqual(
                0,
                main(
                    [
                        "publish",
                        "--release-dir",
                        "release",
                        "--bucket",
                        "maps",
                        "--endpoint-url",
                        "https://account.r2.cloudflarestorage.com",
                    ]
                ),
            )

        publish_release.assert_called_once_with(
            Path("release"),
            bucket="maps",
            endpoint_url="https://account.r2.cloudflarestorage.com",
            aws="aws",
            profile=None,
            keep=2,
        )
        self.assertEqual(
            '{\n  "manifest": "v1/2000908/index.json"\n}\n',
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
