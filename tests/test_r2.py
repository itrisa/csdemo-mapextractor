import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from csdemo_mapextractor.r2 import current_build, publish


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

    def test_resumes_matching_version_and_uploads_root_index_last(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            version = root / "v1" / "2000908"
            map_dir = version / "maps" / "de_mirage"
            map_dir.mkdir(parents=True)
            collision = b"glb"
            (map_dir / "collision.glb").write_bytes(collision)
            record = {
                "path": "maps/de_mirage/collision.glb",
                "content_type": "model/gltf-binary",
                "bytes": len(collision),
                "sha256": hashlib.sha256(collision).hexdigest(),
            }
            (version / "index.json").write_text(
                json.dumps({"maps": [{"name": "de_mirage", "files": {"collision": record}, "radars": []}]}),
                encoding="utf-8",
            )
            (root / "index.json").write_text(
                json.dumps({"manifest": "v1/2000908/index.json"}),
                encoding="utf-8",
            )
            uploads: list[str] = []

            def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
                operation = command[2]
                if operation == "list-objects-v2":
                    key = command[command.index("--prefix") + 1]
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=json.dumps({"Contents": [{"Key": key}]}),
                    )
                key = command[command.index("--key") + 1]
                if operation == "put-object":
                    uploads.append(key)
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

            self.assertEqual("index.json", uploads[-1])
            self.assertEqual("v1/2000908/index.json", uploads[-2])
            self.assertEqual(3, len(uploads))
            self.assertEqual("v1/2000908/index.json", result["manifest"])


if __name__ == "__main__":
    unittest.main()
