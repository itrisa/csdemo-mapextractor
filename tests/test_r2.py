import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from csdemo_mapextractor.r2 import publish


class R2Tests(unittest.TestCase):
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
