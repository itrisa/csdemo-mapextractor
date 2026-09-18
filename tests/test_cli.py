import json
from pathlib import Path
import struct
import tempfile
import unittest

from csdemo_mapextractor.cli import _select_cache_layers, main
from csdemo_mapextractor.geometry import CollisionGeometry
from csdemo_mapextractor.layered import CollisionLayer


class CliTests(unittest.TestCase):
    def test_client_cache_selects_only_default_and_enabled_func_brush(self) -> None:
        geometry = CollisionGeometry(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0, 1, 2),),
        )
        layers = tuple(
            CollisionLayer(name, (), geometry)
            for name in (
                "default",
                "sky",
                "entity:func_brush",
                "entity:func_brush:npcclip+playerclip",
            )
        )

        selected = _select_cache_layers(layers, all_layers=False)

        self.assertEqual(["default", "entity:func_brush"], [layer.name for layer in selected])

    def test_convert_tri_to_glb_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "map.tri"
            output = root / "map.glb"
            source.write_bytes(struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0))

            self.assertEqual(0, main(["convert", str(source), str(output), "--map", "synthetic"]))
            self.assertEqual(0, main(["validate", str(output)]))

            manifest = json.loads(output.with_suffix(".glb.manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("synthetic", manifest["map"])
            self.assertEqual(3, manifest["vertex_count"])
            self.assertEqual(1, manifest["triangle_count"])
            self.assertEqual(64, len(manifest["sha256"]))


if __name__ == "__main__":
    unittest.main()