import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

HAS_BUNDLE_DEPS = all(importlib.util.find_spec(name) for name in ("numpy", "PIL"))
HAS_COLOR_DEPS = HAS_BUNDLE_DEPS and all(
    importlib.util.find_spec(name)
    for name in ("open3d", "scipy", "shapely", "triangle", "meshoptimizer")
)

if HAS_BUNDLE_DEPS:
    import numpy as np
    from PIL import Image

    from csdemo_mapextractor.color import bundle, gltf

from csdemo_mapextractor import colored
from csdemo_mapextractor.release import load_release, write_manifest

METRES_PER_HU = 0.0254  # albedo.HU
TEXTURE_RGBA = (200, 60, 30, 255)


def _collision_glb(positions, triangles) -> bytes:
    """A csdemo collision GLB with one layer (POSITION float32, uint32 indices)."""
    pos = struct.pack(f"<{len(positions) * 3}f", *[c for p in positions for c in p])
    idx = struct.pack(f"<{len(triangles) * 3}I", *[i for t in triangles for i in t])
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "default", "mesh": 0, "extras": {"collision_tags": ["solid"]}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5125, "count": len(triangles) * 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos)},
            {"buffer": 0, "byteOffset": len(pos), "byteLength": len(idx)},
        ],
        "buffers": [{"byteLength": len(pos) + len(idx)}],
    }
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    binary = pos + idx
    total = 12 + 8 + len(js) + 8 + len(binary)
    return (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(js), 0x4E4F534A)
        + js
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def _write_render_world(directory: Path, size_hu: float = 256.0) -> None:
    """A Source2Viewer-style .gltf render world split over two buffers: a textured floor quad (glTF y-up,
    metres, hammer (x, y, z) = glTF (z, x, y)) plus a skipped tools primitive, with the floor texture
    exported next to it."""
    s = size_hu * METRES_PER_HU
    floor_pos = np.array([[0, 0, 0], [s, 0, 0], [s, 0, s], [0, 0, s]], np.float32)
    floor_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    floor_idx = np.array([0, 1, 2, 0, 2, 3], np.uint32)
    tools_pos = floor_pos + np.float32([0, 1, 0])
    buffer0 = floor_pos.tobytes() + floor_uv.tobytes() + floor_idx.tobytes()
    buffer1 = tools_pos.tobytes()
    (directory / "world.bin").write_bytes(buffer0)
    (directory / "world_1.bin").write_bytes(buffer1)
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [{"mesh": 0, "name": "floor"}, {"mesh": 1, "name": "tools"}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]},
            {"primitives": [{"attributes": {"POSITION": 3}, "indices": 2, "material": 1}]},
        ],
        "materials": [
            {
                "name": "floor",
                "doubleSided": True,
                "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0},
                "extras": {
                    "vmat": {
                        "Name": "materials/test/floor.vmat",
                        "ShaderName": "csgo_simple.vfx",
                        "TextureParams": {"g_tColor": "floor_color.vtex"},
                    }
                },
            },
            {"name": "tools", "extras": {"vmat": {"Name": "materials/tools/toolsnodraw.vmat", "ShaderName": "csgo_simple.vfx"}}},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": 2, "componentType": 5125, "count": 6, "type": "SCALAR"},
            {"bufferView": 3, "componentType": 5126, "count": 4, "type": "VEC3"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 48},
            {"buffer": 0, "byteOffset": 48, "byteLength": 32},
            {"buffer": 0, "byteOffset": 80, "byteLength": 24},
            {"buffer": 1, "byteOffset": 0, "byteLength": 48},
        ],
        "buffers": [
            {"byteLength": len(buffer0), "uri": "world.bin"},
            {"byteLength": len(buffer1), "uri": "world_1.bin"},
        ],
    }
    (directory / "world.gltf").write_text(json.dumps(doc), encoding="utf-8")
    Image.new("RGBA", (8, 8), TEXTURE_RGBA).save(directory / "floor_color.png")


def _floor_collision(size_hu: float = 256.0) -> bytes:
    s = size_hu
    return _collision_glb([(0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0)], [(0, 1, 2), (0, 2, 3)])


@unittest.skipUnless(HAS_BUNDLE_DEPS, "numpy and Pillow are required")
class GltfTests(unittest.TestCase):
    def test_multi_buffer_gltf_reads_like_glb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_render_world(root)
            doc, binary = gltf.load(root / "world.gltf")
            gltf.write(root / "world.glb", doc, bytes(binary))
            glb_doc, glb_binary = gltf.load(root / "world.glb")

            for index in range(4):
                np.testing.assert_array_equal(
                    gltf.accessor(doc, binary, index), gltf.accessor(glb_doc, glb_binary, index)
                )
            tools = gltf.accessor(doc, binary, 3)
            self.assertAlmostEqual(1.0, float(tools[0, 1]), places=6)


@unittest.skipUnless(HAS_BUNDLE_DEPS, "numpy and Pillow are required")
class BundleTests(unittest.TestCase):
    def test_bundle_is_deterministic_slim_and_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render = root / "render"
            render.mkdir()
            _write_render_world(render)
            collision = _floor_collision()

            first = bundle.create(root / "a" / "de_test", "de_test", collision, render)
            second = bundle.create(root / "b" / "de_test", "de_test", collision, render)

            self.assertEqual(first, second)
            self.assertEqual(first, bundle.digest(root / "a" / "de_test"))
            index = json.loads((root / "a" / "de_test" / "bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(["floor_color.vtex"], [t["name"] for t in index["textures"]])

            slim, _ = gltf.load(root / "a" / "de_test" / "world.glb")
            self.assertEqual({}, slim["materials"][1])
            self.assertEqual([], slim["meshes"][1]["primitives"])

            name, layers, open_world = bundle.load(root / "a" / "de_test", log=lambda *_: None)
            world = open_world()
            self.assertEqual("de_test", name)
            self.assertEqual(["default"], [layer[0] for layer in layers])
            self.assertEqual(2, len(world.triangles))
            texture = world.textures.get("floor_color.vtex", True)
            np.testing.assert_allclose(
                bundle.A.l2s(texture.mean[:3]), np.array(TEXTURE_RGBA[:3]) / 255, atol=1 / 255
            )

    def test_digest_changes_with_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render = root / "render"
            render.mkdir()
            _write_render_world(render)
            first = bundle.create(root / "a", "de_test", _floor_collision(), render)
            second = bundle.create(root / "b", "de_test", _floor_collision(512.0), render)
            self.assertNotEqual(first, second)


@unittest.skipUnless(HAS_COLOR_DEPS, "colour extras (open3d, scipy, shapely, triangle, meshoptimizer) are required")
class ColorizeTests(unittest.TestCase):
    def test_colours_collision_from_render_texture(self) -> None:
        from csdemo_mapextractor.color import pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render = root / "render"
            render.mkdir()
            _write_render_world(render)
            bundle.create(root / "de_test", "de_test", _floor_collision(), render)

            from csdemo_mapextractor.color import encode as encode_module

            captured = []

            def capture(layers, map_name):
                captured.extend(layers)
                return encode_module.encode(layers, map_name)

            with patch.object(pipeline, "encode", side_effect=capture):
                data = pipeline.colorize(root / "de_test", workers=1, log=lambda *_: None)

            self.assertEqual(b"glTF", data[:4])
            doc = json.loads(data[20 : 20 + struct.unpack_from("<I", data, 12)[0]])
            primitive = doc["meshes"][0]["primitives"][0]
            self.assertIn("COLOR_0", primitive["attributes"])
            self.assertIn("EXT_meshopt_compression", doc.get("extensionsUsed", []))
            (name, _, vertices, _, rgba), = captured
            self.assertEqual("default", name)
            self.assertGreater(len(vertices), 0)
            self.assertTrue((rgba[:, 3] == 255).all())
            # glTF vertex colours are linear
            expected = np.round(bundle.A.s2l(np.array(TEXTURE_RGBA[:3]) / 255) * 255)
            np.testing.assert_allclose(rgba[:, :3], np.broadcast_to(expected, rgba[:, :3].shape), atol=2)


class ColoredPlanTests(unittest.TestCase):
    def _release(self, root: Path, sources: dict[str, str | None]) -> Path:
        output = root / "release"
        (output / "v1" / "objects").mkdir(parents=True)
        maps = []
        for name, source in sources.items():
            record = {"name": name, "files": {"collision": {"path": "objects/x.glb"}}}
            if source:
                record["color_source_sha256"] = source
            maps.append(record)
        write_manifest(output, {"release": "1-2", "client_version": "1", "steam_build_id": "2", "maps": maps})
        return output

    def test_plans_only_maps_without_matching_published_colour(self) -> None:
        published = {"de_a": {"source_sha256": "aaa"}, "de_b": {"source_sha256": "old"}}
        with tempfile.TemporaryDirectory() as directory:
            release = self._release(Path(directory), {"de_a": "aaa", "de_b": "bbb", "de_c": "ccc", "de_d": None})
            with patch.object(colored, "published_colored", return_value=("https://x/m.json", published)):
                self.assertEqual(["de_b", "de_c"], colored.plan(release, "https://x/index.json"))

    def test_attach_uses_built_then_published_and_reports_missing(self) -> None:
        reused = b"reused glb"
        published = {
            "de_a": {
                "path": "objects/r.glb",
                "source_sha256": "aaa",
                "bytes": len(reused),
                "sha256": hashlib.sha256(reused).hexdigest(),
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = self._release(
                root, {"de_a": "aaa", "de_b": "bbb", "de_c": "ccc", "de_d": None, "de_boulder": None}
            )
            built = root / "colored"
            built.mkdir()
            (built / "de_b.glb").write_bytes(b"built glb")
            (built / "de_b.json").write_text(json.dumps({"source_sha256": "bbb"}), encoding="utf-8")
            (built / "de_c.glb").write_bytes(b"stale glb")
            (built / "de_c.json").write_text(json.dumps({"source_sha256": "old"}), encoding="utf-8")

            with patch.object(colored, "published_colored", return_value=("https://x/v1/m.json", published)), patch.object(
                colored, "_download", return_value=reused
            ) as download:
                _, outcome = colored.attach(release, built, "https://x/index.json")

            download.assert_called_once_with("https://x/v1/objects/r.glb", published["de_a"])
            self.assertEqual(
                {"de_a": "reused", "de_b": "built", "de_c": "missing", "de_d": "no-bundle", "de_boulder": "excluded"},
                outcome,
            )
            _, manifest_path, manifest = load_release(release)
            files = {m["name"]: m["files"] for m in manifest["maps"]}
            self.assertEqual("bbb", files["de_b"]["colored"]["source_sha256"])
            self.assertEqual(b"built glb", (manifest_path.parent / files["de_b"]["colored"]["path"]).read_bytes())
            self.assertEqual(b"reused glb", (manifest_path.parent / files["de_a"]["colored"]["path"]).read_bytes())
            self.assertNotIn("colored", files["de_c"])
            self.assertNotIn("colored", files["de_d"])


if __name__ == "__main__":
    unittest.main()
