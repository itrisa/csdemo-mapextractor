import json
from pathlib import Path
import tempfile
import unittest

from csdemo_mapextractor import cs2publish as P


class FakeBucket:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.headers: dict[str, tuple[str, str]] = {}
        self.puts: list[str] = []
        self.deletes: list[str] = []

    def list(self, prefix: str) -> dict[str, int]:
        return {key: len(data) for key, data in self.objects.items() if key.startswith(prefix)}

    def get_json(self, key: str):
        return json.loads(self.objects[key]) if key in self.objects else None

    def put(self, key: str, path: Path, content_type: str, cache_control: str, sha256: str) -> None:
        self.objects[key] = path.read_bytes()
        self.headers[key] = (content_type, cache_control)
        self.puts.append(key)

    def delete(self, key: str) -> None:
        del self.objects[key]
        self.deletes.append(key)


def _write(root: Path, files: dict[str, bytes]) -> None:
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


class Cs2PublishTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.bucket = FakeBucket()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _bake(self, name: str, maps: dict[str, bytes], weapons: bytes = b"w", sounds: bytes = b"s") -> Path:
        out = self.root / name
        for map_name, chunk in maps.items():
            _write(out / "cs2maps" / map_name, {"map.json": b"{}", "geo/c0.glb": chunk, "tex/a.ctex": b"t"})
        _write(out / "cs2weapons", {"manifest.json": b"{}", "world/ak47.glb": weapons})
        _write(out / "cs2sounds", {"sounds.json": b"{}", "core.bin": sounds, "maps/de_inferno.json": b"{}"})
        return out

    def _publish(self, out: Path, expected=("de_inferno", "de_mirage"), key="k1"):
        maps, weapons, sounds = P.collect(out / "cs2maps", out / "cs2weapons", out / "cs2sounds")
        return P.publish(
            self.bucket, maps, weapons, sounds, steam_build_id="1", bake_key=key, expected_maps=expected, workers=2
        )

    def test_first_publish_uploads_packages_and_index(self):
        report = self._publish(self._bake("a", {"de_inferno": b"i1", "de_mirage": b"m1"}))
        index = self.bucket.get_json(P.INDEX_KEY)
        self.assertEqual(index["format"], P.FORMAT)
        self.assertEqual(sorted(index["maps"]), ["de_inferno", "de_mirage"])
        inferno = index["maps"]["de_inferno"]["path"]
        self.assertRegex(inferno, r"^maps/de_inferno/[0-9a-f]{20}/$")
        self.assertIn(f"cs2/{inferno}geo/c0.glb", self.bucket.objects)
        self.assertIn(f"cs2/{index['sounds']['path']}maps/de_inferno.json", self.bucket.objects)
        self.assertEqual(report["uploaded"], 3 + 3 + 2 + 3)
        self.assertEqual(report["deleted"], 0)
        self.assertEqual(self.bucket.headers[f"cs2/{inferno}geo/c0.glb"], ("model/gltf-binary", P.IMMUTABLE_CACHE))
        self.assertEqual(self.bucket.headers[P.INDEX_KEY], ("application/json", "no-cache"))
        self.assertEqual(P.current_key(self.bucket), "k1")

    def test_republish_replaces_only_changed_packages_and_deletes_old_versions(self):
        self._publish(self._bake("a", {"de_inferno": b"i1", "de_mirage": b"m1"}))
        old = self.bucket.get_json(P.INDEX_KEY)
        self.bucket.puts.clear()

        report = self._publish(self._bake("b", {"de_inferno": b"i2", "de_mirage": b"m1"}), key="k2")
        new = self.bucket.get_json(P.INDEX_KEY)
        self.assertNotEqual(old["maps"]["de_inferno"]["path"], new["maps"]["de_inferno"]["path"])
        self.assertEqual(old["maps"]["de_mirage"], new["maps"]["de_mirage"])
        self.assertEqual(sorted(report["unchanged"]), ["maps/de_mirage", "sounds", "weapons"])
        self.assertEqual(report["deleted"], 3)
        self.assertFalse(any(key.startswith("cs2/" + old["maps"]["de_inferno"]["path"]) for key in self.bucket.objects))
        self.assertEqual(self.bucket.puts[-1], P.INDEX_KEY)
        self.assertEqual(len([k for k in self.bucket.objects if k.startswith("cs2/maps/de_inferno/")]), 3)

    def test_failed_map_keeps_its_published_version(self):
        self._publish(self._bake("a", {"de_inferno": b"i1", "de_mirage": b"m1"}))
        mirage = self.bucket.get_json(P.INDEX_KEY)["maps"]["de_mirage"]

        report = self._publish(self._bake("b", {"de_inferno": b"i2"}))
        self.assertEqual(report["carried_over"], ["maps/de_mirage"])
        self.assertEqual(self.bucket.get_json(P.INDEX_KEY)["maps"]["de_mirage"], mirage)
        self.assertIn(f"cs2/{mirage['path']}map.json", self.bucket.objects)
        self.assertEqual(P.current_key(self.bucket), "k1+partial")

    def test_maps_no_longer_hosted_are_removed(self):
        self._publish(self._bake("a", {"de_inferno": b"i1", "de_mirage": b"m1"}))
        report = self._publish(self._bake("b", {"de_inferno": b"i1"}), expected=("de_inferno",))
        self.assertEqual(list(self.bucket.get_json(P.INDEX_KEY)["maps"]), ["de_inferno"])
        self.assertFalse(any(key.startswith("cs2/maps/de_mirage/") for key in self.bucket.objects))
        self.assertEqual(report["deleted"], 3)

    def test_missing_weapons_and_sounds_are_carried_over(self):
        self._publish(self._bake("a", {"de_inferno": b"i1"}), expected=("de_inferno",))
        before = self.bucket.get_json(P.INDEX_KEY)
        maps, _, _ = P.collect(self.root / "a" / "cs2maps", self.root / "none", self.root / "none")
        report = P.publish(self.bucket, maps, None, None, steam_build_id="2", bake_key="k2", expected_maps=["de_inferno"])
        after = self.bucket.get_json(P.INDEX_KEY)
        self.assertEqual(sorted(report["carried_over"]), ["sounds", "weapons"])
        self.assertEqual((after["weapons"], after["sounds"]), (before["weapons"], before["sounds"]))

    def test_refuses_to_publish_without_maps(self):
        with self.assertRaises(RuntimeError):
            P.publish(self.bucket, {}, None, None, steam_build_id="1", bake_key="k", expected_maps=["de_inferno"])
        self.assertEqual(self.bucket.objects, {})

    def test_malformed_previous_entry_is_not_carried_over(self):
        self.bucket.objects[P.INDEX_KEY] = json.dumps({"maps": {"de_mirage": {"path": ""}}}).encode()
        report = self._publish(self._bake("a", {"de_inferno": b"i1"}))
        self.assertEqual(report["missing"], ["maps/de_mirage"])
        self.assertNotIn("de_mirage", self.bucket.get_json(P.INDEX_KEY)["maps"])

    def test_incomplete_previous_package_is_not_carried_over(self):
        self._publish(self._bake("a", {"de_inferno": b"i1", "de_mirage": b"m1"}))
        mirage = self.bucket.get_json(P.INDEX_KEY)["maps"]["de_mirage"]["path"]
        del self.bucket.objects[f"cs2/{mirage}geo/c0.glb"]
        report = self._publish(self._bake("b", {"de_inferno": b"i1"}))
        self.assertEqual(report["missing"], ["maps/de_mirage"])
        self.assertFalse(any(key.startswith("cs2/maps/de_mirage/") for key in self.bucket.objects))

    def test_failed_cleanup_leaves_a_partial_key_and_is_retried(self):
        self._publish(self._bake("a", {"de_inferno": b"i1", "de_mirage": b"m1"}))
        delete = self.bucket.delete
        self.bucket.delete = lambda key: (_ for _ in ()).throw(RuntimeError("delete failed"))
        out = self._bake("b", {"de_inferno": b"i2", "de_mirage": b"m1"})
        with self.assertRaises(RuntimeError):
            self._publish(out, key="k2")
        self.assertEqual(P.current_key(self.bucket), "k2+partial")
        self.assertEqual(len([k for k in self.bucket.objects if k.startswith("cs2/maps/de_inferno/")]), 6)

        self.bucket.delete = delete
        report = self._publish(out, key="k2")
        self.assertEqual((report["uploaded"], report["deleted"]), (0, 3))
        self.assertEqual(P.current_key(self.bucket), "k2")
        self.assertEqual(len([k for k in self.bucket.objects if k.startswith("cs2/maps/de_inferno/")]), 3)

    def test_rejects_unknown_file_types(self):
        _write(self.root / "m" / "de_inferno", {"map.json": b"{}", "notes.txt": b"x"})
        with self.assertRaises(ValueError):
            P.collect(self.root / "m", None, None)

    def test_package_digest_depends_on_paths_and_contents(self):
        _write(self.root / "x", {"map.json": b"{}", "a.bin": b"1"})
        _write(self.root / "y", {"map.json": b"{}", "b.bin": b"1"})
        _write(self.root / "z", {"map.json": b"{}", "a.bin": b"1"})
        digest = lambda name: P.load_package("maps/m", self.root / name, "map.json").digest
        self.assertNotEqual(digest("x"), digest("y"))
        self.assertEqual(digest("x"), digest("z"))


if __name__ == "__main__":
    unittest.main()
