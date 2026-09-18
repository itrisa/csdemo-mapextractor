import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from csdemo_mapextractor.geometry import CollisionGeometry
from csdemo_mapextractor.layered import CollisionLayer
from csdemo_mapextractor.release import build_release, discover_maps, steam_versions
from csdemo_mapextractor.source2viewer import MapAssets


class ReleaseTests(unittest.TestCase):
    def test_discovers_supported_map_vpks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cs2_dir = Path(directory)
            maps_dir = cs2_dir / "game" / "csgo" / "maps"
            maps_dir.mkdir(parents=True)
            for name in (
                "de_mirage.vpk",
                "de_ancient.vpk",
                "de_mirage_preview.vpk",
                "lobby_mapveto.vpk",
                "graphics_settings.vpk",
            ):
                (maps_dir / name).write_bytes(b"vpk")

            self.assertEqual(("de_ancient", "de_mirage"), discover_maps(cs2_dir))

    def test_reads_release_versions_from_steam_inf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cs2_dir = Path(directory)
            steam_inf = cs2_dir / "game" / "csgo" / "steam.inf"
            steam_inf.parent.mkdir(parents=True)
            steam_inf.write_text(
                "ClientVersion=2000908\nPatchVersion=1.41.8.1\n",
                encoding="utf-8",
            )

            versions = steam_versions(cs2_dir)

            self.assertEqual("2000908", versions["client_version"])
            self.assertEqual("1.41.8.1", versions["patch_version"])
            self.assertIsNone(versions["server_version"])

    def test_builds_atomic_versioned_release(self) -> None:
        geometry = CollisionGeometry(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0, 1, 2),),
        )
        layer = CollisionLayer("default", (), geometry)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cs2_dir = root / "cs2"
            steam_inf = cs2_dir / "game" / "csgo" / "steam.inf"
            steam_inf.parent.mkdir(parents=True)
            steam_inf.write_text("ClientVersion=2000908\n", encoding="utf-8")
            output = root / "release"
            assets = MapAssets(
                overview=b'"de_mirage" {}',
                logo=b"<svg/>",
                radars={"default": b"\x89PNG\r\n\x1a\nradar"},
            )
            with patch(
                "csdemo_mapextractor.release.source2viewer.extract_world_physics",
                return_value=b"world",
            ), patch(
                "csdemo_mapextractor.release.source2viewer.extract_entity_physics",
                return_value=b"entities",
            ), patch(
                "csdemo_mapextractor.release.source2viewer.extract_entity_data",
                return_value="m_entityKeyValues = []",
            ), patch(
                "csdemo_mapextractor.release.source2viewer.extract_map_assets",
                return_value=assets,
            ), patch(
                "csdemo_mapextractor.release.source2viewer.version",
                return_value="20.0",
            ), patch(
                "csdemo_mapextractor.release.source2viewer.list_map_asset_resources",
                return_value=frozenset(),
            ), patch(
                "csdemo_mapextractor.release.from_vrf",
                return_value=(layer,),
            ), patch(
                "csdemo_mapextractor.release.entity_layers",
                return_value=(),
            ), patch(
                "csdemo_mapextractor.release.dumps_layered",
                return_value=b"collision",
            ):
                result = build_release(
                    cs2_dir,
                    Path("Source2Viewer-CLI"),
                    output,
                    maps=("de_mirage",),
                    steam_build_id="25218825",
                )

            version_dir = output / "v1" / "2000908-25218825"
            self.assertEqual(version_dir, result.version_directory)
            self.assertEqual(b"collision", (version_dir / "maps" / "de_mirage" / "collision.glb").read_bytes())
            root_index = json.loads((output / "index.json").read_text(encoding="utf-8"))
            self.assertEqual("25218825", root_index["steam_build_id"])
            self.assertEqual("2000908-25218825", root_index["current"])
            manifest = json.loads((version_dir / "index.json").read_text(encoding="utf-8"))
            self.assertEqual("maps/de_mirage/logo.svg", manifest["maps"][0]["files"]["logo"]["path"])
            self.assertEqual("default", manifest["maps"][0]["radars"][0]["section"])

    def test_failed_build_does_not_replace_current_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cs2_dir = root / "cs2"
            steam_inf = cs2_dir / "game" / "csgo" / "steam.inf"
            steam_inf.parent.mkdir(parents=True)
            steam_inf.write_text("ClientVersion=2000908\n", encoding="utf-8")
            output = root / "release"
            output.mkdir()
            current = output / "index.json"
            current.write_text('{"current":"previous"}\n', encoding="utf-8")
            with patch(
                "csdemo_mapextractor.release.source2viewer.list_map_asset_resources",
                return_value=frozenset(),
            ), patch(
                "csdemo_mapextractor.release.source2viewer.extract_world_physics",
                side_effect=RuntimeError("broken map"),
            ):
                with self.assertRaisesRegex(RuntimeError, "broken map"):
                    build_release(
                        cs2_dir,
                        Path("tool"),
                        output,
                        maps=("de_mirage",),
                        steam_build_id="25218825",
                    )

            self.assertEqual('{"current":"previous"}\n', current.read_text(encoding="utf-8"))
            self.assertFalse((output / "v1" / "2000908-25218825").exists())


if __name__ == "__main__":
    unittest.main()
