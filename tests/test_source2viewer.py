from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from csdemo_mapextractor.source2viewer import (
    extract_entity_data,
    extract_entity_physics,
    extract_map_assets,
    extract_world_physics,
    list_map_asset_resources,
    version,
)


class Source2ViewerTests(unittest.TestCase):
    def test_constructs_exact_world_physics_command_and_harvests_glb(self) -> None:
        with tempfile.TemporaryDirectory(prefix="path with spaces ") as directory:
            cs2_dir = Path(directory) / "Counter-Strike Global Offensive"
            map_vpk = cs2_dir / "game" / "csgo" / "maps" / "de_mirage.vpk"
            map_vpk.parent.mkdir(parents=True)
            map_vpk.write_bytes(b"vpk")
            executable = Path(directory) / "Source2Viewer-CLI.exe"

            def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
                output_dir = Path(command[command.index("-o") + 1])
                generated_dir = output_dir / "maps" / "de_mirage"
                generated_dir.mkdir(parents=True)
                (generated_dir / "world_physics.glb").write_bytes(b"model output")
                (generated_dir / "world_physics_physics.glb").write_bytes(b"physics output")
                return subprocess.CompletedProcess(command, 0)

            with patch("csdemo_mapextractor.source2viewer.subprocess.run", side_effect=fake_run) as run:
                output = extract_world_physics(executable, cs2_dir, "de_mirage")

            self.assertEqual(b"physics output", output)
            command = run.call_args.args[0]
            self.assertEqual(str(map_vpk), command[2])
            self.assertEqual("maps/de_mirage/world_physics.vmdl_c", command[4])
            self.assertEqual(
                ["-d", "--gltf_export_format", "glb", "--gltf_export_materials"],
                command[-4:],
            )
            self.assertTrue(run.call_args.kwargs["check"])

    def test_rejects_missing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cs2_dir = Path(directory)
            map_vpk = cs2_dir / "game" / "csgo" / "maps" / "de_mirage.vpk"
            map_vpk.parent.mkdir(parents=True)
            map_vpk.write_bytes(b"vpk")
            with patch("csdemo_mapextractor.source2viewer.subprocess.run"):
                with self.assertRaisesRegex(RuntimeError, "found 0"):
                    extract_world_physics(Path("tool"), cs2_dir, "de_mirage")

    def test_rejects_unsafe_map_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "map name"):
            extract_world_physics(Path("tool"), Path("cs2"), "../pak01")

    def test_extracts_default_entity_physics_companion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cs2_dir = Path(directory)
            map_vpk = cs2_dir / "game" / "csgo" / "maps" / "de_mirage.vpk"
            map_vpk.parent.mkdir(parents=True)
            map_vpk.write_bytes(b"vpk")

            def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
                output_dir = Path(command[command.index("-o") + 1])
                generated = output_dir / "maps" / "de_mirage" / "entities" / "default_ents_physics.glb"
                generated.parent.mkdir(parents=True)
                generated.write_bytes(b"entity physics")
                return subprocess.CompletedProcess(command, 0)

            with patch("csdemo_mapextractor.source2viewer.subprocess.run", side_effect=fake_run) as run:
                output = extract_entity_physics(Path("tool"), cs2_dir, "de_mirage")

            self.assertEqual(b"entity physics", output)
            self.assertIn("maps/de_mirage/entities/default_ents.vents_c", run.call_args.args[0])

    def test_extracts_entity_data_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cs2_dir = Path(directory)
            map_vpk = cs2_dir / "game" / "csgo" / "maps" / "de_mirage.vpk"
            map_vpk.parent.mkdir(parents=True)
            map_vpk.write_bytes(b"vpk")
            completed = subprocess.CompletedProcess(
                ["tool"], 0, stdout="m_entityKeyValues = [ ]", stderr=""
            )
            with patch("csdemo_mapextractor.source2viewer.subprocess.run", return_value=completed) as run:
                output = extract_entity_data(Path("tool"), cs2_dir, "de_mirage")

            self.assertIn("m_entityKeyValues", output)
            self.assertIn("--block", run.call_args.args[0])
            self.assertNotIn("-o", run.call_args.args[0])

    def test_extracts_radar_sections_logo_and_overview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cs2_dir = Path(directory)
            vpk = cs2_dir / "game" / "csgo" / "pak01_dir.vpk"
            vpk.parent.mkdir(parents=True)
            vpk.write_bytes(b"vpk")
            overview = b'''
"de_nuke"
{
    "material" "overviews/de_nuke"
    "verticalsections"
    {
        "default" { "AltitudeMax" "10000" }
        "lower" { "AltitudeMax" "-495" }
    }
}
'''

            def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
                output_dir = Path(command[command.index("-o") + 1])
                internal_path = command[command.index("-f") + 1]
                outputs = {
                    "resource/overviews/de_nuke.txt": (
                        Path("resource/overviews/de_nuke.txt"),
                        overview,
                    ),
                    "panorama/images/overheadmaps/de_nuke_radar_psd.vtex_c": (
                        Path("panorama/images/overheadmaps/de_nuke_radar_psd.png"),
                        b"upper",
                    ),
                    "panorama/images/overheadmaps/de_nuke_lower_radar_psd.vtex_c": (
                        Path("panorama/images/overheadmaps/de_nuke_lower_radar_psd.png"),
                        b"lower",
                    ),
                    "panorama/images/map_icons/map_icon_de_nuke.vsvg_c": (
                        Path("panorama/images/map_icons/map_icon_de_nuke.svg"),
                        b"<svg/>",
                    ),
                }
                relative, data = outputs[internal_path]
                output = output_dir / relative
                output.parent.mkdir(parents=True)
                output.write_bytes(data)
                return subprocess.CompletedProcess(command, 0)

            with patch(
                "csdemo_mapextractor.source2viewer.subprocess.run",
                side_effect=fake_run,
            ) as run:
                assets = extract_map_assets(
                    Path("tool"),
                    cs2_dir,
                    "de_nuke",
                    available_resources=frozenset(
                        (
                            "resource/overviews/de_nuke.txt",
                            "panorama/images/overheadmaps/de_nuke_radar_psd.vtex_c",
                            "panorama/images/overheadmaps/de_nuke_lower_radar_psd.vtex_c",
                            "panorama/images/map_icons/map_icon_de_nuke.vsvg_c",
                        )
                    ),
                )

            self.assertEqual(overview, assets.overview)
            self.assertEqual(b"<svg/>", assets.logo)
            self.assertEqual({"default": b"upper", "lower": b"lower"}, assets.radars)
            self.assertEqual(4, run.call_count)
            self.assertNotIn("-d", run.call_args_list[0].args[0])
            self.assertIn("-d", run.call_args_list[1].args[0])

    def test_lists_available_map_asset_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cs2_dir = Path(directory)
            vpk = cs2_dir / "game" / "csgo" / "pak01_dir.vpk"
            vpk.parent.mkdir(parents=True)
            vpk.write_bytes(b"vpk")
            completed = subprocess.CompletedProcess(
                ["tool"],
                0,
                stdout=(
                    "resource/overviews/de_mirage.txt CRC:001 size:10\n"
                    "panorama/images/map_icons/map_icon_de_mirage.vsvg_c CRC:002 size:20\n"
                ),
            )
            with patch(
                "csdemo_mapextractor.source2viewer.subprocess.run",
                return_value=completed,
            ) as run:
                resources = list_map_asset_resources(Path("tool"), cs2_dir)

            self.assertIn("resource/overviews/de_mirage.txt", resources)
            self.assertIn("-l", run.call_args.args[0])

    def test_records_unavailable_optional_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cs2_dir = Path(directory)
            vpk = cs2_dir / "game" / "csgo" / "pak01_dir.vpk"
            vpk.parent.mkdir(parents=True)
            vpk.write_bytes(b"vpk")

            assets = extract_map_assets(
                Path("tool"),
                cs2_dir,
                "ar_pool_day",
                available_resources=frozenset(),
            )

            self.assertIsNone(assets.overview)
            self.assertIsNone(assets.logo)
            self.assertEqual({}, assets.radars)
            self.assertEqual(("overview", "radar:default", "logo"), assets.missing)

    def test_parses_version_identifier_from_diagnostic_output(self) -> None:
        completed = subprocess.CompletedProcess(
            ["tool", "--version"],
            0,
            stdout="Version: 20.0.6980+abc\nOS: Microsoft Windows\n",
            stderr="",
        )
        with patch("csdemo_mapextractor.source2viewer.subprocess.run", return_value=completed):
            self.assertEqual("20.0.6980+abc", version(Path("tool")))


if __name__ == "__main__":
    unittest.main()