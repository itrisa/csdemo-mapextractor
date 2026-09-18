import struct
import importlib.util
import unittest
from unittest.mock import patch

from csdemo_mapextractor.geometry import CollisionGeometry
from csdemo_mapextractor.glb import dumps, inspect, loads
from csdemo_mapextractor.profiles import AWPY_DEFAULT, VISUAL_OCCLUDERS


class GlbTests(unittest.TestCase):
    def test_round_trip_preserves_exact_indexed_geometry(self) -> None:
        geometry = CollisionGeometry(
            ((-0.0, 2.5, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)),
            ((0, 1, 2),),
        )

        decoded = loads(dumps(geometry))

        self.assertEqual(geometry.triangles, decoded.triangles)
        self.assertEqual(
            [struct.pack("<3f", *position) for position in geometry.positions],
            [struct.pack("<3f", *position) for position in decoded.positions],
        )

    def test_inspect_reports_contract_metadata(self) -> None:
        geometry = CollisionGeometry(
            ((-1.0, 2.0, 3.0), (4.0, -5.0, 6.0), (7.0, 8.0, -9.0)),
            ((0, 1, 2),),
        )

        details = inspect(dumps(geometry, selection_profile="all-physics-v1"))

        self.assertEqual(3, details["vertex_count"])
        self.assertEqual(1, details["triangle_count"])
        self.assertEqual({"min": [-1.0, -5.0, -9.0], "max": [7.0, 8.0, 6.0]}, details["bounds"])
        self.assertEqual("all-physics-v1", details["extras"]["selection_profile"])

    def test_rejects_invalid_header(self) -> None:
        with self.assertRaisesRegex(ValueError, "header"):
            loads(b"not a glb")

    @unittest.skipUnless(importlib.util.find_spec("meshoptimizer"), "Meshopt extra is not installed")
    def test_meshopt_round_trip_preserves_exact_arrays(self) -> None:
        geometry = CollisionGeometry(
            ((-0.0, 2.5, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)),
            ((2, 0, 1),),
        )

        encoded = dumps(geometry, meshopt=True)
        document, _ = _document_and_binary(encoded)
        decoded = loads(encoded)

        self.assertEqual(["EXT_meshopt_compression"], document["extensionsRequired"])
        self.assertEqual("ATTRIBUTES", document["bufferViews"][0]["extensions"]["EXT_meshopt_compression"]["mode"])
        self.assertEqual("INDICES", document["bufferViews"][1]["extensions"]["EXT_meshopt_compression"]["mode"])
        self.assertEqual(geometry.triangles, decoded.triangles)
        self.assertEqual(
            [struct.pack("<3f", *position) for position in geometry.positions],
            [struct.pack("<3f", *position) for position in decoded.positions],
        )

    def test_filters_source2viewer_material_primitives(self) -> None:
        solid = CollisionGeometry(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0, 1, 2),),
        )
        document, binary = _document_and_binary(dumps(solid))
        document["materials"] = [{"name": "physics_solid"}, {"name": "physics_playerclip"}]
        document["meshes"][0]["primitives"][0]["material"] = 0
        clipped = dict(document["meshes"][0]["primitives"][0], material=1)
        document["meshes"][0]["primitives"].append(clipped)

        with patch("csdemo_mapextractor.glb._parse", return_value=(document, binary)):
            geometry = loads(b"ignored", selection_profile=VISUAL_OCCLUDERS)

        self.assertEqual(1, len(geometry.triangles))

    def test_awpy_profile_selects_only_default_collision_group_meshes(self) -> None:
        geometry = CollisionGeometry(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0, 1, 2),),
        )
        document, binary = _document_and_binary(dumps(geometry))
        primitive = document["meshes"][0]["primitives"][0]
        document["meshes"] = [
            {"name": "physics_group_plaster", "primitives": [primitive]},
            {"name": "physics_sky", "primitives": [primitive]},
        ]

        with patch("csdemo_mapextractor.glb._parse", return_value=(document, binary)):
            selected = loads(b"ignored", selection_profile=AWPY_DEFAULT)

        self.assertEqual(1, len(selected.triangles))

    def test_awpy_profile_rejects_glb_without_collision_group_names(self) -> None:
        geometry = CollisionGeometry(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0, 1, 2),),
        )
        with self.assertRaisesRegex(ValueError, "mesh names"):
            loads(dumps(geometry), selection_profile=AWPY_DEFAULT)


def _document_and_binary(data: bytes) -> tuple[dict, bytes]:
    import csdemo_mapextractor.glb as glb_module

    return glb_module._parse(data)


if __name__ == "__main__":
    unittest.main()