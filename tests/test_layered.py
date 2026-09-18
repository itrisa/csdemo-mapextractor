import importlib.util
import unittest
from unittest.mock import patch

from csdemo_mapextractor.geometry import CollisionGeometry
from csdemo_mapextractor.entity_data import EntityState
from csdemo_mapextractor.glb import dumps as dumps_glb, loads
from csdemo_mapextractor.layered import CollisionLayer, dumps, entity_layers, from_vrf


@unittest.skipUnless(importlib.util.find_spec("meshoptimizer"), "Meshopt extra is not installed")
class LayeredTests(unittest.TestCase):
    def test_layers_remain_separate_in_compressed_glb(self) -> None:
        triangle = CollisionGeometry(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0, 1, 2),),
        )
        layers = (
            CollisionLayer("default", (), triangle, ("concrete",)),
            CollisionLayer("playerclip", ("playerclip",), triangle, ("default",)),
        )

        encoded = dumps(layers, map_name="synthetic")
        import csdemo_mapextractor.glb as glb_module

        document, _ = glb_module._parse(encoded)
        decoded = loads(encoded)

        self.assertEqual(["default", "playerclip"], [node["name"] for node in document["nodes"]])
        self.assertEqual([[], ["playerclip"]], [node["extras"]["collision_tags"] for node in document["nodes"]])
        self.assertEqual(2, len(decoded.triangles))

    def test_vrf_nodes_are_grouped_by_interaction_tags(self) -> None:
        triangle = CollisionGeometry(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0, 1, 2),),
        )
        document, binary = _document_and_binary(dumps_glb(triangle))
        document["nodes"] = [
            {"mesh": 0, "extras": {"InteractAs": [], "SurfaceProperty": "concrete"}},
            {"mesh": 0, "extras": {"InteractAs": ["playerclip"], "SurfaceProperty": "default"}},
        ]

        with patch("csdemo_mapextractor.glb._parse", return_value=(document, binary)):
            layers = from_vrf(b"ignored")

        self.assertEqual(["default", "playerclip"], [layer.name for layer in layers])

    def test_entity_layer_bakes_world_transform_back_to_hammer_coordinates(self) -> None:
        triangle = CollisionGeometry(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0, 1, 2),),
        )
        document, binary = _document_and_binary(dumps_glb(triangle))
        document["nodes"] = [
            {
                "name": "func_brush",
                "mesh": 0,
                "matrix": [
                    0.0, 0.0, 0.0254, 0.0,
                    0.0254, 0.0, 0.0, 0.0,
                    0.0, 0.0254, 0.0, 0.0,
                    5.08, 7.62, 2.54, 1.0,
                ],
                "extras": {"InteractAs": [], "SurfaceProperty": "concrete"},
            }
        ]

        with patch("csdemo_mapextractor.glb._parse", return_value=(document, binary)):
            layers = entity_layers(b"ignored", "func_brush")

        self.assertEqual(1, len(layers))
        layer = layers[0]
        self.assertEqual("entity:func_brush", layer.name)
        self.assertEqual((100.0, 200.0, 300.0), layer.geometry.positions[0])
        self.assertEqual((101.0, 200.0, 300.0), layer.geometry.positions[1])

    def test_entity_layers_separate_interaction_tags(self) -> None:
        triangle = CollisionGeometry(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0, 1, 2),),
        )
        document, binary = _document_and_binary(dumps_glb(triangle))
        basis = [
            0.0, 0.0, 0.0254, 0.0,
            0.0254, 0.0, 0.0, 0.0,
            0.0, 0.0254, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        document["nodes"] = [
            {"name": "func_brush", "mesh": 0, "matrix": basis, "extras": {"InteractAs": []}},
            {
                "name": "func_brush",
                "mesh": 0,
                "matrix": basis,
                "extras": {"InteractAs": ["npcclip", "playerclip"]},
            },
        ]

        with patch("csdemo_mapextractor.glb._parse", return_value=(document, binary)):
            layers = entity_layers(b"ignored", "func_brush")

        self.assertEqual(
            ["entity:func_brush", "entity:func_brush:npcclip+playerclip"],
            [layer.name for layer in layers],
        )

    def test_entity_layers_exclude_disabled_entity_origins(self) -> None:
        triangle = CollisionGeometry(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0, 1, 2),),
        )
        document, binary = _document_and_binary(dumps_glb(triangle))
        basis = [
            0.0, 0.0, 0.0254, 0.0,
            0.0254, 0.0, 0.0, 0.0,
            0.0, 0.0254, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        enabled_matrix = list(basis)
        enabled_matrix[14] = 0.254
        document["nodes"] = [
            {"name": "func_brush", "mesh": 0, "matrix": basis, "extras": {"InteractAs": []}},
            {"name": "func_brush", "mesh": 0, "matrix": enabled_matrix, "extras": {"InteractAs": []}},
        ]
        states = (
            EntityState("func_brush", (0.0, 0.0, 0.0), True),
            EntityState("func_brush", (10.0, 0.0, 0.0), False),
        )

        with patch("csdemo_mapextractor.glb._parse", return_value=(document, binary)):
            layers = entity_layers(b"ignored", "func_brush", states=states)

        self.assertEqual(1, len(layers))
        self.assertEqual(1, len(layers[0].geometry.triangles))
        self.assertEqual(10.0, layers[0].geometry.positions[0][0])


def _document_and_binary(data: bytes) -> tuple[dict, bytes]:
    import csdemo_mapextractor.glb as glb_module

    return glb_module._parse(data)


if __name__ == "__main__":
    unittest.main()