import unittest
from pathlib import Path

import numpy as np

from csdemo_mapextractor.cs2render import materials as M
from csdemo_mapextractor.cs2render.geometry import vertex_paint
from csdemo_mapextractor.cs2render.gltf_world import _uses_secondary_uv
from csdemo_mapextractor.cs2render import sky
from csdemo_mapextractor.cs2render.build import daylight_exposure


def _material(shader: str, ints: dict | None = None) -> dict:
    return {
        "name": "test",
        "extras": {
            "vmat": {
                "Name": "materials/test.vmat",
                "ShaderName": shader,
                "IntParams": ints or {},
                "TextureParams": {"g_tColor": "color.vtex", "g_tNormal": "normal.vtex"},
            }
        },
    }


class VertexPaintTests(unittest.TestCase):
    def test_foliage_ignores_color0(self):
        spec = M.classify(0, _material("csgo_foliage.vfx", {"F_PAINT_VERTEX_COLORS": 1}))
        color = np.array([[1.0, 0.0, 1.0, 0.5]], dtype=np.float32)
        paint = vertex_paint(color, spec.params["vertexPaint"], 1)
        np.testing.assert_array_equal(paint, [[255, 255, 255, 255]])

    def test_complex_without_paint_flag_is_white(self):
        spec = M.classify(0, _material("csgo_vertexlitgeneric.vfx"))
        paint = vertex_paint(np.zeros((2, 4), dtype=np.float32), spec.params["vertexPaint"], 2)
        np.testing.assert_array_equal(paint, np.full((2, 4), 255))

    def test_complex_paint_is_srgb_decoded(self):
        spec = M.classify(0, _material("csgo_complex.vfx", {"F_PAINT_VERTEX_COLORS": 1}))
        paint = vertex_paint(np.array([[0.5, 1.0, 0.0, 0.2]], dtype=np.float32), spec.params["vertexPaint"], 1)
        np.testing.assert_array_equal(paint, [[55, 255, 0, 255]])

    def test_static_overlay_zero_paint_is_unpainted(self):
        spec = M.classify(0, _material("csgo_static_overlay.vfx", {"F_VERTEX_COLOR": 1}))
        color = np.array([[0.0, 0.0, 0.0, 0.0], [0.5, 0.25, 0.0, 0.0]], dtype=np.float32)
        paint = vertex_paint(color, spec.params["vertexPaint"], 2)
        np.testing.assert_array_equal(paint, [[255, 255, 255, 255], [128, 64, 0, 255]])

    def test_environment_paint_is_raw(self):
        spec = M.classify(0, _material("csgo_environment.vfx"))
        color = np.array([[0.5, 0.25, 1.0, 0.5]], dtype=np.float32)
        paint = vertex_paint(color, spec.params["vertexPaint"], 1)
        np.testing.assert_array_equal(paint, [[128, 64, 255, 128]])


class SecondaryUvTests(unittest.TestCase):
    def test_force_uv2_and_secondary_uv_flags(self):
        self.assertTrue(_uses_secondary_uv(_material("csgo_vertexlitgeneric.vfx", {"F_FORCE_UV2": 1})))
        self.assertTrue(_uses_secondary_uv(_material("csgo_complex.vfx", {"F_SECONDARY_UV": 1})))
        self.assertFalse(_uses_secondary_uv(_material("csgo_complex.vfx")))


if __name__ == "__main__":
    unittest.main()


class SkyTests(unittest.TestCase):
    def test_prefers_enabled_prefab_sky(self):
        main = [
            {"classname": "env_sky", "skyname": "main.vmat"},
            {"classname": "env_sky", "skyname": "light.vmat", "StartDisabled": True},
        ]
        prefab = [{"classname": "env_sky", "skyname": "prefab.vmat"}]
        self.assertEqual(sky.select_sky(main, prefab)["skyname"], "prefab.vmat")
        self.assertEqual(sky.select_sky(main, [])["skyname"], "main.vmat")
        self.assertEqual(sky.select_sky(main[1:], [])["skyname"], "light.vmat")
        self.assertIsNone(sky.select_sky([], []))

    def test_material_params(self):
        text = """
            m_name = "g_tSkyTexture"
            m_pValue = resource:"materials/skybox/sky_exr_1.vtex"
            m_name = "g_flBrightnessExposureBias"
            m_flValue = 1.000000
            m_name = "g_flRenderOnlyExposureBias"
            m_flValue = -0.500000
        """
        self.assertEqual(sky.material_params(text), ("materials/skybox/sky_exr_1.vtex", 0.5))

    def test_resample_rotates_into_world_azimuth(self):
        # Source2Viewer latlong: pixel (u, v) looks along (sinV cosU, cosV, sinV sinU) = Source (x, z, -y).
        h, w = 64, 128
        u = (np.arange(w) + 0.5) / w * 2 * np.pi
        v = (np.arange(h) + 0.5) / h * np.pi
        source = np.stack(
            [np.sin(v)[:, None] * np.cos(u)[None, :], -np.sin(v)[:, None] * np.sin(u)[None, :], np.repeat(np.cos(v)[:, None], w, 1)],
            axis=-1,
        )
        target = np.array([np.cos(np.radians(30)) * np.cos(np.radians(10)), np.sin(np.radians(30)) * np.cos(np.radians(10)), np.sin(np.radians(10))])
        spot = np.clip(source @ target, 0, None) ** 400
        latlong = np.concatenate([np.repeat(spot[..., None], 3, -1), np.ones((h, w, 1))], -1).astype(np.float32)
        out = sky.resample(latlong, [0.0, 90.0, 0.0], np.array([2.0, 2.0, 2.0]), 128, 0.625)
        self.assertEqual(out.shape, (40, 128, 4))
        row, column = np.unravel_index(np.argmax(out[..., 0]), out.shape[:2])
        self.assertAlmostEqual((column + 0.5) / 128 * 360, 120, delta=4)
        self.assertAlmostEqual((row + 0.5) / 40 * 0.625 * 180, 80, delta=4)
        self.assertAlmostEqual(float(out[..., 0].max()), 2 * float(latlong[..., 0].max()), delta=0.3)

class DaylightExposureTests(unittest.TestCase):
    def test_auto_exposure_settles_on_minimum(self):
        self.assertAlmostEqual(daylight_exposure({"enableexposure": True, "minexposure": 0.8, "maxexposure": 1.1}), 0.8)

    def test_inverted_range_keeps_minimum(self):
        self.assertAlmostEqual(daylight_exposure({"enableexposure": True, "minexposure": 1.0, "maxexposure": 0.85}), 1.0)

    def test_log_exposure_and_compensation(self):
        entity = {"enableexposure": True, "minlogexposure": -1.0, "minexposure": 5.0, "exposurecompensation": 1.0}
        self.assertAlmostEqual(daylight_exposure(entity), 1.0)

    def test_disabled_or_missing(self):
        self.assertEqual(daylight_exposure(None), 1.0)
        self.assertAlmostEqual(daylight_exposure({"enableexposure": False, "minexposure": 0.5}), 1.0)


class UvSetTests(unittest.TestCase):
    def test_environment_layer_on_uv2_reads_secondary_uv(self):
        material = _material("csgo_environment_blend.vfx", {"g_nUVSet1": 2})
        self.assertTrue(_uses_secondary_uv(material))
        self.assertFalse(_uses_secondary_uv(_material("csgo_environment.vfx", {"g_nUVSet1": 1})))

    def test_shared_overlay_defaults_to_uv2(self):
        self.assertTrue(_uses_secondary_uv(_material("csgo_environment_blend.vfx", {"F_SHARED_COLOR_OVERLAY": 1})))
        self.assertFalse(
            _uses_secondary_uv(
                _material("csgo_environment_blend.vfx", {"F_SHARED_COLOR_OVERLAY": 1, "g_nColorOverlayUVSet": 1})
            )
        )

    def test_spec_carries_layer_uv_sets(self):
        spec = M.classify(0, _material("csgo_environment_blend.vfx", {"g_nUVSet1": 2, "g_nColorOverlayUVSet": 1}))
        self.assertEqual(spec.uv_sets, (2, 1))
        self.assertEqual(spec.overlay_uv_set, 1)


class ProbeSliceOrderTests(unittest.TestCase):
    def test_slices_sort_numerically_past_one_thousand(self):
        from extractor.cs2render.probes import _by_slice_number

        names = ["env_light_probe_volume_atlas_z%03d.exr" % i for i in (1000, 101, 100, 99, 1001)]
        ordered = [path.name for path in _by_slice_number(Path(name) for name in names)]
        self.assertEqual(
            ordered,
            [
                "env_light_probe_volume_atlas_z099.exr",
                "env_light_probe_volume_atlas_z100.exr",
                "env_light_probe_volume_atlas_z101.exr",
                "env_light_probe_volume_atlas_z1000.exr",
                "env_light_probe_volume_atlas_z1001.exr",
            ],
        )


_WORLD_NODE = """<!-- kv3 encoding:text:version{e21c7f3c-8a33-41c5-9977-a76d3a32aa0d} format:generic:version{7412167c-06e9-4698-aff2-e63eb59037e7} -->
{
	m_sceneObjects = [  ]
	m_aggregateSceneObjects =
	[
		{
			m_aggregateMeshes =
			[
				{ m_nDrawCallIndex = 0 m_nLightProbeVolumePrecomputedHandshake = 0 },
				{ m_nDrawCallIndex = 1 m_nLightProbeVolumePrecomputedHandshake = 2015583993 },
			]
			m_renderableModel = resource:"maps/de_test/worldnodes/n0_lr0_agg_merge_bricks_1.vmdl"
		},
	]
}
"""


def _write_world(directory: Path, uv_sets: list[np.ndarray], mesh_name: str = "agg_fragment2") -> Path:
    """A one-triangle glTF whose TEXCOORD_n are uv_sets (each (3, 2))."""
    import json

    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    normals = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (3, 1))
    indices = np.array([0, 1, 2], dtype=np.uint32)
    blobs = [positions, normals, *[np.asarray(uv, dtype=np.float32) for uv in uv_sets], indices]
    data = b"".join(blob.tobytes() for blob in blobs)
    views, accessors, offset = [], [], 0
    for blob in blobs:
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": blob.nbytes})
        is_index = blob.dtype == np.uint32
        accessor = {
            "bufferView": len(views) - 1,
            "componentType": 5125 if is_index else 5126,
            "count": len(blob),
            "type": "SCALAR" if is_index else ("VEC3" if blob.shape[1] == 3 else "VEC2"),
        }
        if not is_index:
            accessor["min"] = blob.min(axis=0).tolist()
            accessor["max"] = blob.max(axis=0).tolist()
        accessors.append(accessor)
        offset += blob.nbytes
    attributes = {"POSITION": 0, "NORMAL": 1}
    attributes.update({f"TEXCOORD_{i}": 2 + i for i in range(len(uv_sets))})
    document = {
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {"name": mesh_name, "primitives": [{"attributes": attributes, "indices": len(blobs) - 1, "material": 0}]}
        ],
        "materials": [_material("csgo_environment.vfx")],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"uri": "world.bin", "byteLength": len(data)}],
    }
    (directory / "world.bin").write_bytes(data)
    path = directory / "world.gltf"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class LightmapUvTests(unittest.TestCase):
    _UV = np.array([[0.1, 0.1], [0.3, 0.1], [0.1, 0.3]])

    def test_probe_lit_fragments_from_world_node(self):
        from extractor.cs2render.worldnodes import probe_lit_meshes

        self.assertEqual(probe_lit_meshes(_WORLD_NODE), {"n0_lr0_agg_merge_bricks_1_fragment2"})

    def test_probe_lit_mesh_has_no_lightmap(self):
        import tempfile
        from extractor.cs2render.gltf_world import GltfWorld

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            path = _write_world(Path(folder), [self._UV * 4, self._UV])
            (lit,) = GltfWorld(path).primitives()
            np.testing.assert_allclose(lit.lightmap_uv, self._UV, atol=1e-6)
            world = GltfWorld(path, frozenset({"agg_fragment2"}))
            (probe,) = world.primitives()
            self.assertIsNone(probe.lightmap_uv)
            np.testing.assert_allclose(probe.uv1, self._UV, atol=1e-6)
            del lit, world, probe

    def test_chart_layout_picks_lightmap_over_trailing_set(self):
        import tempfile
        from PIL import Image
        from extractor.cs2render.gltf_world import GltfWorld

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            charts = np.zeros((64, 64, 3), dtype=np.uint8)
            charts[:32, :32] = (255, 0, 0)
            charts[:32, 32:] = (0, 255, 0)
            (Path(folder) / "lightmaps").mkdir()
            Image.fromarray(charts).save(Path(folder) / "lightmaps" / "debug_chart_color.png")
            straddling = np.array([[0.2, 0.1], [0.9, 0.1], [0.2, 0.4]])
            path = _write_world(Path(folder), [self._UV * 4, self._UV, straddling])
            (primitive,) = GltfWorld(path).primitives()
            np.testing.assert_allclose(primitive.lightmap_uv, self._UV, atol=1e-6)
            np.testing.assert_allclose(primitive.uv1, straddling, atol=1e-6)
            del primitive


class GlassAndOverlayTests(unittest.TestCase):
    def test_animated_static_overlay_is_skipped(self):
        self.assertIsNone(M.classify(0, _material("csgo_static_overlay.vfx", {"F_TEXTURE_ANIMATION": 1})))
        self.assertIsNotNone(M.classify(0, _material("csgo_static_overlay.vfx")))

    def test_glass_uses_dust_texture_and_translucency_remap(self):
        material = _material("csgo_glass.vfx")
        vmat = material["extras"]["vmat"]
        vmat["TextureParams"] = {"g_tGlassTintColor": "tint.vtex", "g_tGlassDust": "dust.vtex"}
        vmat["VectorParams"] = {"g_flTranslucencyRemap": [0.07, 0.45, 0.0, 0.0]}
        layer = M.classify(0, material).layers[0]
        self.assertEqual(layer.color, "dust.vtex")
        self.assertEqual(layer.alpha_remap, (0.07, 0.45))

    def test_bake_color_remaps_opacity(self):
        from extractor.cs2render import textures as tx

        class Source:
            def load(self, name, size):
                image = np.ones((size, size, 4), dtype=np.float32)
                image[..., 3] = 0.5
                return image

        layer = M.LayerBake(kind="complex", color="dust.vtex", normal=None, alpha="blend", alpha_remap=(0.2, 0.6))
        np.testing.assert_allclose(tx.bake_color(layer, Source(), 2)[..., 3], 0.4, atol=1e-6)


class EntityLayoutTests(unittest.TestCase):
    def _world(self, folder: Path, name: str):
        import json
        from extractor.cs2render.gltf_world import GltfWorld, _SOURCE_FROM_GLTF, INCHES_PER_METER

        path = _write_world(folder, [LightmapUvTests._UV])
        document = json.loads(path.read_text(encoding="utf-8"))
        document["nodes"] = [{"name": f"{name}.mesh", "mesh": 0, "translation": [1.0, 2.0, 3.0]}]
        path.write_text(json.dumps(document), encoding="utf-8")
        origin = _SOURCE_FROM_GLTF @ np.array([1.0, 2.0, 3.0]) * INCHES_PER_METER
        return GltfWorld(path), " ".join(str(v) for v in origin)

    def test_mode_blockers_are_dropped_by_model_and_origin(self):
        import tempfile
        from extractor.cs2render.objects import layout

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            world, origin = self._world(Path(folder), "barricade")
            prop = {"classname": "prop_dynamic", "model": "models/barricade.vmdl", "origin": origin}
            blocker = {**prop, "targetname": "[PR#]props.2v2", "StartDisabled": True}
            self.assertEqual(layout(world, [blocker]).hidden_nodes, {0})
            self.assertEqual(layout(world, [{**prop, "StartDisabled": False}]).hidden_nodes, set())
            self.assertEqual(layout(world, [{**blocker, "origin": "0 0 0"}]).hidden_nodes, set())
            del world

    def test_doors_and_breakables_become_objects(self):
        import tempfile
        from extractor.cs2render.objects import layout

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            world, origin = self._world(Path(folder), "shutter_window_l_remainder")
            remainder = {
                "classname": "prop_dynamic", "model": "models/shutter_window_l_remainder.vmdl", "origin": origin,
                "angles": "0 90 0", "targetname": "[PR#]shutter.l.remainder", "StartDisabled": True,
            }
            result = layout(world, [remainder])
            self.assertEqual(result.hidden_nodes, set())
            self.assertEqual(result.node_objects, {0: 1})
            (obj,) = result.objects
            self.assertEqual((obj["kind"], obj["hidden"], obj["angles"]), ("breakable", True, [0.0, 90.0, 0.0]))
            lantern = {**remainder, "model": "models/lantern_01_glass.vmdl", "StartDisabled": False}
            self.assertEqual(layout(world, [lantern]).objects, [])
            door = {**remainder, "classname": "prop_door_rotating", "StartDisabled": False}
            self.assertEqual(layout(world, [door]).objects[0]["kind"], "door")
            del world


class BindPoseTests(unittest.TestCase):
    def _model(self, folder: Path, reference_rotation: list[float]) -> Path:
        import json

        # One bone bound at -90 degrees about glTF y, as Nuke's metal_door_001_br.
        half = np.sqrt(0.5)
        bind = np.array([[0.0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        inverse_bind = np.linalg.inv(bind).T.astype(np.float32).tobytes()
        joints = np.zeros((3, 4), dtype=np.uint8).tobytes()
        weights = np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (3, 1)).tobytes()
        times = np.zeros(1, dtype=np.float32).tobytes()
        rotation = np.array(reference_rotation, dtype=np.float32).tobytes()
        blob = inverse_bind + joints + weights + times + rotation
        offsets = np.cumsum([0, 64, 12, 48, 4])
        (folder / "door.bin").write_bytes(blob)
        views = [{"buffer": 0, "byteOffset": int(o), "byteLength": n} for o, n in zip(offsets, (64, 12, 48, 4, 16))]
        document = {
            "scenes": [{"nodes": [0]}],
            "nodes": [
                {"name": "models\\door.vmdl_c", "children": [1]},
                {"name": "door_reference", "rotation": [0.0, -half, 0.0, half]},
                {"name": "models\\door.vmdl_c.door_body", "mesh": 0, "skin": 0},
            ],
            "skins": [{"joints": [1], "inverseBindMatrices": 0}],
            "meshes": [{"primitives": [{"attributes": {"JOINTS_0": 1, "WEIGHTS_0": 2}}]}],
            "animations": [
                {
                    "name": "ref",
                    "channels": [{"sampler": 0, "target": {"node": 1, "path": "rotation"}}],
                    "samplers": [{"input": 3, "output": 4}],
                }
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 1, "type": "MAT4"},
                {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
                {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
                {"bufferView": 3, "componentType": 5126, "count": 1, "type": "SCALAR"},
                {"bufferView": 4, "componentType": 5126, "count": 1, "type": "VEC4"},
            ],
            "bufferViews": views,
            "buffers": [{"uri": "door.bin", "byteLength": len(blob)}],
        }
        path = folder / "door.gltf"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_reference_pose_differing_from_bind_pose_is_a_correction(self):
        import tempfile
        from extractor.cs2render.bindpose import corrections_from_gltf, mesh_key

        self.assertEqual(mesh_key("models\\props\\Door.vmdl_c.door_body"), "door.door_body")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            corrections = corrections_from_gltf(self._model(Path(folder), [0.0, 0.0, 0.0, 1.0]))
            self.assertEqual(list(corrections), ["door.door_body"])
            # Inverse bind: the door's glTF +z (Source +x) swings to glTF +x (Source +y).
            np.testing.assert_allclose(corrections["door.door_body"] @ [0, 0, 1, 0], [1, 0, 0, 0], atol=1e-6)
            half = float(np.sqrt(0.5))
            self.assertEqual(corrections_from_gltf(self._model(Path(folder), [0.0, -half, 0.0, half])), {})
