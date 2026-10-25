import gzip
import struct

import numpy as np

from extractor.cs2sounds import bake, events, surfaces


def test_resolve_applies_base_chain_and_closure_follows_children():
    table = {
        "Base": {"volume": 0.5, "pitch": 1.2},
        "Mid": {"base": "Base", "volume": 0.8},
        "Shot": {"base": "Mid", "vsnd_files_track_01": ["a.vsnd"], "enable_child_events": True, "soundevent_01": "Distant"},
        "Distant": {"vsnd_files_track_01": "b.vsnd"},
        "Loop": {"base": "Loop"},
    }
    resolved = events.resolve(table, "Shot")
    assert resolved["volume"] == 0.8 and resolved["pitch"] == 1.2 and "base" not in resolved
    assert events.closure(table, ["Shot", "Missing"]) == ["Shot", "Distant"]
    assert events.resolve(table, "Loop") == {}


def test_curve_is_piecewise_linear_and_simplified():
    keys = [[0, 1, 0, 0, 2, 2], [1000, 0, 0, 0, 2, 2]]
    flat = events.curve(keys)
    assert flat[:2] == [0.0, 1.0] and flat[-2:] == [1000.0, 0.0]
    xs, ys = flat[0::2], flat[1::2]
    assert xs == sorted(xs) and all(0.0 <= y <= 1.0 for y in ys)
    # A Hermite ease with flat ends is S-shaped: the midpoint is half volume.
    mid = float(np.interp(500, xs, ys))
    assert abs(mid - 0.5) < 0.1
    assert events.curve([[0, 1, 0, 0, 0, 0], [100, 1, 0, 0, 0, 0]]) == [0.0, 1.0, 100.0, 1.0]


def test_compact_keeps_playable_parameters():
    params = {
        "vsnd_files_track_01": ["a.vsnd", "missing.vsnd", "b.vsnd"],
        "volume": 0.7,
        "pitch_random_min": -0.05,
        "pitch_random_max": 0.05,
        "instance_limit": 2,
        "position_offset": [0, 0, 60],
        "enable_retrigger": True,
        "retrigger_interval_min": 3,
        "retrigger_interval_max": 9,
        "use_distance_volume_mapping_curve": False,
        "distance_volume_mapping_curve": [[0, 1, 0, 0, 2, 2], [100, 0, 0, 0, 2, 2]],
    }
    out = events.compact(params, {"a.vsnd": 4, "b.vsnd": -1}, {"b.vsnd": True})
    assert out == {"f": [4, -1], "loop": 1, "v": 0.7, "pr": [-0.05, 0.05], "lim": 2, "oz": 60.0, "rt": [3.0, 9.0]}


def _dump(vertices: np.ndarray, triangles: np.ndarray, materials: list[int]) -> list[str]:
    hexed = lambda data: " ".join(f"{b:02X}" for b in data)  # noqa: E731
    return [
        "{",
        "m_nCollisionAttributeIndex = 1",
        "m_nSurfacePropertyIndex = 0",
        "m_Mesh = ",
        "{",
        "m_vMin = [ 0.0, 0.0, 0.0 ]",
        "m_Materials = ",
        "[",
        ", ".join(str(m) for m in materials) + ",",
        "]",
        "m_Triangles =",
        "#[",
        hexed(triangles.astype("<i4").tobytes()),
        "]",
        "m_Vertices = ",
        "#[",
        hexed(vertices.astype("<f4").tobytes()),
        "]",
        "}",
        "}",
        "m_surfacePropertyHashes = ",
        "[",
        "111, 222,",
        "]",
        "m_collisionAttributes = ",
        "[",
        "{",
        'm_CollisionGroupString = "Default"',
        "m_InteractAsStrings = [  ]",
        "m_InteractExcludeStrings = ",
        "[",
        '"player",',
        "]",
        "},",
        "{",
        'm_CollisionGroupString = "Default"',
        "m_InteractAsStrings = [  ]",
        "m_InteractExcludeStrings = [  ]",
        "},",
        "]",
    ]


def test_parse_keeps_up_facing_triangles_with_their_surfaces():
    vertices = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 1]])  # floor, wall, ceiling
    physics = surfaces.parse(_dump(vertices, triangles, [1, 0, 0]))
    assert physics.surface_hashes == [111, 222]
    assert surfaces.walkable_attributes(physics.attributes) == {1}
    assert len(physics.floors) == 1
    attribute, tris, mats = physics.floors[0]
    assert attribute == 1 and tris.shape == (1, 3, 3) and mats.tolist() == [1]


def test_grid_separates_stacked_floors_and_round_trips():
    ground = np.array([[x, y, 0.0] for x in (4.0, 20.0, 40.0) for y in (4.0, 20.0)])
    bridge = np.array([[4.0, 4.0, 128.0], [20.0, 20.0, 130.0]])
    grid = surfaces.bake_grid(np.concatenate([ground, bridge]), np.array([0] * len(ground) + [1, 1]))
    assert grid.lookup(10, 10, 0) == 0
    assert grid.lookup(10, 10, 140) == 1
    assert grid.lookup(10, 10, 60) == 0
    assert grid.lookup(-500, 10, 0) is None
    raw = gzip.decompress(grid.encode())
    magic, x0, y0, cell, _, width, height, layers = struct.unpack_from("<4sffffHHI", raw)
    assert magic == b"SGR1" and cell == surfaces.CELL and (width, height) == (grid.width, grid.height)
    header = struct.calcsize("<4sffffHHI")
    assert len(raw) == header + width * height + layers * 3


def test_footstep_surface_walks_base_chain():
    table = {"default": {"ct": "a", "t": "b"}, "wood": {"ct": "c", "t": "d"}}
    bases = {"wood_crate": "Wood", "odd": "missing"}
    assert surfaces.footstep_surface("Wood", table, bases) == "wood"
    assert surfaces.footstep_surface("wood_crate", table, bases) == "wood"
    assert surfaces.footstep_surface("odd", table, bases) == "default"


def test_clip_foley_reads_sound_events_as_fractions():
    text = """
        _class = "CNmIDEvent"
        m_flStartTime =
        {
            m_flValue = 0.1
        }
        m_ID = "x"
        _class = "CNmSoundEvent"
        m_flStartTime =
        {
            m_flValue = 0.4521
        }
        m_flDuration =
        {
            m_flValue = 0.0
        }
        m_name = "Weapon_AK47.Clipin"
        _class = "CNmSoundEvent"
        m_flStartTime =
        {
            m_flValue = 0.15
        }
        m_name = "Weapon_AK47.Clipout"
    """
    assert bake._clip_foley(text) == [(0.15, "Weapon_AK47.Clipout"), (0.4521, "Weapon_AK47.Clipin")]
    assert bake.FOLEY_SKIP.search("Weapon_AK47.Single") and not bake.FOLEY_SKIP.search("Weapon_AK47.Clipin")
