import math
import struct
import unittest

from csdemo_mapextractor.geometry import CollisionGeometry
from csdemo_mapextractor.tri import dumps, loads


class TriTests(unittest.TestCase):
    def test_round_trip_welds_bit_identical_positions(self) -> None:
        raw = struct.pack(
            "<18f",
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            -1.0, 0.0, 0.0,
        )

        geometry = loads(raw)

        self.assertEqual(4, len(geometry.positions))
        self.assertEqual(((0, 1, 2), (0, 2, 3)), geometry.triangles)
        self.assertEqual(raw, dumps(geometry))

    def test_rejects_partial_triangle(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple of 36"):
            loads(b"\x00" * 35)

    def test_rejects_non_finite_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            CollisionGeometry((((math.inf, 0.0, 0.0)),), ())

    def test_rejects_out_of_range_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            CollisionGeometry(((0.0, 0.0, 0.0),), ((0, 0, 1),))


if __name__ == "__main__":
    unittest.main()