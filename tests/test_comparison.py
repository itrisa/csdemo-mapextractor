import unittest

from csdemo_mapextractor.comparison import compare
from csdemo_mapextractor.geometry import CollisionGeometry


class ComparisonTests(unittest.TestCase):
    def test_cyclic_vertex_rotation_is_the_same_triangle(self) -> None:
        positions = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        first = CollisionGeometry(positions, ((0, 1, 2),))
        second = CollisionGeometry(positions, ((1, 2, 0),))

        result = compare(first, second)

        self.assertTrue(result["exact_multiset_match"])
        self.assertEqual(1, result["shared_triangle_count"])

    def test_reversed_winding_is_reported_as_different(self) -> None:
        positions = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        first = CollisionGeometry(positions, ((0, 1, 2),))
        second = CollisionGeometry(positions, ((0, 2, 1),))

        result = compare(first, second)

        self.assertFalse(result["exact_multiset_match"])
        self.assertEqual(0, result["shared_triangle_count"])


if __name__ == "__main__":
    unittest.main()