"""Exact differential comparison of collision geometry."""

from __future__ import annotations

from collections import Counter
import struct
from typing import Any

from .geometry import CollisionGeometry, Triangle


def compare(first: CollisionGeometry, second: CollisionGeometry) -> dict[str, Any]:
    first_triangles = Counter(_triangle_key(triangle) for triangle in first.triangle_soup())
    second_triangles = Counter(_triangle_key(triangle) for triangle in second.triangle_soup())
    shared = sum((first_triangles & second_triangles).values())
    first_count = sum(first_triangles.values())
    second_count = sum(second_triangles.values())
    return {
        "first_triangle_count": first_count,
        "second_triangle_count": second_count,
        "shared_triangle_count": shared,
        "first_only_triangle_count": first_count - shared,
        "second_only_triangle_count": second_count - shared,
        "exact_multiset_match": first_triangles == second_triangles,
    }


def _triangle_key(triangle: Triangle) -> bytes:
    vertices = tuple(struct.pack("<3f", *position) for position in triangle)
    rotations = (
        vertices,
        (vertices[1], vertices[2], vertices[0]),
        (vertices[2], vertices[0], vertices[1]),
    )
    return b"".join(min(rotations))