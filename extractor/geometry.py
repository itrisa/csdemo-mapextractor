"""Canonical collision geometry representation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable


Position = tuple[float, float, float]
Triangle = tuple[Position, Position, Position]
IndexTriangle = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class CollisionGeometry:
    """Indexed float32 positions in Source/Hammer coordinates."""

    positions: tuple[Position, ...]
    triangles: tuple[IndexTriangle, ...]

    def __post_init__(self) -> None:
        vertex_count = len(self.positions)
        for position in self.positions:
            if len(position) != 3 or not all(math.isfinite(value) for value in position):
                raise ValueError("positions must contain three finite coordinates")
        for triangle in self.triangles:
            if len(triangle) != 3 or any(index < 0 or index >= vertex_count for index in triangle):
                raise ValueError("triangle index is outside the position array")

    @classmethod
    def from_triangle_soup(cls, triangles: Iterable[Triangle]) -> CollisionGeometry:
        positions: list[Position] = []
        indexed_triangles: list[IndexTriangle] = []
        position_indices: dict[bytes, int] = {}

        for triangle in triangles:
            indices: list[int] = []
            for position in triangle:
                packed = struct.pack("<3f", *position)
                canonical = struct.unpack("<3f", packed)
                index = position_indices.get(packed)
                if index is None:
                    index = len(positions)
                    position_indices[packed] = index
                    positions.append(canonical)
                indices.append(index)
            indexed_triangles.append((indices[0], indices[1], indices[2]))

        return cls(tuple(positions), tuple(indexed_triangles))

    def triangle_soup(self) -> tuple[Triangle, ...]:
        return tuple(
            (
                self.positions[triangle[0]],
                self.positions[triangle[1]],
                self.positions[triangle[2]],
            )
            for triangle in self.triangles
        )