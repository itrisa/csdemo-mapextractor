"""Reader and writer for Awpy's headerless legacy triangle stream."""

from __future__ import annotations

from pathlib import Path
import struct

from .geometry import CollisionGeometry, Triangle


TRIANGLE = struct.Struct("<9f")


def loads(data: bytes) -> CollisionGeometry:
    if len(data) % TRIANGLE.size:
        raise ValueError("TRI data length must be a multiple of 36 bytes")

    triangles: list[Triangle] = []
    for values in TRIANGLE.iter_unpack(data):
        triangles.append(
            (
                (values[0], values[1], values[2]),
                (values[3], values[4], values[5]),
                (values[6], values[7], values[8]),
            )
        )
    return CollisionGeometry.from_triangle_soup(triangles)


def dumps(geometry: CollisionGeometry) -> bytes:
    output = bytearray(len(geometry.triangles) * TRIANGLE.size)
    offset = 0
    for triangle in geometry.triangle_soup():
        TRIANGLE.pack_into(output, offset, *(value for position in triangle for value in position))
        offset += TRIANGLE.size
    return bytes(output)


def read(path: Path) -> CollisionGeometry:
    return loads(path.read_bytes())


def write(path: Path, geometry: CollisionGeometry) -> None:
    path.write_bytes(dumps(geometry))