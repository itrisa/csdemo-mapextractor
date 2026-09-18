"""Release manifest generation."""

from __future__ import annotations

import hashlib
from typing import Any

from .geometry import CollisionGeometry


def create(
    geometry: CollisionGeometry,
    artifact: bytes,
    *,
    map_name: str | None,
    selection_profile: str,
    output_format: str,
    compression: str | None = None,
) -> dict[str, Any]:
    if geometry.positions:
        minimum = [min(position[axis] for position in geometry.positions) for axis in range(3)]
        maximum = [max(position[axis] for position in geometry.positions) for axis in range(3)]
    else:
        minimum = maximum = [0.0, 0.0, 0.0]
    return {
        "format": output_format,
        "compression": compression,
        "position_encoding": "float32",
        "coordinate_system": "source2-hammer-z-up",
        "units": "hammer",
        "transform_applied": False,
        "selection_profile": selection_profile,
        "map": map_name,
        "vertex_count": len(geometry.positions),
        "triangle_count": len(geometry.triangles),
        "bounds": {"min": minimum, "max": maximum},
        "sha256": hashlib.sha256(artifact).hexdigest(),
    }