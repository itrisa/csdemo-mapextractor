"""Versioned collision geometry selection profiles."""

from __future__ import annotations


VISUAL_OCCLUDERS = "visual-occluders-v1"
ALL_PHYSICS = "all-physics-v1"
AWPY_DEFAULT = "awpy-default-v1"

_NON_OCCLUDER_MATERIALS = (
    "playerclip",
    "grenadeclip",
    "sky",
    "passbullets",
    "glass",
    "window",
)


def includes_mesh(profile: str, mesh_name: str | None, material_name: str | None) -> bool:
    if profile == ALL_PHYSICS:
        return True
    if profile == VISUAL_OCCLUDERS:
        name = (material_name or "").lower()
        return not any(fragment in name for fragment in _NON_OCCLUDER_MATERIALS)
    if profile == AWPY_DEFAULT:
        if mesh_name is None:
            raise ValueError("awpy-default-v1 requires VRF collision-group mesh names")
        name = mesh_name.lower()
        return name == "physics_group" or name.startswith("physics_group_")
    raise ValueError(f"unknown collision profile: {profile}")