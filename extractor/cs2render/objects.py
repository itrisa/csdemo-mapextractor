"""
Map entities whose look changes during a match, and entities that never show in a competitive round.

- Dynamic objects: doors (`prop_door_rotating`) and breakables (`func_breakable`, and `prop_dynamic`/`func_brush`
  windows, glass, vents and shutters). Their geometry goes into chunks of their own (chunk key `object`, 1-based) so
  the client can rotate doors and hide broken glass from the demo's entity state. The manifest lists them under
  `objects`, keyed by placement origin for matching against the demo's networked entities.
- Mode blockers: start-disabled wingman/retake brushes and props (`[PR#]brush.blocker`, `[PR#]props.2v2`, retake and
  nav blockers). They are dropped. Other start-disabled breakable parts (Mirage's shutter remainders, shown once the
  shutter breaks) stay as objects that start hidden.

glTF nodes are matched to entities by model name and placement origin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .gltf_world import GltfWorld

ORIGIN_TOLERANCE = 1.0
_MODE_BLOCKER = re.compile(r"blocker|2v2|retake|tag\.hop", re.IGNORECASE)
_BREAKABLE = re.compile(r"window|glass|vent|shutter|breakable", re.IGNORECASE)
_DECOR = re.compile(r"lantern|lamp|light", re.IGNORECASE)
_BREAKABLE_CLASSES = {"prop_dynamic", "prop_dynamic_override", "func_brush"}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true")


def _vector(value: Any) -> np.ndarray | None:
    try:
        parts = value.split() if isinstance(value, str) else list(value)
        return np.array([float(part) for part in parts[:3]], dtype=np.float64) if len(parts) >= 3 else None
    except (TypeError, ValueError):
        return None


def _start_disabled(entity: dict[str, Any]) -> bool:
    return _truthy(entity.get("StartDisabled", entity.get("startdisabled", 0)))


def object_kind(entity: dict[str, Any]) -> str | None:
    """'door' or 'breakable' for entities the demo can change, else None."""
    classname = str(entity.get("classname", ""))
    if classname == "prop_door_rotating":
        return "door"
    if classname == "func_breakable":
        return "breakable"
    if classname in _BREAKABLE_CLASSES:
        names = f"{Path(str(entity.get('model', ''))).stem} {entity.get('targetname', '')}"
        if _BREAKABLE.search(names) and not _DECOR.search(names):
            return "breakable"
    return None


def is_mode_blocker(entity: dict[str, Any]) -> bool:
    return _start_disabled(entity) and bool(_MODE_BLOCKER.search(str(entity.get("targetname", ""))))


@dataclass
class EntityLayout:
    hidden_nodes: set[int] = field(default_factory=set)
    node_objects: dict[int, int] = field(default_factory=dict)  # glTF node -> object id (1-based)
    objects: list[dict[str, Any]] = field(default_factory=list)


def layout(world: GltfWorld, entities: list[dict[str, Any]], tolerance: float = ORIGIN_TOLERANCE) -> EntityLayout:
    """Which glTF nodes to drop and which belong to dynamic objects."""
    wanted: dict[str, list[tuple[np.ndarray, dict[str, Any]]]] = {}
    for entity in entities:
        model = entity.get("model")
        origin = _vector(entity.get("origin", "0 0 0"))
        if model and origin is not None and (_start_disabled(entity) or object_kind(entity)):
            wanted.setdefault(Path(str(model)).stem.lower(), []).append((origin, entity))

    result = EntityLayout()
    ids: dict[int, int] = {}
    for index, node in world.mesh_nodes():
        candidates = wanted.get(str(node.get("name", "")).split(".")[0].lower())
        if not candidates:
            continue
        node_origin = world.node_origin(index)
        distance, origin, entity = min(
            ((float(np.linalg.norm(node_origin - o)), o, e) for o, e in candidates), key=lambda item: item[0]
        )
        if distance >= tolerance:
            continue
        kind = object_kind(entity)
        if kind is None or is_mode_blocker(entity):
            if _start_disabled(entity):
                result.hidden_nodes.add(index)
            continue
        key = id(entity)
        if key not in ids:
            ids[key] = len(result.objects) + 1
            angles = _vector(entity.get("angles", "0 0 0"))
            result.objects.append(
                {
                    "id": ids[key],
                    "kind": kind,
                    "classname": str(entity.get("classname")),
                    "model": Path(str(entity.get("model"))).stem,
                    "origin": [round(float(x), 2) for x in origin],
                    "angles": [round(float(x), 2) for x in (angles if angles is not None else np.zeros(3))],
                    "hidden": _start_disabled(entity),
                }
            )
        result.node_objects[index] = ids[key]
    return result
