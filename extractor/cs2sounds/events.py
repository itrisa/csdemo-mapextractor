"""CS2 soundevents (``.vsndevts``): base inheritance, child events and the compact parameters the web client plays.

Only what a demo viewer can reproduce is kept: the variant files, volume and pitch with their additive random
ranges, the distance and fade-in curves, instance limits and blocking, the position offset (or absolute position
for map ambience), delays, retriggering and child events. Mixing, reverb, occlusion and stereo-width settings
are dropped.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from ..cs2render import kv3


def load(texts: Iterable[str]) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    for text in texts:
        events.update({name: value for name, value in kv3.loads(text).items() if isinstance(value, dict)})
    return events


def resolve(events: dict[str, dict[str, Any]], name: str, _seen: frozenset[str] = frozenset()) -> dict[str, Any] | None:
    """The event's parameters with its `base` chain applied (own values win)."""
    own = events.get(name)
    if own is None or name in _seen:
        return None
    base = own.get("base")
    merged = dict(resolve(events, base, _seen | {name}) or {}) if base else {}
    merged.update({key: value for key, value in own.items() if key != "base"})
    return merged


def children(params: dict[str, Any]) -> list[str]:
    if not params.get("enable_child_events", False):
        return []
    value = params.get("soundevent_01")
    names = [value] if isinstance(value, str) else list(value or [])
    return [name for name in names if name]


def files(params: dict[str, Any]) -> list[str]:
    value = params.get("vsnd_files_track_01") or params.get("vsnd_files") or []
    names = [value] if isinstance(value, str) else list(value)
    return [name for name in names if name]


def closure(events: dict[str, dict[str, Any]], roots: Iterable[str]) -> list[str]:
    """Roots plus every child event they start, in discovery order; unknown names are dropped."""
    seen: dict[str, None] = {}
    stack = list(roots)
    while stack:
        name = stack.pop(0)
        if name in seen:
            continue
        params = resolve(events, name)
        if params is None:
            continue
        seen[name] = None
        stack.extend(children(params))
    return list(seen)


def _hermite(keys: list[list[float]], x: float) -> float:
    if x <= keys[0][0]:
        return keys[0][1]
    if x >= keys[-1][0]:
        return keys[-1][1]
    for k0, k1 in zip(keys, keys[1:]):
        if x <= k1[0]:
            dx = k1[0] - k0[0]
            if dx <= 1e-6:
                return k1[1]
            t = (x - k0[0]) / dx
            t2, t3 = t * t, t * t * t
            return (
                (2 * t3 - 3 * t2 + 1) * k0[1]
                + (t3 - 2 * t2 + t) * dx * k0[3]
                + (-2 * t3 + 3 * t2) * k1[1]
                + (t3 - t2) * dx * k1[2]
            )
    return keys[-1][1]


def _simplify(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    (x0, y0), (x1, y1) = points[0], points[-1]
    worst, index = -1.0, 0
    for i in range(1, len(points) - 1):
        x, y = points[i]
        expected = y0 + (y1 - y0) * ((x - x0) / (x1 - x0) if x1 != x0 else 0.0)
        error = abs(y - expected)
        if error > worst:
            worst, index = error, i
    if worst <= tolerance:
        return [points[0], points[-1]]
    return _simplify(points[: index + 1], tolerance)[:-1] + _simplify(points[index:], tolerance)


def curve(keys: Any, tolerance: float = 0.01) -> list[float] | None:
    """A sound curve (`[x, y, slopeIn, slopeOut, modeIn, modeOut]` keys, Hermite between keys) as flattened
    piecewise-linear `[x0, y0, x1, y1, …]`, clamped to the keys' value range; constant outside the ends."""
    if not keys or not isinstance(keys, list):
        return None
    keys = sorted((list(map(float, key)) + [0.0] * 6)[:6] for key in keys)
    low, high = min(0.0, *(key[1] for key in keys)), max(key[1] for key in keys)
    points: list[tuple[float, float]] = []
    for k0, k1 in zip(keys, keys[1:]):
        for step in range(12):
            x = k0[0] + (k1[0] - k0[0]) * step / 12
            points.append((x, min(high, max(low, _hermite(keys, x)))))
    points.append((keys[-1][0], keys[-1][1]))
    simplified = _simplify(points, tolerance)
    return [round(value, 4 if abs(value) < 10 else 1) for point in simplified for value in point]


def _number(params: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = params.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def compact(params: dict[str, Any], sound_ids: dict[str, int], loops: dict[str, bool]) -> dict[str, Any]:
    """The client's view of one resolved event. `sound_ids` maps vsnd paths to sound-table indices."""
    out: dict[str, Any] = {}
    present = [name for name in files(params) if name in sound_ids]
    if present:
        out["f"] = [sound_ids[name] for name in present]
        if any(loops.get(name, False) for name in present):
            out["loop"] = 1
    volume = _number(params, "volume", 1.0)
    if volume != 1.0:
        out["v"] = round(volume, 3)
    volume_range = (_number(params, "volume_random_min"), _number(params, "volume_random_max"))
    if volume_range != (0.0, 0.0):
        out["vr"] = [round(value, 3) for value in volume_range]
    pitch = _number(params, "pitch", 1.0)
    if pitch != 1.0:
        out["p"] = round(pitch, 3)
    pitch_range = (_number(params, "pitch_random_min"), _number(params, "pitch_random_max"))
    if pitch_range != (0.0, 0.0):
        out["pr"] = [round(value, 3) for value in pitch_range]
    if params.get("use_distance_volume_mapping_curve", True):
        distance = curve(params.get("distance_volume_mapping_curve"))
        if distance:
            out["d"] = distance
    if params.get("use_time_volume_mapping_curve"):
        fade = curve(params.get("time_volume_mapping_curve"))
        if fade:
            out["t"] = fade
    limit = int(_number(params, "instance_limit"))
    if limit > 0:
        out["lim"] = limit
    block = _number(params, "block_duration")
    if block > 0:
        out["bd"] = round(block, 3)
        if params.get("block_match_entity"):
            out["be"] = 1
    offset = params.get("position_offset")
    if isinstance(offset, list) and len(offset) == 3 and offset[2]:
        out["oz"] = round(float(offset[2]), 1)
    position = params.get("position")
    if isinstance(position, list) and len(position) == 3:
        out["pos"] = [round(float(value), 1) for value in position]
    delay = _number(params, "delay")
    if delay > 0:
        out["dl"] = round(delay, 3)
    if params.get("randomize_start_time"):
        out["rs"] = 1
    if params.get("enable_retrigger"):
        out["rt"] = [round(_number(params, "retrigger_interval_min"), 2), round(_number(params, "retrigger_interval_max"), 2)]
    kids = children(params)
    if kids:
        out["c"] = kids
    return out
