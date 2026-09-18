"""Minimal entity state parser for Source2Viewer DATA text."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class EntityState:
    classname: str
    origin: tuple[float, float, float]
    disabled: bool


_VALUES_BLOCK = re.compile(r"\bvalues\s*=\s*\{(.*?)\n\s*\}", re.DOTALL)


def parse_states(text: str, classname: str) -> tuple[EntityState, ...]:
    states = []
    for block in _VALUES_BLOCK.findall(text):
        parsed_classname = _string_property(block, "classname")
        if parsed_classname != classname:
            continue
        origin = _vector_property(block, "origin")
        if origin is None:
            continue
        start_disabled = _boolean_property(block, "startdisabled", False)
        enabled = _boolean_property(block, "enabled", True)
        states.append(EntityState(classname, origin, start_disabled or not enabled))
    return tuple(states)


def _string_property(block: str, key: str) -> str | None:
    match = re.search(rf'^\s*{re.escape(key)}\s*=\s*"([^"]*)"\s*$', block, re.MULTILINE | re.IGNORECASE)
    return match.group(1) if match else None


def _boolean_property(block: str, key: str, default: bool) -> bool:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*(true|false)\s*$", block, re.MULTILINE | re.IGNORECASE)
    return match.group(1).lower() == "true" if match else default


def _vector_property(block: str, key: str) -> tuple[float, float, float] | None:
    match = re.search(
        rf"^\s*{re.escape(key)}\s*=\s*\[\s*([^,]+),\s*([^,]+),\s*([^\]]+)\]\s*$",
        block,
        re.MULTILINE | re.IGNORECASE,
    )
    return tuple(float(value) for value in match.groups()) if match else None