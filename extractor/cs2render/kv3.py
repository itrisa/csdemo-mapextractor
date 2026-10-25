"""Minimal KeyValues3 text reader for Source2Viewer `--block DATA` dumps."""

from __future__ import annotations

import re
from typing import Any

_TOKEN = re.compile(
    r"""
    (?P<ws>\s+|//[^\n]*|/\*.*?\*/|<!--.*?-->)
  | (?P<mstring>\"\"\"\n?(?P<mbody>.*?)\n?\"\"\")
  | (?P<string>"(?P<sbody>(?:[^"\\]|\\.)*)")
  | (?P<blob>\#\[(?P<bbody>[^\]]*)\])
  | (?P<punct>[{}\[\]=,])
  | (?P<flag>[A-Za-z_][A-Za-z0-9_]*:(?=["\[{]))
  | (?P<atom>[^\s{}\[\]=,"]+)
    """,
    re.VERBOSE | re.DOTALL,
)


def _tokens(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    position = 0
    length = len(text)
    while position < length:
        match = _TOKEN.match(text, position)
        if match is None:
            raise ValueError(f"KV3: unexpected character at {position}: {text[position:position + 40]!r}")
        position = match.end()
        kind = match.lastgroup
        if kind in ("ws", "flag"):
            continue
        if kind == "mstring":
            out.append(("str", match.group("mbody")))
        elif kind == "string":
            body = match.group("sbody")
            out.append(("str", body.replace('\\"', '"').replace("\\\\", "\\") if "\\" in body else body))
        elif kind == "blob":
            out.append(("blob", match.group("bbody")))
        elif kind == "punct":
            out.append(("p", match.group("punct")))
        else:
            out.append(("atom", match.group("atom")))
    return out


def _atom(value: str) -> Any:
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.index = 0

    def peek(self) -> tuple[str, str]:
        return self.tokens[self.index]

    def take(self) -> tuple[str, str]:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def value(self) -> Any:
        kind, text = self.take()
        if kind == "p" and text == "{":
            return self.object_body()
        if kind == "p" and text == "[":
            items: list[Any] = []
            while self.peek() != ("p", "]"):
                items.append(self.value())
                if self.peek() == ("p", ","):
                    self.take()
            self.take()
            return items
        if kind == "str":
            return text
        if kind == "blob":
            return bytes.fromhex("".join(text.split()))
        if kind == "atom":
            return _atom(text)
        raise ValueError(f"KV3: unexpected token {text!r}")

    def object_body(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        while self.peek() != ("p", "}"):
            kind, key = self.take()
            if kind not in ("atom", "str"):
                raise ValueError(f"KV3: expected key, got {key!r}")
            if self.take() != ("p", "="):
                raise ValueError(f"KV3: expected '=' after {key!r}")
            result[key] = self.value()
        self.take()
        return result


def loads(text: str) -> Any:
    """Parses the first KV3 root object in text; Source2Viewer prefixes dumps with resource headers."""
    start = text.find("<!-- kv3")
    if start < 0:
        raise ValueError("KV3: header not found")
    parser = _Parser(_tokens(text[start:]))
    if parser.take() != ("p", "{"):
        raise ValueError("KV3: root must be an object")
    return parser.object_body()


def entities(text: str) -> list[dict[str, Any]]:
    """Returns the key/value dictionaries of every entity in an entity-lump DATA dump."""
    root = loads(text)
    result: list[dict[str, Any]] = []
    for entity in root.get("m_entityKeyValues", []):
        values = entity.get("keyValues3Data", {}).get("values")
        if isinstance(values, dict):
            result.append(values)
    return result
