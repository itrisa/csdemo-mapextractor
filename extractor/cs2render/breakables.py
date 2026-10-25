"""
What a breakable leaves behind when it breaks, for the client to show in place of the plain disappearance the demo
records (it only networks the entity's deletion; CS2 spawns the debris client-side).

- Break pieces: a model's `prop_data` `BreakPieceList` names the models it breaks into. Pieces that the game spawns
  frozen (`spawn_motion_disabled`) are remainders that stay in the frame (the jagged glass left in a window); the
  rest is debris that falls. All of an object model's pieces are baked into one small GLB (`breaks/<model>.glb`,
  per-vertex PBR like the dropped weapons, Source inches in the object's frame) and the object gets
  `"breaks": {"file", "glass"}`. Tintable materials (csgo_complex `F_TINT_MASK`) keep their tint mask in the vertex
  colour's alpha and the object gets its linear `rendercolor` as `"tint"` (orange vs yellow Nuke doors share a model).
- Glass: windows break into shards as well (CS2 draws them as particles), so glass objects are flagged even without
  pieces (brush glass, `func_breakable` material 0).
- Overlays: Nuke's additive window brushes (`func_brush`) aren't networked with their window; they get
  `"follows": <object id>` of the window prop they sit on and hide with it.
"""

from __future__ import annotations

import json
import re
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

BAKE_DIR = Path(__file__).resolve().parents[2] / "weapon-bake"
MATCH_TOLERANCE = 2.0
FOLLOW_DISTANCE = 8.0
# Models whose own piece list spawns nothing (`null.vmdl`) but that ship damage pieces: Nuke's breakable metal door
# bursts into its last damage stage.
PIECE_OVERRIDES: dict[str, list[dict[str, Any]]] = {
    "models/props/de_nuke/hr_nuke/metal_door_001/metal_door_001_br.vmdl": [
        {"model": f"models/props/de_nuke/hr_nuke/metal_door_001/metal_door_001_br_dm05_{n:02d}.vmdl", "essential": False}
        for n in (1, 2, 3, 4, 6, 7)
    ],
}
_PIECE = re.compile(r'_class = "BreakPieceExternal"(.*?)(?=_class = )', re.S)


def _log(message: str) -> None:
    print(f"[breakables] {message}", flush=True)


def _vector(value: Any) -> np.ndarray | None:
    try:
        parts = value.split() if isinstance(value, str) else list(value)
        return np.array([float(part) for part in parts[:3]], dtype=np.float64) if len(parts) >= 3 else None
    except (TypeError, ValueError):
        return None


def _field(block: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)} = (.*)$", block, re.M)
    return match[1].strip() if match else None


def _decompile(executable: Path, vpk: Path, models: list[str], out: Path) -> None:
    missing = [m for m in models if not (out / m).is_file()]
    if missing:
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(executable), "-i", str(vpk), "-d", "-f", ",".join(m + "_c" for m in missing), "-o", str(out)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _export(executable: Path, vpk: Path, models: list[str], out: Path) -> None:
    missing = [m for m in models if not (out / Path(m).with_suffix(".glb")).is_file()]
    if missing:
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(executable), "-i", str(vpk), "-d", "-f", ",".join(m + "_c" for m in missing), "-o", str(out),
             "--gltf_export_format", "glb", "--gltf_export_materials", "--gltf_textures_adapt"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _glb_materials(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20:20 + length]).get("materials", [])


def _tint_masks(executable: Path, vpk: Path, exports: list[Path], out: Path) -> dict[str, dict[str, Any]]:
    """Tint masks of the pieces' tintable materials (csgo_complex `F_TINT_MASK`), by material, decompiled to PNG."""
    wanted: dict[str, tuple[str, float]] = {}
    for path in exports:
        for material in _glb_materials(path):
            vmat = material.get("extras", {}).get("vmat", {})
            mask = vmat.get("TextureParams", {}).get("g_tTintMask")
            if mask and vmat.get("IntParams", {}).get("F_TINT_MASK"):
                wanted[vmat["Name"]] = (mask, float(vmat.get("FloatParams", {}).get("g_flModelTintAmount", 1.0)))
    _decompile(executable, vpk, sorted({m for m, _ in wanted.values() if not (out / m).with_suffix(".png").is_file()}), out)
    masks = {}
    for name, (mask, amount) in wanted.items():
        png = (out / mask).with_suffix(".png")
        if png.is_file():
            masks[name] = {"file": str(png.resolve()), "amount": amount}
        else:
            _log(f"could not decompile {mask}")
    return masks


def _linear_color(value: Any) -> list[float] | None:
    """An entity's `rendercolor` as a linear tint, or None when it's white."""
    rgb = _vector(value)
    if rgb is None or np.all(rgb >= 255):
        return None
    srgb = np.clip(rgb / 255.0, 0.0, 1.0)
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return [round(float(x), 4) for x in linear]


def _floats(value: str | None) -> list[float]:
    return [float(x) for x in re.findall(r"-?[\d.]+(?:e-?\d+)?", value or "")][:3] or [0.0, 0.0, 0.0]


def _pieces(text: str) -> list[dict[str, Any]]:
    pieces = []
    for block in _PIECE.findall(text + '\n_class = ""'):
        model = _field(block, "model")
        if model:
            model = model.replace("resource:", "").strip('"')
            if not model.endswith("null.vmdl"):
                pieces.append({
                    "model": model,
                    "offset": _floats(_field(block, "offset")),
                    "angles": _floats(_field(block, "offset_rotation")),
                    "essential": (_field(block, "is_essential_piece") or "").lower() == "true",
                })
    return pieces


def _is_glass(text: str) -> bool:
    base = _field(text, "base") or ""
    return base.strip('"').lower().startswith("glass") or 'surface_prop = "glass"' in text


def _frozen(text: str) -> bool:
    return (_field(text, "spawn_motion_disabled") or "").lower() == "true"


def _placement(text: str) -> dict[str, list[float]] | None:
    """A generated piece's `placementOrigin` attachment: the parent model's frame, in the re-centred piece's."""
    match = re.search(r'name = "placementOrigin"(.*?)\}', text, re.S)
    if not match:
        return None
    return {"origin": _floats(_field(match[1], "relative_origin")), "angles": _floats(_field(match[1], "relative_angles"))}


def _object_entity(obj: dict[str, Any], entities: list[dict[str, Any]]) -> dict[str, Any] | None:
    origin = np.array(obj["origin"], dtype=np.float64)
    best, best_distance = None, MATCH_TOLERANCE
    for entity in entities:
        model = entity.get("model")
        if not model or Path(str(model)).stem != obj["model"] or entity.get("classname") != obj["classname"]:
            continue
        position = _vector(entity.get("origin", "0 0 0"))
        if position is None:
            continue
        distance = float(np.linalg.norm(position - origin))
        if distance < best_distance:
            best, best_distance = entity, distance
    return best


def attach(
    executable: Path, vpk: Path, objects: list[dict[str, Any]], entities: list[dict[str, Any]], work_dir: Path, out_dir: Path
) -> None:
    """Adds `breaks` / `follows` to the manifest objects and writes `out_dir/breaks/*.glb`."""
    for obj in objects:
        obj.pop("breaks", None)
        obj.pop("follows", None)
    vmdl_dir = work_dir / "breaks" / "vmdl"
    export_dir = work_dir / "breaks" / "export"
    targets: dict[int, str] = {}
    tints: dict[int, list[float]] = {}
    for obj in objects:
        if obj["classname"] == "func_brush" and "additive" in obj["model"]:
            continue
        entity = _object_entity(obj, entities)
        if entity is None:
            continue
        if obj["classname"] == "func_breakable":
            if str(entity.get("material", 0)) in ("0", "Glass"):
                obj["breaks"] = {"glass": True}
            continue
        targets[obj["id"]] = str(entity["model"])
        tint = _linear_color(entity.get("rendercolor"))
        if tint is not None:
            tints[obj["id"]] = tint

    models = sorted(set(targets.values()))
    _decompile(executable, vpk, models, vmdl_dir)
    pieces: dict[str, list[str]] = {}
    glass: dict[str, bool] = {}
    for model in models:
        path = vmdl_dir / model
        if not path.is_file():
            _log(f"could not decompile {model}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        pieces[model] = PIECE_OVERRIDES.get(model, _pieces(text))
        glass[model] = _is_glass(text)

    piece_models = sorted({p["model"] for list_ in pieces.values() for p in list_})
    _decompile(executable, vpk, piece_models, vmdl_dir)
    _export(executable, vpk, piece_models, export_dir)
    exports = [export_dir / Path(m).with_suffix(".glb") for m in piece_models]
    masks = _tint_masks(executable, vpk, [p for p in exports if p.is_file()], work_dir / "breaks" / "vtex")

    breaks_dir = out_dir / "breaks"
    if breaks_dir.exists():
        for old in breaks_dir.glob("*.glb"):
            old.unlink()
    files: dict[str, str | None] = {}
    tintable: dict[str, bool] = {}
    for model, list_ in pieces.items():
        specs = []
        for piece in list_:
            exported = export_dir / Path(piece["model"]).with_suffix(".glb")
            if not exported.is_file():
                _log(f"  {piece['model']}: export failed, skipped")
                continue
            vmdl = vmdl_dir / piece["model"]
            text = vmdl.read_text(encoding="utf-8", errors="replace") if vmdl.is_file() else ""
            piece_glass = 'surface_prop = "glass"' in text
            specs.append({
                "file": str(exported.resolve()),
                # The remainder left in a window frame: frozen in place, or the essential glass piece.
                "static": _frozen(text) or (piece_glass and piece["essential"]),
                "glass": piece_glass,
                "offset": piece.get("offset", [0.0, 0.0, 0.0]),
                "angles": piece.get("angles", [0.0, 0.0, 0.0]),
                "placement": _placement(text),
            })
        if not specs:
            files[model] = None
            continue
        name = Path(model).stem
        breaks_dir.mkdir(parents=True, exist_ok=True)
        spec_path = work_dir / "breaks" / f"{name}.json"
        used = {m.get("extras", {}).get("vmat", {}).get("Name") for s in specs for m in _glb_materials(Path(s["file"]))}
        tint_masks = {k: v for k, v in masks.items() if k in used}
        tintable[model] = bool(tint_masks)
        spec_path.write_text(json.dumps({"glass": glass[model], "pieces": specs, "tintMasks": tint_masks}), encoding="utf-8")
        _node("bake-break-pieces.mjs", [str(spec_path.resolve()), str((breaks_dir / f"{name}.glb").resolve())])
        files[model] = f"breaks/{name}.glb"

    for obj in objects:
        model = targets.get(obj["id"])
        if model is None:
            continue
        entry: dict[str, Any] = {}
        if files.get(model):
            entry["file"] = files[model]
            if tintable.get(model) and obj["id"] in tints:
                entry["tint"] = tints[obj["id"]]
        if glass.get(model):
            entry["glass"] = True
        if entry:
            obj["breaks"] = entry

    props = [o for o in objects if o["kind"] == "breakable" and o["classname"] != "func_brush"]
    for obj in objects:
        if obj["classname"] != "func_brush" or "additive" not in obj["model"] or not props:
            continue
        origin = np.array(obj["origin"])
        nearest = min(props, key=lambda o: float(np.linalg.norm(np.array(o["origin"]) - origin)))
        if float(np.linalg.norm(np.array(nearest["origin"]) - origin)) <= FOLLOW_DISTANCE:
            obj["follows"] = nearest["id"]

    with_breaks = [o for o in objects if "breaks" in o]
    total = sum(p.stat().st_size for p in breaks_dir.glob("*.glb")) if breaks_dir.exists() else 0
    _log(
        f"{len(with_breaks)} breakable objects ({sum('file' in o['breaks'] for o in with_breaks)} with pieces, "
        f"{sum(bool(o['breaks'].get('glass')) for o in with_breaks)} glass), "
        f"{sum('follows' in o for o in objects)} overlays, {total // 1024} KB of pieces"
    )


def _node(script: str, args: list[str]) -> None:
    completed = subprocess.run(
        ["node", str(BAKE_DIR / script), *args], cwd=BAKE_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    for line in (completed.stdout + completed.stderr).splitlines():
        if line.strip():
            print(f"  {line}", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{script} failed: {' '.join(args)}")
