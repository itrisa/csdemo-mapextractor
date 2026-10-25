"""Bake CS2's game audio for the web client: Opus banks, the soundevent parameters, first-person foley timings,
map ambience (soundscapes and point sounds) and a footstep-surface grid per map.

Output (``<output-dir>``):

- ``sounds.json``: banks, the sound table (`[bank, offset, length]` into a bank file), the events, the shot
  event of every weapon, the first-person foley of every viewmodel clip and the core event names the client
  triggers.
- ``<bank>.bin``: concatenated Ogg Opus files (48 kHz mono). `core` is always loaded; `weapon_<name>` holds a
  weapon's shot and first-person foley and is loaded only for weapons a demo uses.
- ``maps/<map>.json`` + ``maps/<map>.bin`` + ``maps/<map>_ambience.bin`` + ``maps/<map>_surfaces.bin``: the map's
  own sounds, its soundscapes and point sounds, and the gzip-compressed footstep-surface grid. ``<map>.bin``
  (footsteps, landings and impacts of the map's surfaces) loads with the map; ``<map>_ambience.bin`` (soundscape
  beds and music, at a lower bitrate) is read one sound at a time with HTTP Range requests when first needed.
  Map events refer to map sounds with negative ids (``-1 - index`` into the map's `[bank, offset, length]` list).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .. import cs2weapons, source2viewer
from ..cs2render import kv3
from . import events as ev
from . import surfaces

SOUNDEVENT_FILES = (
    "soundevents/game_sounds_weapons",
    "soundevents/game_sounds_footsteps",
    "soundevents/game_sounds_player",
    "soundevents/game_sounds_physics",
    "soundevents/game_sounds_world",
    "soundevents/vo/announcer/game_sounds_cs2_classic",
)
VSURF = "surfaceproperties/surfaceproperties.vsurf"
FOOTSTEPS = "scripts/surfaceproperties_footsteps.txt"

# Baked weapon name (as in cs2weapons) → the event its shot plays. M4A1-S and USP-S are assumed silenced.
SHOTS: dict[str, str] = {
    "ak47": "Weapon_AK47.Single",
    "m4a4": "Weapon_M4A4.Single",
    "m4a1_silencer": "Weapon_M4A1.Silenced",
    "awp": "Weapon_AWP.Single",
    "ssg08": "Weapon_SSG08.Single",
    "famas": "Weapon_FAMAS.Single",
    "galilar": "Weapon_GalilAR.Single",
    "aug": "Weapon_AUG.Single",
    "sg556": "Weapon_sg556.Single",
    "scar20": "Weapon_SCAR20.Single",
    "g3sg1": "Weapon_G3SG1.Single",
    "mac10": "Weapon_MAC10.Single",
    "mp9": "Weapon_MP9.Single",
    "mp7": "Weapon_MP7.Single",
    "mp5sd": "Weapon_MP5.Single",
    "bizon": "Weapon_bizon.Single",
    "p90": "Weapon_P90.Single",
    "ump45": "Weapon_UMP45.Single",
    "m249": "Weapon_M249.Single",
    "negev": "Weapon_Negev.Single",
    "nova": "Weapon_Nova.Single",
    "xm1014": "Weapon_XM1014.Single",
    "mag7": "Weapon_Mag7.Single",
    "sawedoff": "Weapon_Sawedoff.Single",
    "deagle": "Weapon_DEagle.Single",
    "glock": "Weapon_Glock.Single",
    "usp_silencer": "Weapon_USP.SilencedShot",
    "hkp2000": "Weapon_hkp2000.Single",
    "p250": "Weapon_P250.Single",
    "fiveseven": "Weapon_FiveSeven.Single",
    "tec9": "Weapon_tec9.Single",
    "cz75a": "Weapon_CZ75A.Single",
    "elite": "Weapon_ELITE.Single",
    "revolver": "Weapon_Revolver.Single",
    "taser": "Weapon_Taser.Single",
}

# Events the client starts from demo data (role → event). Kept in the always-loaded core bank.
CORE: dict[str, str] = {
    "throw.flashbang": "Flashbang.Throw",
    "throw.hegrenade": "HEGrenade.Throw",
    "throw.smokegrenade": "SmokeGrenade.Throw",
    "throw.decoy": "Decoy.Throw",
    "throw.molotov": "Molotov.Throw",
    "throw.incgrenade": "IncGrenade.Throw",
    "bounce.flashbang": "Flashbang.Bounce",
    "bounce.hegrenade": "HEGrenade.Bounce",
    "bounce.smokegrenade": "SmokeGrenade.Bounce",
    "bounce.decoy": "HEGrenade.Bounce",
    "bounce.molotov": "Molotov.Bounce",
    "bounce.incgrenade": "IncGrenade.Bounce",
    "detonate.hegrenade": "BaseGrenade.Explode",
    "detonate.flashbang": "Flashbang.Explode",
    "detonate.smokegrenade": "BaseSmokeEffect.Sound",
    "detonate.molotov": "Molotov.Start",
    "detonate.incgrenade": "IncGrenade.Start",
    "fire.loop": "Inferno.Loop",
    "fire.end": "Inferno.FadeOut",
    "fire.extinguish": "Molotov.Extinguish",
    "bomb.arm": "c4.initiate",
    "bomb.key": "c4.keypressquiet",
    "bomb.planted": "c4.plant",
    "bomb.beep": "C4.PlantSound",
    "bomb.trip": "C4.ExplodeTriggerTrip",
    "bomb.warning": "C4.ExplodeWarning",
    "bomb.explode": "c4.explode",
    "bomb.defuse": "c4.disarmstart",
    "bomb.defused": "c4.disarmfinish",
    "hit.body": "Player.DamageBody.Onlooker",
    "hit.bodyArmor": "Player.DamageBodyArmor.Onlooker",
    "hit.head": "Player.DamageHeadShot.Onlooker",
    "hit.headArmor": "Player.DamageHeadShotArmor.Onlooker",
    "death.body": "Player.DeathBody.Onlooker",
    "death.bodyArmor": "Player.DeathBodyArmor.Onlooker",
    "death.head": "Player.DeathHeadShot.Onlooker",
    "death.headArmor": "Player.DeathHeadShotArmor.Onlooker",
    "hit.burn": "Player.BurnDamage",
    "hit.fall": "Player.DamageFall",
    "bullet.crack": "BulletBy.Supersonic.Crack",
    "bullet.subsonic": "BulletBy.Subsonic",
    "bullet.impact": "Concrete.BulletImpact",
    "knife.swing": "Weapon_Knife.Slash",
    "knife.hit": "Weapon_Knife.Hit",
    "knife.stab": "Weapon_Knife.Stab",
    "jump": "Default.WalkJump",
    "gear.ct": "Gear.CT",
    "gear.t": "Gear.T",
    "land.default": "Default.Land",
    "break.glass": "Glass.Break",
    "break.wood": "Wood.Break",
    "break.metal": "Metal_Box.Break",
    "door.open": "Door.wood_start_open",
    "door.opened": "Door.wood_full_open",
    "door.close": "Door.wood_start_close",
    "door.closed": "Door.wood_full_close",
    "announce.planted": "Event.BombPlanted",
    "announce.defused": "Event.BombDefused",
    "announce.ctwin": "Event.CTWin",
    "announce.twin": "Event.TERWin",
}

# Viewmodel clip events not played as foley: the client starts these from demo data for every player.
FOLEY_SKIP = re.compile(r"\.(Single|Silenced|SilencedShot)$|^Weapon_Knife\.(Swish|Hit|Slash|Stab)|\.Throw$")

SOUND_KBPS = 48
AMBIENCE_KBPS = 16
OPUS_ARGS = ["-ac", "1", "-ar", "48000", "-c:a", "libopus", "-application", "audio", "-vbr", "on"]


def _log(message: str) -> None:
    print(f"[cs2sounds {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _chunks(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _decompile(cli: Path, vpk: Path, paths: Sequence[str], to: Path) -> None:
    """Decompiles resources into `to`, keeping the VPK layout (Source2Viewer treats one path as a file name)."""
    to.mkdir(parents=True, exist_ok=True)
    for chunk in _chunks(list(paths), 250):
        args = list(chunk) if len(chunk) > 1 else [chunk[0], chunk[0]]
        subprocess.run(
            [str(cli), "-i", str(vpk), "-d", "-f", ",".join(args), "-o", str(to)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )


def _text(work: Path, path: str) -> str:
    file = work / path
    if not file.is_file():
        raise RuntimeError(f"Source2Viewer did not write {path}")
    return file.read_text(encoding="utf-8", errors="replace")


def _blocks(cli: Path, vpk: Path, paths: Sequence[str], block: str = "DATA") -> dict[str, str]:
    """Compiled resource path (without `_c`) → the text of one of its blocks, many files per CLI call."""
    found: dict[str, str] = {}
    for chunk in _chunks(list(paths), 250):
        completed = subprocess.run(
            [str(cli), "-i", str(vpk), "-f", ",".join(f"{p}_c" for p in chunk), "--block", block],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )
        for part in re.split(r"\n\[\d+/\d+\] ", "\n" + completed.stdout)[1:]:
            name, _, text = part.partition("\n")
            found[re.sub(r"_c$", "", name.strip())] = text
    return found


def _sound_info(cli: Path, vpk: Path, paths: Sequence[str]) -> dict[str, tuple[bool, float]]:
    """vsnd path → (loops, duration), from the DATA blocks (LoopStart -1 is a one-shot, ≥ 0 loops)."""
    info: dict[str, tuple[bool, float]] = {}
    for name, block in _blocks(cli, vpk, paths).items():
        loop = re.search(r"LoopStart: (-?\d+)", block)
        duration = re.search(r"Duration: [^(]*\(([\d.]+)\)", block)
        info[name] = (bool(loop) and int(loop.group(1)) >= 0, float(duration.group(1)) if duration else 0.0)
    return info


def _opus_path(opus_dir: Path, path: str, kbps: int) -> Path:
    return opus_dir / f"{kbps}k" / f"{path}.ogg"


def _encode(ffmpeg: str, decoded: Path, path: str, opus_dir: Path, kbps: int) -> Path | None:
    target = _opus_path(opus_dir, path, kbps)
    if target.is_file():
        return target
    source = next((decoded / Path(path).with_suffix(ext) for ext in (".wav", ".mp3") if (decoded / Path(path).with_suffix(ext)).is_file()), None)
    if source is None:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part.ogg")
    completed = subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-i", str(source), *OPUS_ARGS, "-b:a", f"{kbps}k", "-map_metadata", "-1", str(partial)],
        capture_output=True, check=False,
    )
    if completed.returncode != 0 or not partial.is_file():
        return None
    partial.replace(target)
    return target


class Bank:
    def __init__(self, name: str) -> None:
        self.name = name
        self.data = bytearray()

    def add(self, blob: bytes) -> tuple[int, int]:
        offset = len(self.data)
        self.data += blob
        return offset, len(blob)


def _clip_foley(text: str) -> list[tuple[float, str]]:
    found = []
    for block in text.split('_class = "')[1:]:
        if not block.startswith("CNmSoundEvent"):
            continue
        start = re.search(r"m_flStartTime\s*=\s*\{\s*m_flValue = (-?[\d.e+-]+)", block)
        name = re.search(r'm_name = "([^"]*)"', block)
        if start and name and name.group(1):
            found.append((round(float(start.group(1)), 4), name.group(1)))
    return sorted(found)


def _physics_lines(cli: Path, map_vpk: Path, map_name: str) -> Iterable[str]:
    process = subprocess.Popen(
        [str(cli), "-i", str(map_vpk), "-f", f"maps/{map_name}/world_physics.vmdl_c", "--block", "PHYS"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
    )
    assert process.stdout is not None
    yield from process.stdout
    if process.wait() != 0:
        raise RuntimeError(f"Source2Viewer could not print the physics of {map_name}")


def build(
    cli: Path,
    cs2_dir: Path,
    maps: Sequence[str],
    output_dir: Path,
    work_dir: Path,
    ffmpeg: str,
    ambience_kbps: int = AMBIENCE_KBPS,
) -> dict[str, Any]:
    vpk = cs2_dir / "game" / "csgo" / "pak01_dir.vpk"
    work_dir.mkdir(parents=True, exist_ok=True)
    text_dir = work_dir / "text"

    _log("decompiling soundevents and surface tables")
    _decompile(
        cli, vpk,
        [f"{p}.vsndevts_c" for p in SOUNDEVENT_FILES]
        + [f"soundevents/ambience/game_sounds_{m.split('_', 1)[-1]}.vsndevts_c" for m in maps]
        + [f"{VSURF}_c", FOOTSTEPS],
        text_dir,
    )
    table = ev.load(_text(text_dir, f"{p}.vsndevts") for p in SOUNDEVENT_FILES)
    vsurf = kv3.loads(_text(text_dir, VSURF))
    footsteps = kv3.loads(_text(text_dir, FOOTSTEPS))
    by_hash, bases = surfaces.surface_names(vsurf)
    step_table = surfaces.footstep_table(footsteps)
    impact_by_surface = {
        str(e.get("surfacePropertyName", "")).lower(): (e.get("audiosounds") or {}).get("bulletimpact", "")
        for e in vsurf.get("SurfacePropertiesList", [])
    }

    # First-person foley: sound events of the clips each viewmodel plays.
    _log("reading viewmodel clip sound events")
    role_clips = cs2weapons.viewmodel_role_clips(cli, vpk, work_dir / "viewmodel", cs2weapons.VIEW_MODELS)
    clip_paths = sorted({clip for roles in role_clips.values() for clip in roles.values()})
    cues_file = work_dir / "clip_sounds.json"
    clip_cues: dict[str, list[list[Any]]] = json.loads(cues_file.read_text()) if cues_file.is_file() else {}
    missing_clips = [c for c in clip_paths if c not in clip_cues]
    if missing_clips:
        for clip, text in _blocks(cli, vpk, [f"{c}.vnmclip" for c in missing_clips]).items():
            clip_cues[re.sub(r"\.vnmclip$", "", clip)] = [list(c) for c in _clip_foley(text)]
        cues_file.write_text(json.dumps(clip_cues))
    foley: dict[str, dict[str, list[list[Any]]]] = {}
    for name, roles in role_clips.items():
        per_role = {}
        for role, clip in roles.items():
            cues = [[t, e] for t, e in clip_cues.get(clip, [])
                    if not FOLEY_SKIP.search(e) and e not in CORE.values() and ev.resolve(table, e)]
            if cues:
                per_role[role] = cues
        if per_role:
            foley[name] = per_role

    # Event sets per bank.
    core_events = ev.closure(table, CORE.values())
    weapon_events: dict[str, list[str]] = {}
    for name in sorted(set(SHOTS) | set(foley)):
        roots = ([SHOTS[name]] if name in SHOTS else []) + [e for cues in foley.get(name, {}).values() for _, e in cues]
        weapon_events[name] = [e for e in ev.closure(table, roots) if e not in core_events]

    # Map ambience and footsteps.
    map_data: dict[str, dict[str, Any]] = {}
    for map_name in maps:
        _log(f"{map_name}: ambience and footstep surfaces")
        entities = kv3.entities(source2viewer.extract_entity_data(cli, cs2_dir, map_name))
        # Ambience events live in game_sounds_<prefix> files; maps can reuse another map's (Ancient plays aztec.*).
        sound_names = [e.get("soundevent") or "" for e in entities if e.get("classname") == "env_soundscape"]
        sound_names += [e.get("soundName") or "" for e in entities if e.get("classname") == "point_soundevent"]
        stems = [map_name.split("_", 1)[-1]] + sorted({n.split(".", 1)[0] for n in sound_names if "." in n})
        stems = list(dict.fromkeys(s.lower() for s in stems))
        files = [f"soundevents/ambience/game_sounds_{s}.vsndevts" for s in stems]
        missing = [f"{f}_c" for f in files if not (text_dir / f).is_file()]
        if missing:
            _decompile(cli, vpk, missing, text_dir)
        map_table = dict(table)
        loaded = [(text_dir / f).read_text(encoding="utf-8", errors="replace") for f in files if (text_dir / f).is_file()]
        if loaded:
            map_table.update(ev.load(loaded))
        soundscapes = [
            {"o": [round(float(v), 1) for v in e["origin"]], "r": float(e.get("radius", 0) or 0), "e": e["soundevent"],
             **({"off": 1} if e.get("StartDisabled") else {})}
            for e in entities
            if e.get("classname") == "env_soundscape" and e.get("soundevent") and map_table.get(e["soundevent"])
        ]
        points = [
            {"o": [round(float(v), 1) for v in e["origin"]], "e": e["soundName"]}
            for e in entities
            if e.get("classname") == "point_soundevent" and e.get("startOnSpawn") and map_table.get(e.get("soundName", ""))
        ]

        phys = surfaces.parse(_physics_lines(cli, cs2_dir / "game" / "csgo" / "maps" / f"{map_name}.vpk", map_name))
        walkable = surfaces.walkable_attributes(phys.attributes)
        kept = [(t, m) for a, t, m in phys.floors if a in walkable]
        names = [by_hash.get(h, "default") for h in phys.surface_hashes]
        step_names = [surfaces.footstep_surface(n, step_table, bases) for n in names]
        used = sorted(set(step_names))
        index = np.array([used.index(s) for s in step_names], dtype=np.uint8)
        triangles = np.concatenate([t for t, _ in kept])
        materials = np.concatenate([m for _, m in kept])
        points_xyz, owners = surfaces.sample(triangles)
        materials = np.clip(materials, 0, len(index) - 1)
        grid = surfaces.bake_grid(points_xyz, index[materials[owners]])
        present = sorted({used[i] for i in np.unique(grid.surface)})
        steps = {}
        for surface in present:
            ct, t = step_table[surface]["ct"], step_table[surface]["t"]
            land = re.sub(r"^CT_", "Land_", ct)
            impact = impact_by_surface.get(surface, "")
            steps[surface] = {
                "ct": ct, "t": t,
                "land": land if land in table else CORE["land.default"],
                **({"impact": impact} if impact and impact in table else {}),
            }
        step_roots = [e for s in steps.values() for e in (s["ct"], s["t"], s["land"], s.get("impact")) if e]
        ambience_roots = [s["e"] for s in soundscapes] + [p["e"] for p in points]
        map_data[map_name] = {
            "table": map_table,
            "events": [e for e in ev.closure(map_table, step_roots + ambience_roots) if e not in core_events],
            "step_events": [e for e in ev.closure(map_table, step_roots) if e not in core_events],
            "soundscapes": soundscapes,
            "points": points,
            "surfaces": used,
            "steps": steps,
            "grid": grid,
        }

    # Files per bank: a file shared by several banks (other than maps) goes to core.
    def files_of(names: Iterable[str], source: dict[str, dict[str, Any]]) -> list[str]:
        return list(dict.fromkeys(f for n in names for f in ev.files(ev.resolve(source, n) or {})))

    owners: dict[str, set[str]] = {}
    for f in files_of(core_events, table):
        owners.setdefault(f, set()).add("core")
    for name, names in weapon_events.items():
        for f in files_of(names, table):
            owners.setdefault(f, set()).add(f"weapon_{name}")
    global_bank = {f: ("core" if len(o) > 1 else next(iter(o))) for f, o in owners.items()}
    # Per map: bank 0 (footsteps, landings, impacts) loads with the map; bank 1 (ambience) is fetched per sound
    # with Range requests when a soundscape or point sound first needs it, and is encoded at a lower bitrate.
    map_files: dict[str, list[list[str]]] = {}
    for m, d in map_data.items():
        step_files = [f for f in files_of(d["step_events"], d["table"]) if f not in global_bank]
        ambience_files = [f for f in files_of(d["events"], d["table"]) if f not in global_bank and f not in step_files]
        map_files[m] = [step_files, ambience_files]
    ambience = {f for banks_of_map in map_files.values() for f in banks_of_map[1]}
    bitrate = {f: ambience_kbps if f in ambience else SOUND_KBPS for f in global_bank}
    bitrate.update({f: ambience_kbps for f in ambience})
    bitrate.update({f: SOUND_KBPS for banks_of_map in map_files.values() for f in banks_of_map[0]})

    every = sorted(bitrate)
    decoded = work_dir / "decoded"
    undecoded = [f for f in every if not any((decoded / Path(f).with_suffix(x)).is_file() for x in (".wav", ".mp3"))
                 and not _opus_path(work_dir / "opus", f, bitrate[f]).is_file()]
    if undecoded:
        _log(f"decoding {len(undecoded)} sounds")
        _decompile(cli, vpk, [f"{f}_c" for f in undecoded], decoded)
    info_file = work_dir / "sound_info.json"
    info = json.loads(info_file.read_text()) if info_file.is_file() else {}
    unknown = [f for f in every if f not in info]
    if unknown:
        _log(f"reading loop flags of {len(unknown)} sounds")
        info.update({k: list(v) for k, v in _sound_info(cli, vpk, unknown).items()})
        info_file.write_text(json.dumps(info))
    loops = {f: bool(info.get(f, [False])[0]) for f in every}

    _log(f"encoding {len(every)} sounds to Opus")
    with ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4))) as pool:
        encoded = dict(zip(every, pool.map(lambda f: _encode(ffmpeg, decoded, f, work_dir / "opus", bitrate[f]), every)))
    failed = sorted(f for f, path in encoded.items() if path is None)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "maps").mkdir(parents=True)

    banks: dict[str, Bank] = {}
    sounds: list[list[Any]] = []
    sound_ids: dict[str, int] = {}
    for f in sorted(global_bank, key=lambda f: (global_bank[f], f)):
        if encoded.get(f) is None:
            continue
        bank_name = global_bank[f]
        bank = banks.setdefault(bank_name, Bank(bank_name))
        offset, length = bank.add(encoded[f].read_bytes())
        sound_ids[f] = len(sounds)
        sounds.append([bank_name, offset, length])

    def compact_all(names: Iterable[str], source: dict[str, dict[str, Any]], ids: dict[str, int]) -> dict[str, Any]:
        out = {}
        for n in names:
            params = ev.resolve(source, n)
            if params is not None:
                out[n] = ev.compact(params, ids, loops)
        return out

    global_events = compact_all(core_events + [e for names in weapon_events.values() for e in names], table, sound_ids)
    weapons = {
        name: {"bank": f"weapon_{name}", **({"shot": SHOTS[name]} if name in SHOTS else {}), **({"foley": foley[name]} if name in foley else {})}
        for name in weapon_events
    }
    weapons = {name: w if f"weapon_{name}" in banks else {k: v for k, v in w.items() if k != "bank"} for name, w in weapons.items()}
    manifest = {
        "version": 1,
        "banks": {name: f"{name}.bin" for name in banks},
        "sounds": sounds,
        "core": CORE,
        "weapons": weapons,
        "events": global_events,
    }
    for name, bank in banks.items():
        (output_dir / f"{name}.bin").write_bytes(bytes(bank.data))
    (output_dir / "sounds.json").write_text(json.dumps(manifest, separators=(",", ":")))

    report: dict[str, Any] = {"banks": {name: len(bank.data) for name, bank in banks.items()}, "maps": {}, "failed": failed}
    for map_name, data in map_data.items():
        map_banks = [Bank(map_name), Bank(f"{map_name}_ambience")]
        ids = dict(sound_ids)
        local: list[list[int]] = []
        for bank_index, files in enumerate(map_files[map_name]):
            for f in files:
                if encoded.get(f) is None:
                    continue
                offset, length = map_banks[bank_index].add(encoded[f].read_bytes())
                ids[f] = -1 - len(local)
                local.append([bank_index, offset, length])
        grid_bytes = data["grid"].encode()
        for bank in map_banks:
            (output_dir / "maps" / f"{bank.name}.bin").write_bytes(bytes(bank.data))
        (output_dir / "maps" / f"{map_name}_surfaces.bin").write_bytes(grid_bytes)
        (output_dir / "maps" / f"{map_name}.json").write_text(json.dumps({
            "version": 1,
            "banks": [f"{bank.name}.bin" for bank in map_banks],
            "sounds": local,
            "events": compact_all(data["events"], data["table"], ids),
            "soundscapes": data["soundscapes"],
            "points": data["points"],
            "surfaces": data["surfaces"],
            "steps": data["steps"],
            "grid": f"{map_name}_surfaces.bin",
        }, separators=(",", ":")))
        report["maps"][map_name] = {
            "bank": len(map_banks[0].data), "ambience": len(map_banks[1].data),
            "grid": len(grid_bytes), "soundscapes": len(data["soundscapes"]),
        }
    report["total"] = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file())
    return report
