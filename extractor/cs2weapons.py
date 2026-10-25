"""Bakes CS2's weapon models for the web client's dropped weapons, grenades and first-person viewmodel.

Like cs2render, this is an offline bake from a local CS2 install; the client loads the hosted result
(VITE_CS2_WEAPONS_URL, or public/cs2weapons/ in development). Source2Viewer-CLI exports the models, the
Node scripts in weapon-bake/ reduce and encode them (run `npm install` there once).

Output layout (output_dir):
  manifest.json      format, version, file sizes
  world/<name>.glb   dropped-weapon / grenade world model: reduced, untextured; per-vertex colour (COLOR_0)
                     and AO/roughness/metalness (_ORM), meshopt-compressed, Source inches, Z up; node extras
                     `weaponBone` hold the root bone's bind pose, where the model attaches to a player's hand
  fp/<name>.glb      first-person viewmodel: skinned weapon, AnimGraph2 clips (draw, idle, shoot, ...),
                     WebP colour / ORM / normal maps
  fp/arms.glb        the shared first-person arms
  characters/ct.glb  third-person player model per side (SAS, Phoenix): skinned body and gloves, reduced,
  characters/t.glb   WebP colour / ORM / normal maps, glTF metres, Y up
  characters/anims.bin  the shared third-person clips (locomotion, per-weapon idle/shoot/draw/reload, grenade,
                     C4, defuse, deaths) on the model skeleton, with per-clip kind, ground speed and direction;
                     a compact meshopt-compressed binary (format in weapon-bake/bake-character.mjs)

Exports are cached in work_dir (world/<name>/, fp/graphs/, fp/<name>/ clips, characters/); model texture
exports are large and deleted after each viewmodel bake.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Sequence

from . import source2viewer

BAKE_DIR = Path(__file__).resolve().parent.parent / "weapon-bake"

# Every knife in the default finish: model folder name under weapons/models/knife/knife_<name>/, and its
# viewmodel graph (a viewmodel_knife variant, or a graph of its own).
KNIVES: tuple[tuple[str, str], ...] = (
    ("bayonet", "viewmodel_knife.vnmgraph+bayonet"),
    ("bowie", "viewmodel_knife.vnmgraph+bowie"),
    ("butterfly", "viewmodel_knife_butterfly"),
    ("canis", "viewmodel_knife.vnmgraph+canis"),
    ("cord", "viewmodel_knife.vnmgraph+cord"),
    ("css", "viewmodel_knife.vnmgraph+css"),
    ("falchion", "viewmodel_knife.vnmgraph+falchion"),
    ("flip", "viewmodel_knife.vnmgraph+flip"),
    ("gut", "viewmodel_knife.vnmgraph+gut"),
    ("karambit", "viewmodel_knife_karambit"),
    ("kukri", "viewmodel_knife.vnmgraph+kukri"),
    ("m9", "viewmodel_knife.vnmgraph+m9"),
    ("navaja", "viewmodel_knife.vnmgraph+navajo"),
    ("outdoor", "viewmodel_knife.vnmgraph+outdoor"),
    ("push", "viewmodel_knife.vnmgraph+push"),
    ("skeleton", "viewmodel_knife.vnmgraph+skeleton"),
    ("stiletto", "viewmodel_knife.vnmgraph+stiletto"),
    ("tactical", "viewmodel_knife.vnmgraph+tactical"),
    ("talon", "viewmodel_knife_talon"),
    ("ursus", "viewmodel_knife.vnmgraph+ursus"),
)

# name, model path under weapons/models/ (without .vmdl_c)
WORLD_MODELS: tuple[tuple[str, str], ...] = (
    ("ak47", "ak47/weapon_rif_ak47"),
    ("aug", "aug/weapon_rif_aug"),
    ("awp", "awp/weapon_snip_awp"),
    ("bizon", "bizon/weapon_smg_bizon"),
    ("famas", "famas/weapon_rif_famas"),
    ("g3sg1", "g3sg1/weapon_snip_g3sg1"),
    ("galilar", "galilar/weapon_rif_galilar"),
    ("m249", "m249/weapon_mach_m249"),
    ("m4a1_silencer", "m4a1_silencer/weapon_rif_m4a1_silencer"),
    ("m4a4", "m4a4/weapon_rif_m4a4"),
    ("mac10", "mac10/weapon_smg_mac10"),
    ("mag7", "mag7/weapon_shot_mag7"),
    ("mp5sd", "mp5sd/weapon_smg_mp5sd"),
    ("mp7", "mp7/weapon_smg_mp7"),
    ("mp9", "mp9/weapon_smg_mp9"),
    ("negev", "negev/weapon_mach_negev"),
    ("nova", "nova/weapon_shot_nova"),
    ("p90", "p90/weapon_smg_p90"),
    ("sawedoff", "sawedoff/weapon_shot_sawedoff"),
    ("scar20", "scar20/weapon_snip_scar20"),
    ("sg556", "sg556/weapon_rif_sg556"),
    ("ssg08", "ssg08/weapon_snip_ssg08"),
    ("ump45", "ump45/weapon_smg_ump45"),
    ("xm1014", "xm1014/weapon_shot_xm1014"),
    ("cz75a", "cz75a/weapon_pist_cz75a"),
    ("deagle", "deagle/weapon_pist_deagle"),
    ("elite", "elite/weapon_pist_elite"),
    ("fiveseven", "fiveseven/weapon_pist_fiveseven"),
    ("glock", "glock18/weapon_pist_glock18"),
    ("hkp2000", "hkp2000/weapon_pist_hkp2000"),
    ("p250", "p250/weapon_pist_p250"),
    ("revolver", "revolver/weapon_pist_revolver"),
    ("tec9", "tec9/weapon_pist_tec9"),
    ("usp_silencer", "usp_silencer/weapon_pist_usp_silencer"),
    ("taser", "taser/weapon_pist_taser"),
    ("c4", "c4/weapon_c4"),
    ("decoy", "grenade/decoy/weapon_decoy"),
    ("flashbang", "grenade/flashbang/weapon_flashbang"),
    ("hegrenade", "grenade/hegrenade/weapon_hegrenade"),
    ("incgrenade", "grenade/incendiary/weapon_incendiarygrenade"),
    ("molotov", "grenade/molotov/weapon_molotov"),
    ("smokegrenade", "grenade/smokegrenade/weapon_smokegrenade"),
    *((f"knife_{side}", f"knife/knife_default_{side}/weapon_knife_default_{side}") for side in ("t", "ct")),
    *((f"knife_{name}", f"knife/knife_{name}/weapon_knife_{name}") for name, _ in KNIVES),
)

# name, model path, AnimGraph2 graph under animation/graphs/viewmodel/ (+variant), optional clip filter for
# graphs shared by several items.
VIEW_MODELS: tuple[tuple[str, str, str, str | None], ...] = (
    *((name, path, f"viewmodel_gun.vnmgraph+{variant}", None) for name, path, variant in (
        ("ak47", "ak47/weapon_rif_ak47", "ak47"),
        ("aug", "aug/weapon_rif_aug", "aug"),
        ("awp", "awp/weapon_snip_awp", "awp"),
        ("bizon", "bizon/weapon_smg_bizon", "bizon"),
        ("famas", "famas/weapon_rif_famas", "famas"),
        ("g3sg1", "g3sg1/weapon_snip_g3sg1", "g3sg1"),
        ("galilar", "galilar/weapon_rif_galilar", "galil"),
        ("m249", "m249/weapon_mach_m249", "m249"),
        ("m4a1_silencer", "m4a1_silencer/weapon_rif_m4a1_silencer", "m4a1s"),
        ("m4a4", "m4a4/weapon_rif_m4a4", "m4a4"),
        ("mac10", "mac10/weapon_smg_mac10", "mac10"),
        ("mag7", "mag7/weapon_shot_mag7", "mag7"),
        ("mp5sd", "mp5sd/weapon_smg_mp5sd", "mp5sd"),
        ("mp7", "mp7/weapon_smg_mp7", "mp7"),
        ("mp9", "mp9/weapon_smg_mp9", "mp9"),
        ("negev", "negev/weapon_mach_negev", "negev"),
        ("nova", "nova/weapon_shot_nova", "nova"),
        ("p90", "p90/weapon_smg_p90", "p90"),
        ("sawedoff", "sawedoff/weapon_shot_sawedoff", "sawedoff"),
        ("scar20", "scar20/weapon_snip_scar20", "scar20"),
        ("sg556", "sg556/weapon_rif_sg556", "sg556"),
        ("ssg08", "ssg08/weapon_snip_ssg08", "ssg08"),
        ("ump45", "ump45/weapon_smg_ump45", "ump45"),
        ("xm1014", "xm1014/weapon_shot_xm1014", "xm1014"),
        ("deagle", "deagle/weapon_pist_deagle", "deagle"),
        ("fiveseven", "fiveseven/weapon_pist_fiveseven", "five_seven"),
        ("glock", "glock18/weapon_pist_glock18", "glock"),
        ("hkp2000", "hkp2000/weapon_pist_hkp2000", "hkp2000"),
        ("p250", "p250/weapon_pist_p250", "p250"),
        ("tec9", "tec9/weapon_pist_tec9", "tec9"),
        ("usp_silencer", "usp_silencer/weapon_pist_usp_silencer", "usp"),
        ("taser", "taser/weapon_pist_taser", "taser"),
    )),
    ("cz75a", "cz75a/weapon_pist_cz75a", "viewmodel_gun_cz75", None),
    ("elite", "elite/weapon_pist_elite", "viewmodel_gun_elites", None),
    ("revolver", "revolver/weapon_pist_revolver", "viewmodel_gun_revolver", None),
    ("c4", "c4/weapon_c4", "viewmodel", r"_c4$"),
    *((name, path, f"viewmodel_grenade.vnmgraph+{variant}", None) for name, path, variant in (
        ("decoy", "grenade/decoy/weapon_decoy", "decoy"),
        ("flashbang", "grenade/flashbang/weapon_flashbang", "flash"),
        ("hegrenade", "grenade/hegrenade/weapon_hegrenade", "he"),
        ("incgrenade", "grenade/incendiary/weapon_incendiarygrenade", "incendiary"),
        ("molotov", "grenade/molotov/weapon_molotov", "molotov"),
        ("smokegrenade", "grenade/smokegrenade/weapon_smokegrenade", "smoke"),
    )),
    ("knife_t", "knife/knife_default_t/weapon_knife_default_t", "viewmodel_knife.vnmgraph+default_t", None),
    ("knife_ct", "knife/knife_default_ct/weapon_knife_default_ct", "viewmodel_knife.vnmgraph+default_ct", None),
    *((f"knife_{name}", f"knife/knife_{name}/weapon_knife_{name}", graph, None) for name, graph in KNIVES),
)
ARMS_MODEL = "weapons/models/shared/arms/weapon_arms.vmdl_c"

# The first clip in graph order whose path matches wins. shoot2 is the alternate attack (right-hand Elites
# shot, knife second swing); idle variants that only play in special states are skipped. Grenades: pin
# pull, then the full-charge hold until the throw. C4: the plant.
ROLES: dict[str, str] = {
    "draw": r"/draw_(?!2_)[^/]*$",
    "idle": r"/idle1?_(?!.*(slide_back|from_activity|leftempty|leftrightempty))[^/]*$",
    "shoot": r"/(shoot1_|shoot_left1_|light_miss1_|throw_overhand_)[^/]*$",
    "shoot2": r"/(shoot_right1_|light_miss2_)[^/]*$",
    "reload": r"/reload_(?!empty)[^/]*$",
    "pullpin": r"/pullpin_[^/]*$",
    "hold": r"/throwcharge_high_[^/]*$",
    "plant": r"/plant_c4$",
}
GLTF = ["--gltf_export_format", "glb", "--gltf_export_animations"]

# Third-person player models: one agent per side. Only the third-person meshes are exported; the skin is
# only emitted with --gltf_export_animations, and an animation list that matches nothing keeps the ~2,000
# embedded clips (about 1 GB) out of the export.
CHARACTER_MODELS: tuple[tuple[str, str], ...] = (
    ("ct", "agents/models/ctm_sas/ctm_sas.vmdl_c"),
    ("t", "agents/models/tm_phoenix/tm_phoenix.vmdl_c"),
)
CHARACTER_MODEL_ARGS = [
    *GLTF, "--gltf_animation_list", "__none__", "--gltf_mesh_list", "thirdperson_body,thirdperson_default_gloves",
    "--gltf_export_materials", "--gltf_textures_adapt",
]
WORLD_ANIMS = "animation/anims/world"
DIRECTIONS = ("n", "ne", "e", "se", "s", "sw", "w", "nw")
# name (as in WORLD_MODELS), clip folder under rifle/ or pistol/, clip suffix, shoot clip when not shoot_<suffix>
CHARACTER_WEAPONS: tuple[tuple[str, str, str, str | None], ...] = (
    ("ak47", "rifle/rifle_ak", "ak", None),
    ("aug", "rifle/rifle_aug", "aug", None),
    ("awp", "rifle/rifle_awp", "awp", None),
    ("bizon", "rifle/rifle_bizon", "bizon", None),
    ("famas", "rifle/rifle_famas", "famas", None),
    ("g3sg1", "rifle/rifle_g3sg1", "g3sg1", None),
    ("galilar", "rifle/rifle_galilar", "galilar", None),
    ("m249", "rifle/rifle_m249", "m249", None),
    ("m4a1_silencer", "rifle/rifle_m4a1_silencer", "m4a1s", None),
    ("m4a4", "rifle/rifle_m4a4", "m4a4", None),
    ("mac10", "rifle/rifle_mac10", "mac10", None),
    ("mag7", "rifle/rifle_mag7", "mag7", None),
    ("mp5sd", "rifle/rifle_mp5sd", "mp5sd", None),
    ("mp7", "rifle/rifle_mp7", "mp7", None),
    ("mp9", "rifle/rifle_mp9", "mp9", None),
    ("negev", "rifle/rifle_negev", "negev", None),
    ("nova", "rifle/rifle_nova", "nova", None),
    ("p90", "rifle/rifle_p90", "p90", None),
    ("sawedoff", "rifle/rifle_sawedoff", "sawedoff", None),
    ("scar20", "rifle/rifle_scar", "scar20", None),
    ("sg556", "rifle/rifle_sg556", "sg556", None),
    ("ssg08", "rifle/rifle_ssg08", "ssg08", None),
    ("ump45", "rifle/rifle_ump45", "ump45", None),
    ("xm1014", "rifle/rifle_xm1014", "xm1014", None),
    ("cz75a", "pistol/pistol_cz75a", "cz75a", "shoot_cz75"),
    ("deagle", "pistol/pistol_deagle", "deagle", None),
    ("elite", "pistol/pistol_elite", "elite", "shoot_left1_elite"),
    ("fiveseven", "pistol/pistol_fiveseven", "fiveseven", None),
    ("glock", "pistol/pistol_glock", "glock", None),
    ("hkp2000", "pistol/pistol_hkp2000", "hkp", None),
    ("p250", "pistol/pistol_p250", "p250", None),
    ("revolver", "pistol/pistol_revolver", "revolver", None),
    ("taser", "pistol/pistol_taser", "taser", None),
    ("tec9", "pistol/pistol_tec9", "tec9", None),
    ("usp_silencer", "pistol/pistol_usp", "usp", None),
)


def character_clips() -> list[tuple[str, str, str]]:
    """(name, clip path without .vnmclip_c, kind) for the third-person animation set. Kinds: `move` (full-body
    locomotion cycle, ground speed measured), `full` (full-body), `upper` (upper-body layer, legs dropped),
    `additive` (local delta over the class idle: rotation post-multiplied, translation added)."""
    clips: list[tuple[str, str, str]] = []
    for kind in ("rifle", "pistol", "knife"):
        base = f"{WORLD_ANIMS}/{kind}/_default_{kind}"
        clips += [
            (f"{kind}_idle", f"{base}/idle_{kind}", "full"),
            (f"{kind}_idle_crouch", f"{base}/idle_crouch_{kind}", "full"),
            (f"{kind}_inair", f"{base}/inair_stand_{kind}", "full"),
            (f"{kind}_inair_crouch", f"{base}/inair_crouch_stand_{kind}", "full"),
        ]
        clips += [(f"{kind}_{gait}_{d}", f"{base}/{gait}_{d}_{kind}", "move") for gait in ("walk", "run", "crouch") for d in DIRECTIONS]
        clips += [
            (f"defuse_{kind}", f"{WORLD_ANIMS}/shared/defuse/defuse_loop_{kind}", "full"),
            (f"defuse_crouch_{kind}", f"{WORLD_ANIMS}/shared/defuse/defuse_crouch_loop_{kind}", "full"),
        ]
    knife = f"{WORLD_ANIMS}/knife/_default_knife"
    grenade = f"{WORLD_ANIMS}/grenade/_default_grenade"
    c4 = f"{WORLD_ANIMS}/equipment/c4"
    clips += [
        ("knife_draw", f"{knife}/draw_knife", "upper"),
        ("knife_attack", f"{knife}/frontswing_knife", "upper"),
        # Grenades and C4 are held in the draw clips' last pose: idle_grenade / idle_c4 leave the weapon bone at the feet.
        ("grenade_draw", f"{grenade}/draw_grenade", "upper"),
        ("grenade_pullpin", f"{grenade}/pullpin_grenade", "upper"),
        ("grenade_throw", f"{grenade}/throw_overhand_grenade", "upper"),
        ("c4_draw", f"{c4}/draw_c4", "upper"),
        ("c4_plant", f"{c4}/planting", "full"),
        ("c4_plant_crouch", f"{c4}/planting_crouch", "full"),
        ("rifle_shoot", f"{WORLD_ANIMS}/rifle/_default_rifle/shoot_rifle", "additive"),
    ]
    clips += [(name, f"{WORLD_ANIMS}/shared/{name}", "full") for name in ("death_chest_a", "death_chest_b", "death_gut_a", "death_rknee_a", "death_rshoulder")]
    for name, folder, suffix, shoot in CHARACTER_WEAPONS:
        path = f"{WORLD_ANIMS}/{folder}"
        clips += [
            (f"{name}_idle", f"{path}/idle_{suffix}", "additive"),
            (f"{name}_shoot", f"{path}/{shoot or 'shoot_' + suffix}", "additive"),
            (f"{name}_draw", f"{path}/draw_{suffix}", "upper"),
        ]
        if name != "taser":
            clips.append((f"{name}_reload", f"{path}/reload_{suffix}", "upper"))
    return clips


def _log(message: str) -> None:
    print(f"[cs2weapons {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _export(cli: Path, vpk: Path, paths: Sequence[str], to: Path, extra: Sequence[str], extension: str | None = None) -> None:
    """With several paths -o is a folder that keeps the VPK layout; with one it is the output file, placed
    where the folder layout would have put it. The CLI logs harmless shader exceptions, so output is dropped."""
    if len(paths) == 1:
        target = to / re.sub(r"\.[^./]+$", extension or "", re.sub(r"_c$", "", paths[0]))
        target.parent.mkdir(parents=True, exist_ok=True)
        output = target
    else:
        to.mkdir(parents=True, exist_ok=True)
        output = to
    _run_cli([str(cli), "-i", str(vpk), "-d", "-f", ",".join(paths), "-o", str(output), *extra])


_last_cli_output = ""


def _run_cli(args: list[str]) -> None:
    global _last_cli_output
    _last_cli_output = source2viewer.run_quiet(args)


def _export_error(message: str) -> RuntimeError:
    """An export error carrying Source2Viewer's last output, which is otherwise not shown."""
    return RuntimeError(f"{message}\n{_last_cli_output}")


def _node(script: str, args: Sequence[str]) -> None:
    completed = subprocess.run(
        ["node", str(BAKE_DIR / script), *args], cwd=BAKE_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    for line in (completed.stdout + completed.stderr).splitlines():
        if line.strip():
            print(f"  {line}", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{script} failed: {' '.join(args)}")


def bake_world(cli: Path, vpk: Path, work_dir: Path, out_dir: Path, only: set[str], target_verts: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, path in WORLD_MODELS:
        if only and name not in only:
            continue
        export = work_dir / name / f"{name}.glb"
        if not export.is_file():
            _log(f"exporting world model {name}")
            export.parent.mkdir(parents=True, exist_ok=True)
            _run_cli(
                [str(cli), "-i", str(vpk), "-d", "-f", f"weapons/models/{path}.vmdl_c", "-o", str(export),
                 "--gltf_export_format", "glb", "--gltf_export_materials", "--gltf_textures_adapt"]
            )
            if not export.is_file():
                raise _export_error(f"export failed for {name} ({path})")
        _node("bake-weapon-model.mjs", [str(export), str(out_dir / f"{name}.glb"), f"--target-verts={target_verts}"])
        skeleton = work_dir / name / "skeleton" / f"{name}.glb"
        if not skeleton.is_file():
            # Exporting animations is what makes Source2Viewer write the skeleton; no clip is needed.
            skeleton.parent.mkdir(parents=True, exist_ok=True)
            _run_cli(
                [str(cli), "-i", str(vpk), "-d", "-f", f"weapons/models/{path}.vmdl_c", "-o", str(skeleton),
                 "--gltf_export_format", "glb", "--gltf_export_animations", "--gltf_animation_list", "__none__"]
            )
            if not skeleton.is_file():
                raise _export_error(f"skeleton export failed for {name} ({path})")
        _node("weapon-attach.mjs", [str(skeleton), str(out_dir / f"{name}.glb")])


def _export_model(cli: Path, vpk: Path, textures: Path, vpk_path: str) -> Path:
    shutil.rmtree(textures, ignore_errors=True)
    _export(cli, vpk, [vpk_path], textures, [*GLTF, "--gltf_export_materials", "--gltf_textures_adapt"], ".glb")
    file = textures / re.sub(r"\.[^./]+$", ".glb", re.sub(r"_c$", "", vpk_path))
    if not file.is_file():
        raise _export_error(f"model export failed: {vpk_path}")
    return file


def viewmodel_role_clips(
    cli: Path, vpk: Path, work_dir: Path, selected: Sequence[tuple[str, str, str, str | None]]
) -> dict[str, dict[str, str]]:
    """Viewmodel name → {role: clip path without extension}, from the (cached) AnimGraph2 graphs."""
    graphs = work_dir / "graphs"
    graph_file = lambda graph: graphs / "animation" / "graphs" / "viewmodel" / f"{graph}.vnmgraph"  # noqa: E731
    missing = sorted({graph for _, _, graph, _ in selected if not graph_file(graph).is_file()})
    if missing:
        _log(f"exporting {len(missing)} animation graphs")
        _export(cli, vpk, [f"animation/graphs/viewmodel/{graph}.vnmgraph_c" for graph in missing], graphs, [], ".vnmgraph")

    result: dict[str, dict[str, str]] = {}
    for name, _path, graph, clip_filter in selected:
        if not graph_file(graph).is_file():
            raise _export_error(f"graph export failed for {name} ({graph})")
        text = graph_file(graph).read_text(encoding="utf-8", errors="replace")
        clips = list(dict.fromkeys(re.findall(r'm_clip = "([^"]+)\.vnmclip"', text)))
        if clip_filter:
            clips = [clip for clip in clips if re.search(clip_filter, clip)]
        picked = {}
        for role, pattern in ROLES.items():
            match = next((clip for clip in clips if re.search(pattern, clip)), None)
            if match:
                picked[role] = match
        if "draw" not in picked or "idle" not in picked:
            raise RuntimeError(f"no draw/idle clip for {name} in {graph}")
        result[name] = picked
    return result


def bake_viewmodels(cli: Path, vpk: Path, work_dir: Path, out_dir: Path, only: set[str], texture_size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    textures = work_dir / "textures"
    texture_args = [f"--texture-size={texture_size}"]
    if not only or "arms" in only:
        _log("baking arms")
        _node("bake-viewmodel.mjs", ["--arms", str(_export_model(cli, vpk, textures, ARMS_MODEL)), str(out_dir / "arms.glb"), *texture_args])

    selected = [m for m in VIEW_MODELS if not only or m[0] in only]
    role_clips = viewmodel_role_clips(cli, vpk, work_dir, selected)
    for name, path, _graph, _clip_filter in selected:
        picked = role_clips[name]
        clip_dir = work_dir / name
        clip_files = {role: clip_dir / f"{clip}.glb" for role, clip in picked.items()}
        if any(not file.is_file() for file in clip_files.values()):
            _log(f"exporting {len(picked)} clips for {name}")
            _export(cli, vpk, [f"{clip}.vnmclip_c" for clip in picked.values()], clip_dir, GLTF)
            lost = [str(file) for file in clip_files.values() if not file.is_file()]
            if lost:
                raise _export_error(f"clip export failed for {name}: {lost}")
        _log(f"baking viewmodel {name}")
        model = _export_model(cli, vpk, textures, f"weapons/models/{path}.vmdl_c")
        _node(
            "bake-viewmodel.mjs",
            [str(model), str(out_dir / f"{name}.glb"), *(f"--clip={role}:{file}" for role, file in clip_files.items()), *texture_args],
        )
    shutil.rmtree(textures, ignore_errors=True)


def bake_characters(
    cli: Path, vpk: Path, work_dir: Path, out_dir: Path, only: set[str], texture_size: int, ratio: float
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [(side, path) for side, path in CHARACTER_MODELS if not only or side in only]
    exports: dict[str, Path] = {}
    for side, path in CHARACTER_MODELS:
        export = work_dir / "models" / side / re.sub(r"\.vmdl_c$", ".glb", path)
        if not export.is_file():
            _log(f"exporting character model {side}")
            _export(cli, vpk, [path], work_dir / "models" / side, CHARACTER_MODEL_ARGS, ".glb")
            if not export.is_file():
                raise _export_error(f"character export failed: {path}")
        exports[side] = export
    for side, _ in models:
        _log(f"baking character {side}")
        _node(
            "bake-character.mjs",
            ["--model", str(exports[side]), str(out_dir / f"{side}.glb"), f"--texture-size={texture_size}", f"--ratio={ratio}"],
        )
    if only and "anims" not in only:
        return

    clips = character_clips()
    clip_dir = work_dir / "clips"
    files = {name: clip_dir / f"{path}.glb" for name, path, _ in clips}
    missing = [path for name, path, _ in clips if not files[name].is_file()]
    if missing:
        _log(f"exporting {len(missing)} character clips")
        for start in range(0, len(missing), 60):
            _export(cli, vpk, [f"{path}.vnmclip_c" for path in missing[start:start + 60]], clip_dir, GLTF, ".glb")
        lost = [str(files[name]) for name, _, _ in clips if not files[name].is_file()]
        if lost:
            raise _export_error(f"character clip export failed: {lost}")
    spec = work_dir / "anims.json"
    spec.write_text(
        json.dumps({
            "skeleton": str(exports[CHARACTER_MODELS[0][0]]),
            "clips": [{"name": name, "file": str(files[name]), "kind": kind} for name, _, kind in clips],
        }, indent=1),
        encoding="utf-8",
    )
    _log(f"baking {len(clips)} character clips")
    _node("bake-character.mjs", ["--anims", str(spec), str(out_dir / "anims.bin")])


def build(
    cli: Path,
    cs2_dir: Path,
    output_dir: Path,
    work_dir: Path,
    only: Sequence[str] = (),
    target_verts: str = "15%",
    texture_size: int = 512,
    sets: Sequence[str] = ("world", "fp", "characters"),
    character_texture_size: int = 1024,
    character_ratio: float = 0.5,
) -> dict[str, object]:
    if not (BAKE_DIR / "node_modules").is_dir():
        raise RuntimeError(f"run `npm install` in {BAKE_DIR} first")
    vpk = cs2_dir / "game" / "csgo" / "pak01_dir.vpk"
    if not vpk.is_file():
        raise FileNotFoundError(f"CS2 VPK was not found: {vpk}")
    selected = set(only)
    output_dir = output_dir.resolve()
    work_dir = work_dir.resolve()
    started = time.time()
    if "world" in sets:
        bake_world(cli, vpk, work_dir / "world", output_dir / "world", selected, target_verts)
    if "fp" in sets:
        bake_viewmodels(cli, vpk, work_dir / "fp", output_dir / "fp", selected, texture_size)
    if "characters" in sets:
        bake_characters(
            cli, vpk, work_dir / "characters", output_dir / "characters", selected, character_texture_size, character_ratio
        )

    files = {
        folder: {path.stem: path.stat().st_size for path in sorted((output_dir / folder).iterdir()) if path.suffix in (".glb", ".bin")}
        for folder in ("world", "fp", "characters")
        if (output_dir / folder).is_dir()
    }
    manifest = {"format": "cs2weapons", "version": 1, "files": files}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return {
        "seconds": round(time.time() - started, 1),
        "bytes": {folder: sum(sizes.values()) for folder, sizes in files.items()},
        "models": {folder: len(sizes) for folder, sizes in files.items()},
    }
