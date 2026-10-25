# csdemo-mapextractor

Extract browser-ready map data from a local Counter-Strike 2 installation.

For each map, the extractor can produce:

- collision geometry as GLB;
- radar images, including vertical sections such as `lower`;
- overview metadata;
- the map logo as SVG.

It uses [Source2Viewer](https://github.com/ValveResourceFormat/ValveResourceFormat)
to read CS2's Source 2 files.

## Requirements

- Python 3.11 or newer
- Counter-Strike 2
- Source2Viewer CLI
- Node.js (only for `cs2weapons`)
- ffmpeg with libopus (only for `cs2sounds`)

Install the project with Meshopt support:

```powershell
python -m pip install -e ".[meshopt]"
```

Check that the extractor can find CS2 and Source2Viewer:

```powershell
csdemo-mapextractor doctor `
  --cs2-dir "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive" `
  --source2viewer "C:\tools\Source2Viewer-CLI.exe"
```

## Extract a map

Extract collision geometry:

```powershell
csdemo-mapextractor extract `
  --cs2-dir "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive" `
  --source2viewer "C:\tools\Source2Viewer-CLI.exe" `
  --map de_mirage `
  --profile visual-occluders-v1 `
  --meshopt `
  --output out\de_mirage.glb
```

Extract radar images, overview metadata, and logos:

```powershell
csdemo-mapextractor extract-assets `
  --cs2-dir "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive" `
  --source2viewer "C:\tools\Source2Viewer-CLI.exe" `
  --maps de_mirage de_nuke
```

Visual assets are written to `out\<map>\`. Use `--force` to replace existing
files.

The CLI also includes `convert`, `inspect`, `validate`, and `compare` commands
for working with GLB and legacy `.tri` files.

## Releases

`build-release` packages every `de_*` map (or the maps passed with `--maps`)
into a content-addressed release:

```text
index.json                                   root pointer (short cache)
v1/release-<client>-<build>-<sha12>.json     release manifest (immutable)
v1/objects/<sha256>.<ext>                    collision, coloured collision, overview, logo, radars (immutable)
```

Manifest paths are relative to the manifest (`objects/<sha256>.glb`). Unchanged
files keep the same object key between releases, so `publish-r2` uploads only
missing objects, then the manifest, then `index.json`. After publishing, it
deletes every key under `v1/` that isn't referenced by the newest `--keep`
releases (default 2, the new release included).

## Coloured maps

Collision meshes carry no textures. `files.colored` is the same collision
geometry, remeshed so colour can vary across large faces, with a per-vertex
`COLOR_0` (linear RGBA) taken from the albedo of the textured render world
under each vertex. Clients should prefer it and fall back to `files.collision`
when a map has no coloured GLB.

Colouring runs in two stages, so the slow part runs without CS2 installed:

1. `build-release --color-bundles DIR` also exports each map's render world
   (`world.vwrld_c` as glTF with materials) and writes `DIR/<map>/`: the
   collision GLB, the render world reduced to what colouring reads, and the
   sampled textures as PNGs. The map record gets `color_source_sha256`, a
   digest of the bundle and the colouring code. If a map's render world can't
   be exported, the map is released without a bundle. Maps in
   `COLOR_EXCLUDED_MAPS` (`extractor/release.py`; currently `de_boulder`, too
   large for a CI runner) are never bundled and always ship uncoloured.
2. `colorize --bundle DIR/<map> --output colored/<map>.glb` builds the coloured
   GLB and a `<map>.json` sidecar naming its source digest. It needs the
   `color` extra:

   ```powershell
   python -m pip install -e ".[color]"
   ```

`color-plan --release-dir out\release --published-index <url>/index.json`
lists the maps whose digest differs from the published coloured GLB, and
`attach-colored` adds `files.colored` to the release, using freshly built GLBs
or reusing the published object when the digest is unchanged. Maps with
neither are released uncoloured.

The publish workflow builds the bundles, colours the planned maps in a matrix
job (one runner per map) and attaches the results before publishing. Run it
with `recolor` to colour every map regardless of what's published.

## CS2 render packages

`cs2render` builds the compact package that csdemo-client's `CS2` map mode renders. It
contains:

- meshopt-compressed geometry (`geo/`), each chunk with a simplified far LOD index list
  (`--lod-error`, in world units; 0 disables it)
- block-compressed (BC) texture arrays and baked lightmaps (`tex/`). Desktop GPUs sample them
  natively; clients without the BC WebGL extensions (phones) use the classified map
- the environment cubemaps and light-probe volumes
- the colour-correction LUT
- `visibility.bin`

`visibility.bin` maps the map's baked voxel PVS onto 3D cells. Every triangle is assigned to a
cell, so the client can skip hidden index ranges. The command needs Source2Viewer-CLI and
[texconv](https://github.com/microsoft/DirectXTex):

```powershell
csdemo-mapextractor cs2render --cs2-dir "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive" `
  --map de_inferno --texconv "$env:LOCALAPPDATA\csdemo-tools\texconv.exe"
```

The package is written to `out\cs2render\<map>\package`. `--geometry-only` rebuilds just the
geometry and `visibility.bin` of an existing package. `--sky-only` rebuilds just `tex/sky.ctex`,
the map's `env_sky` panorama. That file is a 2048×640 BC6H latlong with the sky's brightness and
yaw baked in, 0.2–1.1 MB gzipped (see `extractor/cs2render/sky.py`). Every binary file is stored gzipped
(manifest version 5, `compression: "gzip"`) so a static host serves it compressed; `--no-compress`
writes raw files. A package is about 45–150 MB (Mirage is the smallest, Ancient the largest), and
the client downloads all of it. The build takes 3–12 minutes once the export is cached.

The package also includes `lightgrid.bin`, which lights dynamic objects such as weapons. It
resamples the map's light-probe volumes into small uint8 grids:

- a six-direction ambient cube: luminance every 64×64×48 units, colour every 128×128×96
- the baked sun visibility every 32 units

The file is gzipped, about 1.4 MB for Inferno. The format is described in
`extractor/cs2render/lightgrid.py`.

The package also holds `breaks/<model>.glb`, the break pieces of the map's doors, windows and vents.
Their objects in `map.json` carry `breaks` (the file, `glass`, and `tint`, the entity's linear
`rendercolor`). The pieces are placed in their parent model's frame, decimated, and baked with
vertex-colour materials, including the `csgo_complex` tint mask. Building them needs Node and
`npm install --prefix weapon-bake`. `--breaks-only` rebuilds just the pieces and the `breaks`
fields of an existing package. See `extractor/cs2render/breakables.py` and
`weapon-bake/bake-break-pieces.mjs`.

## CS2 weapon models

`cs2weapons` bakes the weapon and player models that csdemo-client draws:

| Output | Contents |
|---|---|
| `world/<name>.glb` | Dropped weapons, grenades in flight and the weapons players hold. 42 reduced world models with per-vertex linear albedo and AO/roughness/metalness, meshopt-compressed, about 1.2 MB in total. Node extras `weaponBone` hold the attach point for a player's hand. |
| `fp/<name>.glb` and `fp/arms.glb` | First-person viewmodels. 44 skinned weapons plus the arms, with the game's AnimGraph2 clips and WebP colour, ORM and normal maps, about 19 MB in total. The client loads them one weapon at a time. |
| `characters/ct.glb`, `t.glb` and `anims.bin` | Third-person players for the client's CS2 look: the SAS and Phoenix agents (skinned, half the triangles, 1024 px WebP maps; 0.7 and 1.0 MB) and the shared third-person clips (about 0.8 MB). |

Install the Node bake scripts once, then run the bake:

```powershell
npm install --prefix weapon-bake
csdemo-mapextractor cs2weapons --cs2-dir "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive"
# Bake only some models, or only some of the sets (world, fp, characters):
#   --only ak47 glock arms   --set world   --target-verts 15%   --texture-size 512
#   --set characters   --character-texture-size 1024   --character-ratio 0.5
```

The command writes to `out\cs2weapons` and caches Source2Viewer exports in
`out\cs2weapons-work`. Host the output as the client's `VITE_CS2_WEAPONS_URL`, or copy it
to the client's `public/cs2weapons/` for development.

## CS2 sounds

`cs2sounds` bakes the game audio that csdemo-client plays in its CS2 view. Every sound is
re-encoded to Ogg Opus, 48 kHz mono, and concatenated into banks the client slices by offset:

| Output | Contents |
|---|---|
| `sounds.json` | The sound table (`[bank, offset, length]`), the soundevents the client plays with their volume, pitch, distance and fade curves, instance limits, delays and child events, the events the client starts from demo data, each weapon's shot event and its first-person foley (sound events of the viewmodel clips, as fractions of the clip). |
| `core.bin` | Grenades, the bomb, hits and deaths, bullet cracks, breaks, doors, knife swings, jump and gear sounds, the announcer's plant, defuse and round-win lines, and foley several weapons share (about 1.6 MB at 48 kbps). Always loaded. |
| `weapon_<name>.bin` | One weapon's shot (with its distant layer) and its own foley, 7–80 KB. Loaded for the weapons a demo uses. |
| `maps/<map>.json`, `<map>.bin` | The map's soundscapes (`env_soundscape`), point sounds (`point_soundevent`), and the footsteps, landings and bullet impacts of the surfaces the map has (about 1 MB on Inferno). |
| `maps/<map>_ambience.bin` | Soundscape beds and music at 16 kbps (`--ambience-kbps`), 0.75–2.7 MB per map. The client fetches one sound at a time with Range requests when a soundscape first needs it. Events come from `soundevents/ambience/game_sounds_<map>` and from the files named by the map's soundscape prefixes (Ancient plays `aztec.*`). |
| `maps/<map>_surfaces.bin` | A gzip-compressed footstep-surface grid baked from the map's collision: 32-unit cells, with every walkable floor layer's height and surface (about 60 KB). |

The bake needs an `ffmpeg` with libopus (`--ffmpeg` or the `FFMPEG` environment variable):

```powershell
csdemo-mapextractor cs2sounds --cs2-dir "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive" --maps de_inferno de_ancient de_anubis de_cache de_dust2 de_mirage de_nuke de_overpass
```

Bake every hosted map in one run: `sounds.json` and the global banks are rewritten each time.

The command writes to `out\cs2sounds` and caches decoded and encoded sounds in
`out\cs2sounds-work`. Host the output as the client's `VITE_CS2_SOUNDS_URL`, or copy it to
the client's `public/cs2sounds/` for development.

## License

The extractor is available under the [MIT License](LICENSE). Counter-Strike 2
and its assets belong to Valve; this license does not grant rights to
redistribute Valve's content.
