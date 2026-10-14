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

## License

The extractor is available under the [MIT License](LICENSE). Counter-Strike 2
and its assets belong to Valve; this license does not grant rights to
redistribute Valve's content.
