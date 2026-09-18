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

## License

The extractor is available under the [MIT License](LICENSE). Counter-Strike 2
and its assets belong to Valve; this license does not grant rights to
redistribute Valve's content.
