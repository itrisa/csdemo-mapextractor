# csdemo-mapextractor

Produce deterministic collision geometry artifacts from a local Counter-Strike
2 installation. The current vertical slice provides:

- Source2Viewer-CLI discovery and version checks;
- `world_physics.vmdl_c` extraction from a map VPK;
- radar, overview metadata, and map-logo extraction from the main CS2 VPK;
- explicit `visual-occluders-v1` and `all-physics-v1` GLB selection profiles;
- exact float32 vertex welding and indexed geometry;
- minimal uncompressed GLB and legacy little-endian `.tri` output;
- a JSON manifest containing bounds, counts, profile, versions, and checksum.

Python 3.11 or newer is required. Install the package locally with:

```powershell
python -m pip install -e .
```

Install the optional lossless Meshopt encoder with:

```powershell
python -m pip install -e ".[meshopt]"
```

Check the local tools and CS2 layout:

```powershell
csdemo-mapextractor doctor `
	--cs2-dir "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive" `
	--source2viewer "C:\tools\Source2Viewer-CLI.exe"
```

Extract one map to either `.glb` or `.tri` based on the output extension:

```powershell
csdemo-mapextractor extract `
	--cs2-dir "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive" `
	--source2viewer "C:\tools\Source2Viewer-CLI.exe" `
	--map de_mirage `
	--profile visual-occluders-v1 `
	--meshopt `
	--output out\de_mirage.glb
```

Meshopt output keeps float32 positions and index sequences byte-exact. It uses
`ATTRIBUTES` and `INDICES` modes with no quantization, reordering, or filters.

Extract radar images, overview metadata, and map logos:

```powershell
csdemo-mapextractor extract-assets `
	--cs2-dir "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive" `
	--source2viewer "C:\tools\Source2Viewer-CLI.exe" `
	--maps de_mirage de_nuke
```

Assets are written to `out\<map>\` by default. Each directory contains
`radar.png`, `logo.svg`, and `overview.txt`. Maps with vertical sections also
contain files such as `radar_lower.png`. Use `--output-dir` to select another
root and `--force` to overwrite existing assets.

Build client caches containing default world collision and enabled ordinary
`func_brush` entities using:

```powershell
csdemo-mapextractor cache-maps `
	--cs2-dir "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive" `
	--source2viewer "C:\tools\Source2Viewer-CLI.exe" `
	--maps de_mirage de_inferno de_dust2 de_ancient `
	--output-dir out\maps
```

Pass `--all-layers` to retain every collision group for inspection.

Layered caches also include `func_brush` physics from each map's
`default_ents.vents_c`. Entity transforms are baked into Hammer Z-up world
coordinates before compression. Interaction-tag variants such as
`entity:func_brush:npcclip+playerclip` are separate layers, allowing large clip
volumes to be hidden without removing ordinary solid brushes.

The cache also reads the entity lump's `DATA` block and follows Source2Viewer's
default visibility rule: entities with `startdisabled=true` or `enabled=false`
are excluded from the cached `func_brush` layers.

Existing `.tri` and minimal `.glb` artifacts can be converted, inspected, and
validated with the `convert`, `inspect`, and `validate` commands. `compare`
reports exact triangle-multiset overlap between any two supported artifacts.

`awpy-default-v1` selects VRF physics meshes named `physics_group` or
`physics_group_<surface>`. VRF derives those names from PHYS collision groups;
the extractor rejects GLBs without this metadata rather than treating material
filtering as equivalent. The profile preserves Hammer coordinates and does not
apply VRF's node transform to glTF meters and Y-up.

## Automated releases

Build a complete, versioned browser release from every supported map VPK:

```powershell
csdemo-mapextractor build-release `
	--cs2-dir "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive" `
	--source2viewer "C:\tools\Source2Viewer-CLI.exe" `
	--output-dir out\release `
	--steam-build-id 25218825
```

Map names are discovered from `game\csgo\maps\*.vpk`. Names containing
`_preview`, `_vanity`, `lobby_`, or `graphics_` are excluded. Pass `--maps`
to build an explicit subset. The output layout is:

```text
out/release/
|-- index.json
`-- v1/<ClientVersion>-<SteamBuildId>/
    |-- index.json
    `-- maps/<map>/
        |-- collision.glb
        |-- logo.svg
        |-- overview.txt
        |-- radar.png
        `-- radar_<section>.png
```

The version manifest contains per-file sizes and SHA-256 checksums. Missing
optional visual assets are recorded per map. Files below
`v1/<ClientVersion>-<SteamBuildId>/` are immutable; the root `index.json`
identifies the current release.

Publish a completed release through Cloudflare R2's S3-compatible endpoint:

```powershell
$env:AWS_ACCESS_KEY_ID = "<R2 access key>"
$env:AWS_SECRET_ACCESS_KEY = "<R2 secret key>"
$env:AWS_DEFAULT_REGION = "auto"

csdemo-mapextractor publish-r2 `
	--release-dir out\release `
	--bucket csdemo-maps `
	--endpoint-url "https://<account-id>.r2.cloudflarestorage.com"
```

The publisher validates every manifest checksum, uploads and verifies every
versioned object, and updates the root index last. A matching partial upload is
safe to resume. Existing version manifests with different content are not
replaced unless `--force` is supplied.

The scheduled workflow in `.github/workflows/publish-map-assets.yml` obtains
CS2 anonymously through SteamCMD, skips builds already present in the public
root index, installs a checksum-pinned Linux Source2Viewer CLI, builds the
release, and publishes it to R2.

Configure these GitHub Actions secrets:

- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_ACCOUNT_ID`

Configure these GitHub Actions variables:

- `R2_BUCKET`, for example `csdemo-maps`
- `R2_PUBLIC_BASE_URL`, for example `https://maps.example.com`

The R2 token should have Object Read & Write access only to the target bucket.
Configure the bucket's custom domain and CORS policy separately in Cloudflare.