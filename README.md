# csdemo-mapextractor

Produce deterministic collision geometry artifacts from a local Counter-Strike
2 installation. The current vertical slice provides:

- Source2Viewer-CLI discovery and version checks;
- `world_physics.vmdl_c` extraction from a map VPK;
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