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

## Build a browser release

Build every supported map into a versioned release:

```powershell
csdemo-mapextractor build-release `
  --cs2-dir "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive" `
  --source2viewer "C:\tools\Source2Viewer-CLI.exe" `
  --steam-build-id 25218825
```

Maps are discovered automatically from the CS2 installation. The result looks
like this:

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

Each manifest includes file sizes and SHA-256 checksums. Version folders are
immutable; the root `index.json` points to the current release.

## Use the release from a client

The client only needs a stable public base URL. It first fetches the root
`index.json`, then follows its `manifest` field:

```ts
const MAP_ASSET_BASE_URL = "https://maps.example.com/";
const rootUrl = new URL("index.json", MAP_ASSET_BASE_URL);
const root = await fetch(rootUrl).then((response) => response.json());

const manifestUrl = new URL(root.manifest, rootUrl);
const release = await fetch(manifestUrl).then((response) => response.json());

const map = release.maps.find(({ name }) => name === "de_mirage");
const collisionUrl = new URL(map.files.collision.path, manifestUrl);
```

This keeps the client independent of Steam build IDs and version-folder names.

## Publish to Cloudflare R2

Set credentials for an R2 token limited to the target bucket:

```powershell
$env:AWS_ACCESS_KEY_ID = "<R2 access key>"
$env:AWS_SECRET_ACCESS_KEY = "<R2 secret key>"
$env:AWS_DEFAULT_REGION = "auto"
```

Publish the release:

```powershell
csdemo-mapextractor publish-r2 `
  --bucket csdemo-maps `
  --endpoint-url "https://<account-id>.r2.cloudflarestorage.com"
```

The publisher validates and verifies every file before replacing the root
index. Interrupted uploads can be resumed safely.

Configure a public custom domain and CORS policy for the bucket in Cloudflare.
The S3 endpoint above is only used for authenticated publication.

## GitHub Actions

[The publication workflow](.github/workflows/publish-map-assets.yml) checks for
new CS2 builds each day and can also be started manually. It downloads CS2,
builds the release, and publishes it to R2.

Add these repository secrets:

- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_ACCOUNT_ID`

Add one repository variable:

- `R2_BUCKET`

SteamCMD and extraction run without R2 credentials, and their dependencies are
hash-locked. Only a validated build artifact reaches the separate publication
job.

## License

The extractor is available under the [MIT License](LICENSE). Counter-Strike 2
and its assets belong to Valve; this license does not grant rights to
redistribute Valve's content.
