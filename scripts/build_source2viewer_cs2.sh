#!/usr/bin/env bash
# Builds the Source2Viewer CLI used by the cs2render, cs2weapons and cs2sounds bakes. They rely on export
# fixes that are on upstream master but not yet in a release (glTF LoD filtering, animation and physics
# export, CollisionGroup node extras, KV3 value types 22/26/27 used by current CS2 builds), so the CLI
# is built from a pinned commit of the itrisa fork.
# Requires the .NET 10 SDK.
set -euo pipefail

readonly REPOSITORY="https://github.com/itrisa/ValveResourceFormat.git"
readonly COMMIT="d4066afda916cde78f39146c589f1187a0efad2c"
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly SOURCE_DIR="${ROOT}/.work/ValveResourceFormat"
readonly TOOLS_DIR="${ROOT}/tools/source2viewer-cs2"

rm -rf "${SOURCE_DIR}" "${TOOLS_DIR}"
git init --quiet "${SOURCE_DIR}"
git -C "${SOURCE_DIR}" fetch --quiet --depth 1 "${REPOSITORY}" "${COMMIT}"
git -C "${SOURCE_DIR}" checkout --quiet FETCH_HEAD
test "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" = "${COMMIT}"

dotnet publish "${SOURCE_DIR}/CLI/CLI.csproj" \
  --configuration Release \
  --self-contained \
  --runtime linux-x64 \
  -p:RunAnalyzersDuringBuild=false \
  --output "${TOOLS_DIR}"
rm -rf "${SOURCE_DIR}"
chmod +x "${TOOLS_DIR}/Source2Viewer-CLI"
"${TOOLS_DIR}/Source2Viewer-CLI" --version
