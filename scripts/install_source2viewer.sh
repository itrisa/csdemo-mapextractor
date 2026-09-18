#!/usr/bin/env bash
set -euo pipefail

readonly VERSION="20.0"
readonly SHA256="3e8af47cd6ce52e8068904f2aa1dda23c56a6b96a8310b25090f0711cda76a8a"
readonly URL="https://github.com/ValveResourceFormat/ValveResourceFormat/releases/download/${VERSION}/cli-linux-x64.zip"
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TOOLS_DIR="${ROOT}/tools"
readonly ARCHIVE="${TOOLS_DIR}/cli-linux-x64.zip"

mkdir -p "${TOOLS_DIR}"
curl --fail --location --retry 3 --output "${ARCHIVE}" "${URL}"
echo "${SHA256}  ${ARCHIVE}" | sha256sum --check -
unzip -oq "${ARCHIVE}" -d "${TOOLS_DIR}"
rm "${ARCHIVE}"
chmod +x "${TOOLS_DIR}/Source2Viewer-CLI"
"${TOOLS_DIR}/Source2Viewer-CLI" --version
