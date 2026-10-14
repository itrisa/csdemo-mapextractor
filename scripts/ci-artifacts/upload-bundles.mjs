// Uploads every colour bundle as its own artifact, so each colorize matrix job downloads only its map.
// actions/upload-artifact takes one fixed artifact per step; this runs from actions/github-script, whose
// environment carries the runtime token @actions/artifact needs.
import { readdirSync } from "node:fs";
import { join } from "node:path";
import { DefaultArtifactClient } from "@actions/artifact";

function listFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? listFiles(path) : [path];
  });
}

export default async function uploadBundles({ core, root, maps, prefix, retentionDays = 1 }) {
  const client = new DefaultArtifactClient();
  for (const map of maps) {
    const files = listFiles(join(root, map));
    if (files.length === 0) {
      throw new Error(`colour bundle for ${map} is empty`);
    }
    const { id, size } = await client.uploadArtifact(`${prefix}${map}`, files, root, { retentionDays });
    core.info(`${map}: artifact ${id}, ${(size / 2 ** 20).toFixed(1)} MiB`);
  }
}
