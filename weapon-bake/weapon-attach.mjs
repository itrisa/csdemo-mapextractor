// Records where a weapon world model attaches to a player's hand: the model's root `weapon` bone, which
// CS2 bone-merges onto the player's weapon bone (`wpn`). Driven by `python -m extractor.cli cs2weapons`.
//
// Usage: node weapon-attach.mjs <skeleton-export.glb> <baked.glb>
//
// - <skeleton-export.glb> is a Source2Viewer-CLI export with --gltf_export_animations (that is what makes it
//   write the skeleton); <baked.glb> is bake-weapon-model.mjs's output, patched in place.
// - The bone's bind pose is written as node extras `weaponBone: { position, quaternion }` in the baked
//   model's frame (Source inches, X = barrel, Z = up). The client places held weapons at the inverse of it
//   relative to the player's `wpn` bone. Only the JSON chunk is rewritten; the mesh bytes are unchanged.
// - Models without a `weapon` bone are left as they are.
import { NodeIO } from '@gltf-transform/core';
import { readFile, writeFile } from 'node:fs/promises';

const [skeletonFile, bakedFile] = process.argv.slice(2);
if (!skeletonFile || !bakedFile) {
  console.error('Usage: node weapon-attach.mjs <skeleton-export.glb> <baked.glb>');
  process.exit(1);
}
const METRES_TO_INCHES = 1 / 0.0254;

const skeleton = await new NodeIO().read(skeletonFile);
const bone = skeleton.getRoot().listNodes().find((node) => node.getName() === 'weapon');
if (!bone) {
  console.log(`${bakedFile}: no weapon bone in ${skeletonFile}, left as is`);
  process.exit(0);
}

// glTF metres, Y up → Source inches (x, y, z) = (gltf z, gltf x, gltf y), as in bake-weapon-model.mjs.
const m = bone.getWorldMatrix(); // column-major
const toSource = (x, y, z) => [z, x, y];
const position = toSource(m[12], m[13], m[14]).map((v) => round(v * METRES_TO_INCHES));
// Columns are the bone's local axes in glTF; the bone's own axes already follow Source's convention.
const axes = [0, 1, 2].map((c) => toSource(m[c * 4], m[c * 4 + 1], m[c * 4 + 2]));
const quaternion = quaternionFromAxes(axes).map(round);

const glb = await readFile(bakedFile);
const jsonLength = glb.readUInt32LE(12);
const json = JSON.parse(glb.subarray(20, 20 + jsonLength).toString());
const node = json.nodes.find((n) => n.mesh !== undefined);
node.extras = { ...node.extras, weaponBone: { position, quaternion } };
let text = JSON.stringify(json);
text += ' '.repeat((4 - (Buffer.byteLength(text) % 4)) % 4);
const jsonChunk = Buffer.from(text);
const rest = glb.subarray(20 + jsonLength);
const header = Buffer.alloc(20);
header.write('glTF', 0);
header.writeUInt32LE(2, 4);
header.writeUInt32LE(20 + jsonChunk.length + rest.length, 8);
header.writeUInt32LE(jsonChunk.length, 12);
header.writeUInt32LE(0x4e4f534a, 16); // JSON
await writeFile(bakedFile, Buffer.concat([header, jsonChunk, rest]));
console.log(`${bakedFile}: weapon bone at [${position.join(', ')}] in, rotation [${quaternion.join(', ')}]`);

function round(value) {
  return Math.round(value * 1e4) / 1e4 + 0; // + 0 drops -0
}

/** Unit quaternion [x, y, z, w] of the rotation whose columns are `axes` (normalized first). */
function quaternionFromAxes(axes) {
  const [x, y, z] = axes.map((axis) => {
    const length = Math.hypot(...axis);
    return axis.map((v) => v / length);
  });
  const m00 = x[0], m10 = x[1], m20 = x[2];
  const m01 = y[0], m11 = y[1], m21 = y[2];
  const m02 = z[0], m12 = z[1], m22 = z[2];
  const trace = m00 + m11 + m22;
  let q;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    q = [(m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s, 0.25 / s];
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    q = [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s];
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    q = [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s];
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    q = [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s];
  }
  const length = Math.hypot(...q);
  return q.map((v) => v / length);
}
