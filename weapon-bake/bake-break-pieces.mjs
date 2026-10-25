// Bakes a breakable model's break pieces (each exported by Source2Viewer-CLI as GLB) into one small GLB for the
// client's breakables (csdemo-client src/scene/mapBreaks.ts). Driven by extractor/cs2render/breakables.py.
//
// Usage: node bake-break-pieces.mjs <spec.json> <out.glb>
//   spec: { pieces: [{ file, static, glass, offset: [x, y, z], angles: [pitch, yaw, roll], placement }],
//           tintMasks: { <vmat name>: { file: <png>, amount: g_flModelTintAmount } } }
//
// - Every piece becomes one node + mesh in the parent model's frame (Source inches, Z-up): glTF metres / Y-up are
//   converted like the weapon bake, then the piece's `offset` / `offset_rotation` is applied. Generated pieces are
//   re-centred and carry the parent's frame as a `placementOrigin` attachment (`placement: { origin, angles }`),
//   undone first: v -> R(angles)^T (v - origin).
// - Base color (linear) and ORM are sampled per vertex (COLOR_0 + `_ORM`, unorm8, like the dropped weapons); the
//   client computes creased normals. Untextured pieces (glass) keep their factors. COLOR_0's alpha is the tint mask
//   (x g_flModelTintAmount) of tintable materials, 0 elsewhere: the client multiplies the albedo by
//   mix(1, rendercolor, mask) per object like csgo_complex.
// - Welded by position and simplified with meshoptimizer (frozen remainders keep more detail: their jagged edge is
//   what shows).
// - Node extras `{ static, glass }`: static pieces stay in the frame (spawn_motion_disabled), the rest falls.
import { Document, Logger, NodeIO } from '@gltf-transform/core';
import { EXTMeshoptCompression, KHRMeshQuantization } from '@gltf-transform/extensions';
import { quantize } from '@gltf-transform/functions';
import { readFile, writeFile } from 'node:fs/promises';
import { MeshoptEncoder, MeshoptSimplifier } from 'meshoptimizer';
import sharp from 'sharp';
import { reorderForCompression } from './weapon-glb.mjs';

const [specPath, output] = process.argv.slice(2);
if (!specPath || !output) {
  console.error('Usage: node bake-break-pieces.mjs <spec.json> <out.glb>');
  process.exit(1);
}
await MeshoptEncoder.ready;
await MeshoptSimplifier.ready;

const METRES_TO_INCHES = 1 / 0.0254;
const WELD_EPSILON = 1e-3;
const COLOR_SAMPLE_SIZE = 64;
const CHANNELS = 7;
const NEUTRAL_ORM = [1, 0.6, 0];
const MAX_TRIANGLES = { debris: 500, static: 1500 };
const TARGET_ERROR = 0.004; // relative to the piece's extent
const COLOR_WEIGHT = 0.05;

const spec = JSON.parse(await readFile(specPath, 'utf8'));
const io = new NodeIO();

const textureCache = new Map();
function texture(tex) {
  if (!textureCache.has(tex)) {
    textureCache.set(tex, sharp(typeof tex === 'string' ? tex : Buffer.from(tex.getImage())).removeAlpha()
      .resize(COLOR_SAMPLE_SIZE, COLOR_SAMPLE_SIZE, { kernel: 'cubic' }).raw().toBuffer({ resolveWithObject: true }));
  }
  return textureCache.get(tex);
}

function sampleBilinear({ data: pixels, info }, u, v, out) {
  const w = info.width;
  const h = info.height;
  const x = (((u % 1) + 1) % 1) * w - 0.5;
  const y = (((v % 1) + 1) % 1) * h - 0.5;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  for (let c = 0; c < 3; c++) {
    let acc = 0;
    for (const [dx, dy, wt] of [[0, 0, (1 - fx) * (1 - fy)], [1, 0, fx * (1 - fy)], [0, 1, (1 - fx) * fy], [1, 1, fx * fy]]) {
      const px = (((x0 + dx) % w) + w) % w;
      const py = (((y0 + dy) % h) + h) % h;
      acc += pixels[(py * w + px) * 3 + c] * wt;
    }
    out[c] = acc / 255;
  }
}

/** Source AngleMatrix (pitch, yaw, roll in degrees), row-major. */
function angleMatrix([pitch, yaw, roll]) {
  const r = Math.PI / 180;
  const [sp, cp, sy, cy, sr, cr] = [Math.sin(pitch * r), Math.cos(pitch * r), Math.sin(yaw * r), Math.cos(yaw * r),
    Math.sin(roll * r), Math.cos(roll * r)];
  return [
    cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy,
    cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy,
    -sp, sr * cp, cr * cp,
  ];
}

async function bakePiece({ file, static: frozen, offset = [0, 0, 0], angles = [0, 0, 0], placement = null }) {
  const source = await io.read(file);
  const m = angleMatrix(angles);
  const po = placement?.origin ?? [0, 0, 0];
  const pr = angleMatrix(placement?.angles ?? [0, 0, 0]);
  const keyToIndex = new Map();
  const positions = [];
  const sums = [];
  const counts = [];
  const indices = [];
  const surface = [0, 0, 0];
  const orm = [0, 0, 0];
  let sourceTriangles = 0;
  for (const node of source.getRoot().listNodes()) {
    const mesh = node.getMesh();
    if (!mesh || /_lod[1-9]/i.test(node.getName())) continue;
    for (const prim of mesh.listPrimitives()) {
      const material = prim.getMaterial();
      const baseTexture = material?.getBaseColorTexture();
      const ormTexture = material?.getMetallicRoughnessTexture();
      const base = baseTexture ? await texture(baseTexture) : null;
      const ormImage = ormTexture ? await texture(ormTexture) : null;
      const factor = material?.getBaseColorFactor() ?? [1, 1, 1, 1];
      const tint = spec.tintMasks?.[material?.getExtras()?.vmat?.Name];
      const tintImage = tint ? await texture(tint.file) : null;
      const gltf = prim.getAttribute('POSITION').getArray();
      const uvs = prim.getAttribute('TEXCOORD_0')?.getArray();
      const sourceIndices = prim.getIndices()?.getArray() ?? Uint32Array.from({ length: gltf.length / 3 }, (_, i) => i);
      const remap = new Uint32Array(gltf.length / 3);
      for (let i = 0; i < remap.length; i++) {
        const px = gltf[i * 3 + 2] * METRES_TO_INCHES - po[0];
        const py = gltf[i * 3] * METRES_TO_INCHES - po[1];
        const pz = gltf[i * 3 + 1] * METRES_TO_INCHES - po[2];
        const lx = pr[0] * px + pr[3] * py + pr[6] * pz;
        const ly = pr[1] * px + pr[4] * py + pr[7] * pz;
        const lz = pr[2] * px + pr[5] * py + pr[8] * pz;
        const sx = m[0] * lx + m[1] * ly + m[2] * lz + offset[0];
        const sy = m[3] * lx + m[4] * ly + m[5] * lz + offset[1];
        const sz = m[6] * lx + m[7] * ly + m[8] * lz + offset[2];
        const key = `${Math.round(sx / WELD_EPSILON)},${Math.round(sy / WELD_EPSILON)},${Math.round(sz / WELD_EPSILON)}`;
        let index = keyToIndex.get(key);
        if (index === undefined) {
          index = positions.length / 3;
          keyToIndex.set(key, index);
          positions.push(sx, sy, sz);
          for (let c = 0; c < CHANNELS; c++) sums.push(0);
          counts.push(0);
        }
        if (base && uvs) sampleBilinear(base, uvs[i * 2], uvs[i * 2 + 1], surface);
        else surface.fill(1);
        for (let c = 0; c < 3; c++) sums[index * CHANNELS + c] += Math.pow(surface[c], 2.2) * factor[c];
        if (ormImage && uvs) sampleBilinear(ormImage, uvs[i * 2], uvs[i * 2 + 1], orm);
        for (let c = 0; c < 3; c++) sums[index * CHANNELS + 3 + c] += ormImage && uvs ? orm[c] : NEUTRAL_ORM[c];
        if (tintImage && uvs) {
          sampleBilinear(tintImage, uvs[i * 2], uvs[i * 2 + 1], orm);
          sums[index * CHANNELS + 6] += orm[0] * tint.amount;
        }
        counts[index]++;
        remap[i] = index;
      }
      for (let i = 0; i < sourceIndices.length; i += 3) {
        const a = remap[sourceIndices[i]];
        const b = remap[sourceIndices[i + 1]];
        const c = remap[sourceIndices[i + 2]];
        if (a !== b && b !== c && a !== c) indices.push(a, b, c);
      }
      sourceTriangles += sourceIndices.length / 3;
    }
  }
  if (indices.length === 0) return null;
  const vertexCount = positions.length / 3;
  const weldedPositions = Float32Array.from(positions);
  const colors = new Float32Array(vertexCount * CHANNELS);
  for (let v = 0; v < vertexCount; v++) for (let c = 0; c < CHANNELS; c++) colors[v * CHANNELS + c] = sums[v * CHANNELS + c] / counts[v];

  const cap = frozen ? MAX_TRIANGLES.static : MAX_TRIANGLES.debris;
  const [simplified] = MeshoptSimplifier.simplifyWithAttributes(
    Uint32Array.from(indices), weldedPositions, 3, colors, CHANNELS,
    [COLOR_WEIGHT, COLOR_WEIGHT, COLOR_WEIGHT, COLOR_WEIGHT * 0.5, COLOR_WEIGHT, COLOR_WEIGHT, COLOR_WEIGHT],
    null, Math.min(cap * 3, indices.length), TARGET_ERROR, [],
  );
  // Compact to the used vertices.
  const used = new Int32Array(vertexCount).fill(-1);
  const outPositions = [];
  const outColors = [];
  const outIndices = new Uint32Array(simplified.length);
  for (let i = 0; i < simplified.length; i++) {
    const v = simplified[i];
    if (used[v] < 0) {
      used[v] = outPositions.length / 3;
      outPositions.push(weldedPositions[v * 3], weldedPositions[v * 3 + 1], weldedPositions[v * 3 + 2]);
      for (let c = 0; c < CHANNELS; c++) outColors.push(colors[v * CHANNELS + c]);
    }
    outIndices[i] = used[v];
  }
  return {
    ...reorderForCompression({
      positions: Float32Array.from(outPositions), colors: Float32Array.from(outColors), normals: null, indices: outIndices,
      ringVertexCount: 0, colorStride: CHANNELS,
    }),
    sourceTriangles,
  };
}

const doc = new Document().setLogger(new Logger(Logger.Verbosity.WARN));
const buffer = doc.createBuffer();
const scene = doc.createScene();
const unorm8 = (x) => Math.round(Math.min(1, Math.max(0, x)) * 255);
const summary = [];
for (const piece of spec.pieces) {
  const baked = await bakePiece(piece);
  const name = piece.file.split(/[\\/]/).pop().replace(/\.glb$/, '');
  if (!baked) {
    console.warn(`${name}: no geometry, skipped`);
    continue;
  }
  const { positions, colors, indices } = baked;
  const count = positions.length / 3;
  const color8 = new Uint8Array(count * 4);
  const orm8 = new Uint8Array(count * 3);
  for (let v = 0; v < count; v++) {
    for (let c = 0; c < 3; c++) {
      color8[v * 4 + c] = unorm8(colors[v * CHANNELS + c]);
      orm8[v * 3 + c] = unorm8(colors[v * CHANNELS + 3 + c]);
    }
    color8[v * 4 + 3] = unorm8(colors[v * CHANNELS + 6]);
  }
  const primitive = doc.createPrimitive()
    .setAttribute('POSITION', doc.createAccessor().setType('VEC3').setArray(positions).setBuffer(buffer))
    .setAttribute('COLOR_0', doc.createAccessor().setType('VEC4').setArray(color8).setNormalized(true).setBuffer(buffer))
    .setAttribute('_ORM', doc.createAccessor().setType('VEC3').setArray(orm8).setNormalized(true).setBuffer(buffer))
    .setIndices(doc.createAccessor().setType('SCALAR')
      .setArray(count <= 65535 ? Uint16Array.from(indices) : Uint32Array.from(indices)).setBuffer(buffer));
  scene.addChild(doc.createNode(name).setMesh(doc.createMesh(name).addPrimitive(primitive))
    .setExtras({ static: !!piece.static, glass: !!piece.glass }));
  summary.push(`${name}${piece.static ? ' (static)' : ''}: ${baked.sourceTriangles} -> ${indices.length / 3} tris`);
}
if (summary.length === 0) throw new Error('No piece geometry');
await doc.transform(quantize({ quantizePosition: 16, quantizeColor: 8 }));
doc.createExtension(EXTMeshoptCompression).setRequired(true)
  .setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.QUANTIZE });
const ioOut = new NodeIO().registerExtensions([KHRMeshQuantization, EXTMeshoptCompression])
  .registerDependencies({ 'meshopt.encoder': MeshoptEncoder });
const bytes = await ioOut.writeBinary(doc);
await writeFile(output, bytes);
console.log(`${output.split(/[\\/]/).pop()}: ${summary.join(', ')}; ${(bytes.length / 1024).toFixed(1)} KB`);
