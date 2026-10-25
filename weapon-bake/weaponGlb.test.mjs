import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NodeIO } from '@gltf-transform/core';
import { EXTMeshoptCompression, KHRMeshQuantization } from '@gltf-transform/extensions';
import { MeshoptDecoder } from 'meshoptimizer';
import { encodeWeaponGlb } from './weapon-glb.mjs';

/** Open octagonal tube (ring vertices first) capped by a square plate whose vertices are not rings. */
function bakedMesh() {
  const positions = [];
  for (const x of [0, 10]) {
    for (let i = 0; i < 8; i++) positions.push(x, Math.cos((i * Math.PI) / 4), Math.sin((i * Math.PI) / 4));
  }
  positions.push(12, -2, -2, 12, 2, -2, 12, 2, 2, 12, -2, 2);
  const indices = [];
  for (let i = 0; i < 8; i++) {
    const j = (i + 1) % 8;
    indices.push(i, j, 8 + j, i, 8 + j, 8 + i);
  }
  indices.push(16, 17, 18, 16, 18, 19);
  const colors = positions.map((_, i) => (i % 3) / 3);
  return { positions: Float32Array.from(positions), colors: Float32Array.from(colors), indices: Uint32Array.from(indices), ringVertexCount: 16 };
}

async function decode(bytes) {
  await MeshoptDecoder.ready;
  const io = new NodeIO()
    .registerExtensions([KHRMeshQuantization, EXTMeshoptCompression])
    .registerDependencies({ 'meshopt.decoder': MeshoptDecoder });
  const doc = await io.readBinary(bytes);
  const node = doc.getRoot().listNodes()[0];
  const primitive = node.getMesh().listPrimitives()[0];
  const matrix = node.getWorldMatrix();
  const position = primitive.getAttribute('POSITION');
  const positions = [];
  for (let i = 0; i < position.getCount(); i++) {
    const [x, y, z] = position.getElement(i, []);
    positions.push([
      matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
      matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
      matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
    ]);
  }
  return { doc, primitive, positions };
}

test('meshopt output keeps tube-ring vertices identifiable through ringRanges', async () => {
  const bytes = await encodeWeaponGlb(bakedMesh(), 'tube', { meshopt: true });
  const { doc, primitive, positions } = await decode(bytes);
  assert.ok(doc.getRoot().listExtensionsUsed().some((e) => e.extensionName === 'EXT_meshopt_compression'));
  assert.equal(positions.length, 20);
  assert.equal(primitive.getIndices().getCount(), 54);
  const ranges = primitive.getExtras().ringRanges;
  const ring = new Set();
  for (let r = 0; r < ranges.length; r += 2) for (let v = ranges[r]; v < ranges[r] + ranges[r + 1]; v++) ring.add(v);
  assert.equal(ring.size, 16);
  for (let v = 0; v < positions.length; v++) {
    // Tube vertices sit at x = 0 or 10; plate vertices at x = 12.
    assert.equal(ring.has(v), positions[v][0] < 11, `vertex ${v} at x=${positions[v][0]}`);
  }
});

test('without meshopt the ring vertices stay first and are recorded as one range', async () => {
  const { primitive, positions } = await decode(await encodeWeaponGlb(bakedMesh(), 'tube'));
  assert.deepEqual(primitive.getExtras().ringRanges, [0, 16]);
  assert.ok(Math.abs(positions[16][0] - 12) < 0.01);
});
