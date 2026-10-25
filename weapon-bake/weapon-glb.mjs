// GLB writer for baked weapon world models (bake-weapon-model.mjs).
import { Document, Logger, NodeIO } from '@gltf-transform/core';
import { EXTMeshoptCompression, KHRMeshQuantization } from '@gltf-transform/extensions';
import { quantize } from '@gltf-transform/functions';
import { MeshoptEncoder } from 'meshoptimizer';

await MeshoptEncoder.ready;
const io = new NodeIO()
  .registerExtensions([KHRMeshQuantization, EXTMeshoptCompression])
  .registerDependencies({ 'meshopt.encoder': MeshoptEncoder });

/**
 * Encodes a baked mesh (Source inches; per vertex linear 0..1 color, AO, roughness, metalness) as a GLB.
 * - Positions are quantized to 16 bits (KHR_mesh_quantization; the dequantizing scale/offset lands on
 *   the node, ~0.002 in precision at weapon size). Color is COLOR_0 and AO/roughness/metalness the
 *   custom `_ORM` attribute, both unorm8.
 * - `normals` is optional; without it the client computes creased normals at load
 *   (csdemo-client src/scene/weaponNormals.ts).
 * - `ringVertexCount` > 0 means input vertices [0, ringVertexCount) are tube-ring vertices. They are
 *   recorded as primitive extras `ringRanges` ([start, count, ...] in output vertex order) so the client
 *   keeps edges between them smooth up to a wider crease angle.
 * - `quantized: false` keeps float positions (for external tools such as Blender).
 * - `meshopt: true` reorders for the vertex cache and compresses with EXT_meshopt_compression
 *   (roughly halves the file; the client needs the meshopt decoder).
 */
export async function encodeWeaponGlb(
  { positions, colors, normals = null, indices, ringVertexCount = 0 },
  name,
  { quantized = true, meshopt = false } = {},
) {
  let ringRanges = ringVertexCount > 0 ? [0, ringVertexCount] : [];
  if (meshopt) {
    ({ positions, colors, normals, indices, ringRanges } = reorderForCompression({ positions, colors, normals, indices, ringVertexCount }));
  }
  const doc = new Document().setLogger(new Logger(Logger.Verbosity.WARN));
  const buffer = doc.createBuffer();
  // glTF COLOR_0 is linear, so the linear colors are stored directly as unorm8.
  const count = positions.length / 3;
  const color8 = new Uint8Array(count * 3);
  const orm8 = new Uint8Array(count * 3);
  const unorm8 = (x) => Math.round(Math.min(1, Math.max(0, x)) * 255);
  for (let v = 0; v < count; v++) {
    for (let c = 0; c < 3; c++) {
      color8[v * 3 + c] = unorm8(colors[v * 6 + c]);
      orm8[v * 3 + c] = unorm8(colors[v * 6 + 3 + c]);
    }
  }
  const primitive = doc.createPrimitive()
    .setAttribute('POSITION', doc.createAccessor().setType('VEC3').setArray(positions).setBuffer(buffer))
    .setAttribute('COLOR_0', doc.createAccessor().setType('VEC3').setArray(color8).setNormalized(true).setBuffer(buffer))
    .setAttribute('_ORM', doc.createAccessor().setType('VEC3').setArray(orm8).setNormalized(true).setBuffer(buffer))
    .setIndices(doc.createAccessor().setType('SCALAR')
      .setArray(positions.length / 3 <= 65535 ? Uint16Array.from(indices) : Uint32Array.from(indices)).setBuffer(buffer));
  if (normals) primitive.setAttribute('NORMAL', doc.createAccessor().setType('VEC3').setArray(normals).setBuffer(buffer));
  if (ringRanges.length > 0) primitive.setExtras({ ringRanges });
  const mesh = doc.createMesh(name).addPrimitive(primitive);
  doc.createScene().addChild(doc.createNode('weapon').setMesh(mesh));
  if (quantized) await doc.transform(quantize({ quantizePosition: 16, quantizeNormal: 8, quantizeColor: 8 }));
  if (meshopt) {
    doc.createExtension(EXTMeshoptCompression).setRequired(true)
      .setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.QUANTIZE });
  }
  return io.writeBinary(doc);
}

/**
 * meshopt's vertex-cache triangle order and first-use vertex order, which is what makes its index
 * and attribute codecs effective. Ring vertices end up scattered, so they are returned as ranges.
 */
export function reorderForCompression({ positions, colors, normals, indices, ringVertexCount, colorStride = 6 }) {
  const reordered = Uint32Array.from(indices);
  const [remap, unique] = MeshoptEncoder.reorderMesh(reordered, true, false);
  const permute = (array, size) => {
    if (!array) return array;
    const result = new array.constructor(unique * size);
    for (let v = 0; v < remap.length; v++) {
      if (remap[v] === 0xffffffff) continue;
      for (let k = 0; k < size; k++) result[remap[v] * size + k] = array[v * size + k];
    }
    return result;
  };
  const isRing = new Uint8Array(unique);
  for (let v = 0; v < ringVertexCount; v++) if (remap[v] !== 0xffffffff) isRing[remap[v]] = 1;
  const ringRanges = [];
  for (let v = 0; v < unique; v++) {
    if (!isRing[v]) continue;
    if (v > 0 && isRing[v - 1]) ringRanges[ringRanges.length - 1]++;
    else ringRanges.push(v, 1);
  }
  return {
    positions: permute(positions, 3), colors: permute(colors, colorStride), normals: permute(normals, 3), indices: reordered, ringRanges,
  };
}