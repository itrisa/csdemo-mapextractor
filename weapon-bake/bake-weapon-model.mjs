// Bakes a CS2 weapon world model (exported by Source2Viewer-CLI as GLB) into a tiny, untextured
// low-poly GLB for the dropped-weapon renderer. Driven by `python -m extractor.cli cs2weapons`.
//
// Usage: node bake-weapon-model.mjs <vrf-export.glb> <out.glb> [targetTriangles] [colorWeight]
//          [--target-kb=<kb>] [--target-verts=<n|pct%>] [--method=quality|legacy] [--rings=false]
//          [--ring-share=0.2] [--normals] [--meshopt=false] [--full] [--color-size=128]
//
// - Keeps only the `body_hd` mesh (`body` for grenades, which have a single body group).
// - `--target-kb` replaces the triangle target with a file-size budget, and `--target-verts=<n|pct%>`
//   with a vertex budget (a percentage is of the prepared, welded mesh): the largest triangle count
//   that fits is found by bisection. `--full` skips reduction entirely.
// - Converts glTF metres / Y-up into Source inches / Z-up (X = barrel, Z = up), so the client can
//   pose it straight from the entity's QAngle with no extra transform.
// - Samples the base-color and ORM textures into per-vertex color + AO/roughness/metalness and welds
//   by position, so UV seams no longer block collapses.
// - Drops detached props (and, with --min-part-size=<in>, tiny parts), then simplifies with meshoptimizer:
//   `quality` (default) reduces tubes to clean locked rings first (weapon-rings.mjs, disable
//   with --rings=false) and then runs one global error-ordered pass; `legacy` simplifies each part
//   separately with an area share of the budget. Color is weighted so wood/metal/plastic boundaries
//   survive.
// - Normals are not stored by default; the client computes creased normals at load. `--normals` bakes
//   them in instead (more vertices: every hard edge splits them).
// - Output is meshopt-compressed (EXT_meshopt_compression) unless `--meshopt=false`; `--target-kb`
//   then measures the compressed file.
import { NodeIO } from '@gltf-transform/core';
import { writeFile } from 'node:fs/promises';
import { MeshoptSimplifier } from 'meshoptimizer';
import sharp from 'sharp';
import { encodeWeaponGlb } from './weapon-glb.mjs';
import { reduceRings } from './weapon-rings.mjs';

const flags = Object.fromEntries(
  process.argv.slice(2).filter((a) => a.startsWith('--')).map((a) => {
    const [key, value = 'true'] = a.slice(2).split('=');
    return [key, value];
  }),
);
const [input, output, targetArg = '400', weightArg] = process.argv.slice(2).filter((a) => !a.startsWith('--'));
if (!input || !output) {
  console.error(
    'Usage: node bake-weapon-model.mjs <vrf-export.glb> <out.glb> [targetTriangles] [colorWeight]\n'
    + '         [--target-kb=<kb>] [--target-verts=<n|pct%>] [--method=quality|legacy] [--rings=false]\n'
    + '         [--ring-share=0.2] [--normals] [--meshopt=false] [--full] [--color-size=128]\n'
    + '         [--crease=35] [--min-part-size=0] [--prepared=<unsimplified.glb>] [--body=<group>]',
  );
  process.exit(1);
}
const method = flags.method ?? 'quality';
if (method !== 'quality' && method !== 'legacy') throw new Error(`Unknown --method=${method}`);
const targetTriangles = Number(targetArg);
const colorWeight = Number(weightArg ?? (method === 'legacy' ? 0.5 : 0.05));
const creaseDegrees = Number(flags.crease ?? 35);
const METRES_TO_INCHES = 1 / 0.0254;
// The texture is downscaled first so per-vertex samples are pre-blurred; denser meshes can use more.
const COLOR_SAMPLE_SIZE = Number(flags['color-size'] ?? 128);
const WELD_EPSILON = 1e-3; // inches
// inches; connected parts whose longest side is shorter are dropped. Off by default: small parts are
// cheap after simplification, and some cover openings in the body (e.g. the AK dust-cover button).
const MIN_PART_SIZE = Number(flags['min-part-size'] ?? 0);
const MIN_PART_TRIANGLES = 6; // floor per kept part so thin parts keep at least a prism
const STRAY_GAP = 1; // inches; parts farther than this from the attached cluster are dropped

const targetBytes = flags['target-kb'] ? Number(flags['target-kb']) * 1024 : null;
const bakeNormals = flags.normals === 'true';
const useMeshopt = flags.meshopt !== 'false';
// Body group preference: guns ship `body_legacy` + `body_hd` (only `body_hd` is used), grenades only
// `body`, C4/Zeus only `body_hd`, and the default T knife only `body_legacy` (the last resort).
// `--body=<group>` forces one group (e.g. `--body=body_legacy`).
const BODY_GROUPS = flags.body ? [flags.body] : ['body_hd', 'body', 'body_legacy'];
// Decals and effect cards that aren't part of the solid weapon (sticker cut-outs, molotov flame card,
// Zeus charge meter, C4 screen).
const SKIP_MATERIALS = /sticker_gaps|flame|chargemete|panorama|digits/i;

const io = new NodeIO();
const source = await io.read(input);
const meshNodes = source.getRoot().listNodes().filter((n) => n.getMesh());
const node = BODY_GROUPS.map((group) => meshNodes.find((n) => n.getName().endsWith('.' + group))).find(Boolean);
if (!node) throw new Error(`No ${BODY_GROUPS.join('/')} mesh in ${input}`);
// Every textured material primitive is merged (e.g. SSG 08 body + scope, M249 body + box).
const prims = node.getMesh().listPrimitives()
  .filter((p) => p.getMaterial()?.getBaseColorTexture() && !SKIP_MATERIALS.test(p.getMaterial().getName()));
if (prims.length === 0) throw new Error('No textured primitives in ' + input);

// Per-vertex surface: linear base color, then the ORM texture's ambient occlusion, roughness and metalness,
// so the client can shade drops with the map's PBR lighting (sun, cubemaps, light grid).
const CHANNELS = 6;
const NEUTRAL_ORM = [1, 0.6, 0];
// Simplification weights per channel: color boundaries matter most, AO the least.
const attributeWeights = () => [colorWeight, colorWeight, colorWeight, colorWeight * 0.5, colorWeight, colorWeight];

async function loadTexture(texture) {
  return sharp(Buffer.from(texture.getImage()))
    .removeAlpha()
    .resize(COLOR_SAMPLE_SIZE, COLOR_SAMPLE_SIZE, { kernel: 'cubic' })
    .raw()
    .toBuffer({ resolveWithObject: true });
}
const textureCache = new Map();
function cachedTexture(texture) {
  if (!textureCache.has(texture)) textureCache.set(texture, loadTexture(texture));
  return textureCache.get(texture);
}

const colorSample = [0, 0, 0];
const ormSample = [0, 0, 0];
/** Linear base color (0..1) followed by AO, roughness and metalness. */
function sampleSurface({ baseColor, orm }, u, v, out) {
  sampleBilinear(baseColor, u, v, colorSample);
  for (let c = 0; c < 3; c++) out[c] = Math.pow(colorSample[c], 2.2);
  if (orm) sampleBilinear(orm, u, v, ormSample);
  for (let c = 0; c < 3; c++) out[3 + c] = orm ? ormSample[c] : NEUTRAL_ORM[c];
}

/** Bilinear sample of an 8-bit RGB image -> 0..1 (no color-space conversion), wrapping UVs. */
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

// Weld by quantized position (across all merged primitives), averaging the sampled colors of merged
// vertices.
const keyToIndex = new Map();
const positions = [];
const colorSums = [];
const colorCounts = [];
const surface = new Array(CHANNELS).fill(0);
const welded = [];
let sourceTriangles = 0;
for (const prim of prims) {
  const material = prim.getMaterial();
  const ormTexture = material.getMetallicRoughnessTexture();
  const textures = {
    baseColor: await cachedTexture(material.getBaseColorTexture()),
    orm: ormTexture ? await cachedTexture(ormTexture) : null,
  };
  const gltfPositions = prim.getAttribute('POSITION').getArray();
  const uvs = prim.getAttribute('TEXCOORD_0').getArray();
  const sourceIndices = prim.getIndices().getArray();
  const remap = new Uint32Array(gltfPositions.length / 3);
  for (let i = 0; i < remap.length; i++) {
    const sx = gltfPositions[i * 3 + 2] * METRES_TO_INCHES;
    const sy = gltfPositions[i * 3] * METRES_TO_INCHES;
    const sz = gltfPositions[i * 3 + 1] * METRES_TO_INCHES;
    const key = `${Math.round(sx / WELD_EPSILON)},${Math.round(sy / WELD_EPSILON)},${Math.round(sz / WELD_EPSILON)}`;
    let index = keyToIndex.get(key);
    if (index === undefined) {
      index = positions.length / 3;
      keyToIndex.set(key, index);
      positions.push(sx, sy, sz);
      for (let c = 0; c < CHANNELS; c++) colorSums.push(0);
      colorCounts.push(0);
    }
    sampleSurface(textures, uvs[i * 2], uvs[i * 2 + 1], surface);
    for (let c = 0; c < CHANNELS; c++) colorSums[index * CHANNELS + c] += surface[c];
    colorCounts[index]++;
    remap[i] = index;
  }
  for (let i = 0; i < sourceIndices.length; i += 3) {
    const a = remap[sourceIndices[i]];
    const b = remap[sourceIndices[i + 1]];
    const c = remap[sourceIndices[i + 2]];
    if (a !== b && b !== c && a !== c) welded.push(a, b, c);
  }
  sourceTriangles += sourceIndices.length / 3;
}
const vertexCount = positions.length / 3;
const weldedPositions = Float32Array.from(positions);
const colors = new Float32Array(vertexCount * CHANNELS);
for (let i = 0; i < vertexCount; i++) {
  for (let c = 0; c < CHANNELS; c++) colors[i * CHANNELS + c] = colorSums[i * CHANNELS + c] / colorCounts[i];
}

// Split into connected parts (barrel, magazine, stock, screws, ...). Detached props are dropped, and the
// legacy method simplifies each part on its own (a triangle floor plus an area-proportional share of
// the budget) because a global simplify used to collapse thin-but-long parts like the M4 barrel.
const parent = Int32Array.from({ length: vertexCount }, (_, i) => i);
const find = (x) => {
  while (parent[x] !== x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
};
for (let i = 0; i < welded.length; i += 3) {
  const a = find(welded[i]);
  parent[find(welded[i + 1])] = a;
  parent[find(welded[i + 2])] = a;
}
const partsByRoot = new Map();
for (let i = 0; i < welded.length; i += 3) {
  const root = find(welded[i]);
  let part = partsByRoot.get(root);
  if (!part) {
    part = { indices: [], area: 0, min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
    partsByRoot.set(root, part);
  }
  const tri = [welded[i], welded[i + 1], welded[i + 2]];
  part.indices.push(...tri);
  for (const v of tri) {
    for (let c = 0; c < 3; c++) {
      part.min[c] = Math.min(part.min[c], weldedPositions[v * 3 + c]);
      part.max[c] = Math.max(part.max[c], weldedPositions[v * 3 + c]);
    }
  }
  const p = tri.map((v) => [0, 1, 2].map((c) => weldedPositions[v * 3 + c]));
  const e1 = [0, 1, 2].map((c) => p[1][c] - p[0][c]);
  const e2 = [0, 1, 2].map((c) => p[2][c] - p[0][c]);
  part.area += 0.5 * Math.hypot(
    e1[1] * e2[2] - e1[2] * e2[1],
    e1[2] * e2[0] - e1[0] * e2[2],
    e1[0] * e2[1] - e1[1] * e2[0],
  );
}
const allParts = [...partsByRoot.values()];
// Detached props posed away from the weapon (R8 speedloader, XM1014 spare shells, molotov lighter,
// the second Dual Beretta) are dropped: starting from the largest part, keep only parts whose
// bounding box comes within STRAY_GAP of an already-kept part. Of parts about as large as the
// largest, the one nearest the origin (the weapon root) wins: the Dual Berettas' right pistol sits
// at the root and the left one 20 in to the side, and the client puts the kept one in each hand.
const near = (a, b) => [0, 1, 2].every((c) => a.min[c] - STRAY_GAP <= b.max[c] && b.min[c] - STRAY_GAP <= a.max[c]);
const largestArea = Math.max(...allParts.map((part) => part.area));
const originDistance = (part) => Math.hypot(...[0, 1, 2].map((c) => Math.max(part.min[c], 0, -part.max[c])));
const attached = [allParts
  .filter((part) => part.area >= largestArea * 0.99)
  .reduce((best, part) => (originDistance(part) < originDistance(best) ? part : best))];
for (let grew = true; grew;) {
  grew = false;
  for (const part of allParts) {
    if (!attached.includes(part) && attached.some((kept) => near(part, kept))) {
      attached.push(part);
      grew = true;
    }
  }
}
const parts = attached.filter((part) => Math.max(...part.max.map((x, c) => x - part.min[c])) >= MIN_PART_SIZE);
console.log(
  `parts: ${parts.length} kept, ${allParts.length - attached.length} detached, `
  + `${attached.length - parts.length} below ${MIN_PART_SIZE}in`,
);

await MeshoptSimplifier.ready;

const allIndices = Uint32Array.from(parts.flatMap((part) => part.indices));
const preparedTriangles = allIndices.length / 3;
const outputName = output.replace(/^.*[\\/]/, '').replace(/\.glb$/, '');
if (flags.prepared) {
  // The cleaned-up mesh before any reduction (welded, colored, detached props removed), e.g. as
  // input for an external decimator.
  const prepared = compact(allIndices, weldedPositions, colors);
  await writeFile(flags.prepared, await encodeWeaponGlb(prepared, 'prepared', { quantized: false }));
  console.log(`prepared: ${preparedTriangles} tris -> ${flags.prepared}`);
}

let result;
const preparedVertices = new Set(allIndices).size;
const targetVertices = flags['target-verts']
  ? Math.round(flags['target-verts'].endsWith('%')
    ? preparedVertices * Number(flags['target-verts'].slice(0, -1)) / 100
    : Number(flags['target-verts']))
  : null;
if (flags.full === 'true') {
  result = { mesh: compact(allIndices, weldedPositions, colors), error: 0 };
  result.bytes = await encodeWeaponGlb(result.mesh, outputName, { meshopt: useMeshopt });
} else if (targetBytes || targetVertices) {
  // Bisect the triangle target for the largest bake that fits the byte or vertex budget.
  const attempt = async (triangles) => {
    const baked = bake(triangles);
    return { ...baked, bytes: await encodeWeaponGlb(baked.mesh, outputName, { meshopt: useMeshopt }) };
  };
  const fits = (candidate) => (!targetBytes || candidate.bytes.length <= targetBytes)
    && (!targetVertices || candidate.mesh.positions.length / 3 <= targetVertices);
  result = await attempt(preparedTriangles);
  if (!fits(result)) {
    let lo = 1;
    let hi = preparedTriangles;
    result = null;
    while (hi - lo > Math.max(2, lo * 0.002)) {
      const mid = Math.floor((lo + hi) / 2);
      const candidate = await attempt(mid);
      if (fits(candidate)) {
        lo = mid;
        result = candidate;
      } else {
        hi = mid;
      }
    }
    result ??= await attempt(lo);
  }
} else {
  result = bake(targetTriangles);
  result.bytes = await encodeWeaponGlb(result.mesh, outputName, { meshopt: useMeshopt });
}
await writeFile(output, result.bytes);
if (result.note) console.log(result.note);

const errorLabel = method === 'legacy' ? `error ${(result.error * 100).toFixed(2)}%` : `error ${result.error.toFixed(3)}in`;
console.log(
  `${input.replace(/^.*[\\/]/, '')} [${method}]: ${sourceTriangles} tris (${preparedTriangles} prepared, `
  + `${preparedVertices} verts) -> `
  + `${result.mesh.indices.length / 3} tris, ${result.mesh.positions.length / 3} verts `
  + `(${(result.mesh.positions.length / 3 / preparedVertices * 100).toFixed(1)}%), `
  + `${(result.bytes.length / 1024).toFixed(1)} KB, ${errorLabel} -> ${output}`,
);

/** Simplifies the prepared mesh to about `targetTriangles`. */
function bake(targetTriangles) {
  if (method === 'legacy') {
    // Per-part simplify: each part gets a triangle floor plus an area-proportional budget share.
    const floorTotal = parts.length * MIN_PART_TRIANGLES;
    const shareable = Math.max(0, targetTriangles - floorTotal);
    const totalArea = parts.reduce((sum, part) => sum + part.area, 0);
    const simplifiedParts = [];
    let error = 0;
    for (const part of parts) {
      const budget = MIN_PART_TRIANGLES + Math.round((shareable * part.area) / totalArea);
      const [partIndices, partError] = MeshoptSimplifier.simplifyWithAttributes(
        Uint32Array.from(part.indices),
        weldedPositions,
        3,
        colors,
        CHANNELS,
        attributeWeights(),
        null,
        Math.min(budget * 3, part.indices.length),
        1,
        [],
      );
      simplifiedParts.push(...partIndices);
      error = Math.max(error, partError);
    }
    return { mesh: finish(compact(Uint32Array.from(simplifiedParts), weldedPositions, colors), null), error, note: '' };
  }
  // One pass over the whole weapon: edge collapses are taken in order of increasing (absolute)
  // geometric error, so flat panels collapse first and curved/detailed areas keep their triangles.
  // simplifyWithUpdate also moves the surviving vertices to error-minimizing positions (in place, so
  // every attempt works on copies).
  const positions = weldedPositions.slice();
  const attributes = colors.slice();
  let indices = allIndices.slice();
  let locked = null;
  let note = '';
  if (flags.rings !== 'false') {
    // Tubes (barrels, silencers, buffer tubes) are reduced to clean, evenly spaced rings first and
    // locked; see weapon-rings.mjs. They may use at most `ringShare` of the budget.
    const ringShare = Number(flags['ring-share'] ?? 0.2);
    const rings = reduceRings(indices, positions, { ringBudget: targetTriangles * ringShare });
    indices = rings.indices;
    locked = rings.locked;
    note = `rings: ${JSON.stringify(rings.stats)}`;
  }
  const [count, error] = MeshoptSimplifier.simplifyWithUpdate(
    indices,
    positions,
    3,
    attributes,
    CHANNELS,
    attributeWeights(),
    locked,
    Math.min(targetTriangles * 3, indices.length),
    1e9,
    ['ErrorAbsolute', 'Sparse'],
  );
  const reduced = compact(indices.subarray(0, count), positions, attributes);
  const ringVertices = locked ? reduced.sourceIndex.map((v) => locked[v] === 1) : null;
  return { mesh: finish(reduced, ringVertices), error, note };
}

/** Bakes normals (`--normals`) or, by default, moves tube-ring vertices to the front for the client. */
function finish(mesh, ringVertices) {
  if (bakeNormals) return splitCreases(mesh, creaseDegrees, ringVertices);
  if (!ringVertices || !ringVertices.some(Boolean)) return mesh;
  const order = [...ringVertices.keys()].sort((a, b) => ringVertices[b] - ringVertices[a] || a - b);
  const newIndex = new Uint32Array(order.length);
  order.forEach((v, i) => { newIndex[v] = i; });
  const positions = new Float32Array(mesh.positions.length);
  const vertexColors = new Float32Array(mesh.colors.length);
  order.forEach((v, i) => {
    positions.set(mesh.positions.subarray(v * 3, v * 3 + 3), i * 3);
    vertexColors.set(mesh.colors.subarray(v * CHANNELS, (v + 1) * CHANNELS), i * CHANNELS);
  });
  return {
    positions,
    colors: vertexColors,
    normals: null,
    indices: mesh.indices.map((v) => newIndex[v]),
    ringVertexCount: ringVertices.filter(Boolean).length,
  };
}

/** Keeps only the vertices referenced by `indices`. */
function compact(indices, positions, vertexColors) {
  const used = new Int32Array(positions.length / 3).fill(-1);
  const outPositions = [];
  const outColors = [];
  const outIndices = new Uint32Array(indices.length);
  const sourceIndex = [];
  for (let i = 0; i < indices.length; i++) {
    const v = indices[i];
    if (used[v] < 0) {
      used[v] = outPositions.length / 3;
      sourceIndex.push(v);
      outPositions.push(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]);
      for (let c = 0; c < CHANNELS; c++) outColors.push(vertexColors[v * CHANNELS + c]);
    }
    outIndices[i] = used[v];
  }
  return {
    positions: Float32Array.from(outPositions),
    colors: Float32Array.from(outColors),
    indices: outIndices,
    normals: null,
    sourceIndex,
  };
}

/**
 * Smooth normals with hard edges where adjacent faces meet at more than `creaseDeg` (like Blender's
 * "smooth by angle"). Each corner averages the area-weighted normals of the faces around its vertex
 * joined to its own face through smooth edges; vertices are split along sharp edges only.
 * Edges between two `smoothVertices` (locked tube rings) stay smooth up to 65 deg, so 6- and 8-sided
 * tubes shade round while their 90 deg cap edges stay sharp.
 */
function splitCreases(mesh, creaseDeg, smoothVertices = null) {
  const { positions, colors: vertexColors, indices } = mesh;
  const triCount = indices.length / 3;
  const faceNormals = new Float32Array(triCount * 3);
  const faceAreas = new Float32Array(triCount);
  const vertexFaces = Array.from({ length: positions.length / 3 }, () => []);
  for (let t = 0; t < triCount; t++) {
    const [a, b, c] = [indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]];
    const e1 = [0, 1, 2].map((k) => positions[b * 3 + k] - positions[a * 3 + k]);
    const e2 = [0, 1, 2].map((k) => positions[c * 3 + k] - positions[a * 3 + k]);
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const length = Math.hypot(...n) || 1;
    for (let k = 0; k < 3; k++) faceNormals[t * 3 + k] = n[k] / length;
    faceAreas[t] = length / 2;
    vertexFaces[a].push(t);
    vertexFaces[b].push(t);
    vertexFaces[c].push(t);
  }
  // Around each vertex, faces joined by a smooth edge (shared edge, dihedral within the crease angle)
  // form one smoothing group, and each group gets one area-weighted normal (Blender's smooth-by-angle).
  const cosCrease = Math.cos((creaseDeg * Math.PI) / 180);
  const cosRingCrease = Math.cos((65 * Math.PI) / 180);
  const keyToIndex = new Map();
  const outPositions = [];
  const outColors = [];
  const outNormals = [];
  const outIndices = new Uint32Array(indices.length);
  const groupOf = new Map();
  for (let v = 0; v < vertexFaces.length; v++) {
    const faces = vertexFaces[v];
    const parent = faces.map((_, i) => i);
    const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])));
    for (let i = 0; i < faces.length; i++) {
      for (let j = i + 1; j < faces.length; j++) {
        const f = faces[i];
        const g = faces[j];
        const shared = [0, 1, 2].map((k) => indices[f * 3 + k]).filter((x) => x !== v
          && (x === indices[g * 3] || x === indices[g * 3 + 1] || x === indices[g * 3 + 2]));
        if (shared.length === 0) continue;
        const dot = faceNormals[f * 3] * faceNormals[g * 3] + faceNormals[f * 3 + 1] * faceNormals[g * 3 + 1]
          + faceNormals[f * 3 + 2] * faceNormals[g * 3 + 2];
        const ringEdge = smoothVertices && smoothVertices[v] && shared.some((x) => smoothVertices[x]);
        if (dot >= (ringEdge ? cosRingCrease : cosCrease)) parent[find(i)] = find(j);
      }
    }
    const sums = new Map();
    for (let i = 0; i < faces.length; i++) {
      const root = find(i);
      const sum = sums.get(root) ?? [0, 0, 0];
      for (let k = 0; k < 3; k++) sum[k] += faceNormals[faces[i] * 3 + k] * faceAreas[faces[i]];
      sums.set(root, sum);
      groupOf.set(`${faces[i]},${v}`, root);
    }
    for (const [root, sum] of sums) {
      const length = Math.hypot(...sum) || 1;
      keyToIndex.set(`${v},${root}`, outPositions.length / 3);
      outPositions.push(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]);
      for (let c = 0; c < CHANNELS; c++) outColors.push(vertexColors[v * CHANNELS + c]);
      outNormals.push(sum[0] / length, sum[1] / length, sum[2] / length);
    }
  }
  for (let t = 0; t < triCount; t++) {
    for (let corner = 0; corner < 3; corner++) {
      const v = indices[t * 3 + corner];
      outIndices[t * 3 + corner] = keyToIndex.get(`${v},${groupOf.get(`${t},${v}`)}`);
    }
  }
  return {
    positions: Float32Array.from(outPositions),
    colors: Float32Array.from(outColors),
    normals: Float32Array.from(outNormals),
    indices: outIndices,
  };
}

