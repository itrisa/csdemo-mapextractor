// Bakes a CS2 third-person player model, or the shared third-person animation set, for the web client's
// CS2 look. Driven by `python -m extractor.cli cs2weapons --set characters`.
//
// Usage: node bake-character.mjs --model <agent-export.glb> <out.glb> [--texture-size=1024] [--ratio=0.5]
//        node bake-character.mjs --anims <spec.json> <out.bin>
//
// Model input: a Source2Viewer-CLI export of an agent vmdl with --gltf_export_animations (needed for the
// skin), an animation list that matches nothing, the third-person mesh groups, and adapted textures.
// Output scene children: `rig` (the model skeleton, root `root_motion`) and one skinned `body` mesh, one
// primitive per material, simplified to --ratio of its triangles. Colour maps are WebP at --texture-size
// on the long side, ORM and normal maps at half that.
//
// Anims input: { skeleton: <model export>, clips: [{ name, file, kind }] }, one CLI export per .vnmclip.
// Clip exports hold the `worldmodel.vnmskel` skeleton, whose root_motion carries the Z-up to Y-up rotation;
// the model skeleton has it folded into root_motion's children, so their tracks are rotated to match.
// Channels that never leave the rest pose (identity for additive clips) are dropped, as are root_motion,
// scale, the legs of `upper` clips and finger tracks outside the idle poses; keys are reduced.
//
// The anims output is a compact binary rather than glTF, whose per-channel JSON would be ~3 MB here:
//   "CS2A", u32 version (1), u32 header length, JSON header (space padded to 4 bytes), data sections.
// Header: { bones: [name], clips: [{ name, kind, fps, duration, speed?, direction?, tracks }], frames,
// rotations, translations }. Each section is { count, byteOffset (from the data start), byteLength }, a
// meshopt glTF ATTRIBUTES buffer: frames u32 key frame numbers at the clip's fps (stride 4), rotations QUATERNION-filtered
// 16-bit quaternions (stride 8), translations EXPONENTIAL-filtered float xyz in metres (stride 12). A clip's
// `tracks` is a flat list of [bone, path (0 rotation, 1 translation), key count, first frame index, first
// value index]. For `move` clips, `speed` is the ground speed (Source units per second) and `direction`
// the travel direction (radians, Source yaw convention: 0 = forward, positive = left), measured from the
// planted foot because the clips have no root motion.
import { Document, Logger, NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS, EXTMeshoptCompression, EXTTextureWebP, KHRMeshQuantization } from '@gltf-transform/extensions';
import { dedup, joinPrimitives, prune, quantize, reorder, simplify, weld } from '@gltf-transform/functions';
import { MeshoptEncoder, MeshoptSimplifier } from 'meshoptimizer';
import { existsSync, promises as fs, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import sharp from 'sharp';

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const arg = args.find((a) => a.startsWith(`--${name}=`));
  return arg ? Number(arg.split('=')[1]) : fallback;
};
const mode = args.includes('--model') ? 'model' : args.includes('--anims') ? 'anims' : null;
const [input, output] = args.filter((a) => !a.startsWith('--'));
const textureSize = option('texture-size', 1024);
const ratio = option('ratio', 0.5);
if (!mode || !input || !output || !(textureSize > 0) || !(ratio > 0 && ratio <= 1)) {
  console.error('Usage: node bake-character.mjs --model <agent-export.glb> <out.glb> [--texture-size=1024] [--ratio=0.5]\n'
    + '       node bake-character.mjs --anims <spec.json> <out.glb>');
  process.exit(1);
}

const SKINNED_ATTRIBUTES = ['POSITION', 'NORMAL', 'TEXCOORD_0', 'JOINTS_0', 'WEIGHTS_0'];
const REST_EPSILON = 1e-4;
const DEFAULT_ROUGHNESS = 0.7;
const WEBP_QUALITY = 85;
const MAP_SCALE = 0.5;
const MAP_WEBP = { quality: 90, smartSubsample: true };
const MIN_MAP_SIZE = 64;
/** Simplification error limit, as a share of the mesh radius. */
const SIMPLIFY_ERROR = 0.01;
const METRES_TO_INCHES = 1 / 0.0254;
/** A foot within this height (metres) of its lowest point is planted. */
const CONTACT_HEIGHT = 0.015;
/** Key reduction limits per component: quaternion units (~0.06°) and metres. */
const ROTATION_TOLERANCE = 5e-4;
const TRANSLATION_TOLERANCE = 2e-4;
/** Other bones' translation tracks that stay this close (metres) to the rest pose are dropped. */
const MIN_TRANSLATION = 2e-3;
const ROOT_BONES = new Set(['pelvis', 'wpnPivot', 'wpn']);
// Finger curls only come from the idle poses (and per-weapon idle deltas); the client keeps them under the
// other clips, which saves about half of all tracks.
const FINGER = /^finger_/;
const IDLE_POSE = /_idle(_crouch)?$/;

await MeshoptEncoder.ready;
await MeshoptSimplifier.ready;
const readIO = new NodeIO().registerExtensions(ALL_EXTENSIONS);
const writeIO = new NodeIO()
  .registerExtensions([KHRMeshQuantization, EXTMeshoptCompression, EXTTextureWebP])
  .registerDependencies({ 'meshopt.encoder': MeshoptEncoder });

const out = new Document().setLogger(new Logger(Logger.Verbosity.WARN));
const buffer = out.createBuffer();
const scene = out.createScene();

if (mode === 'model') {
  await bakeModel(await readSource(input, true));
  await out.transform(
    weld(),
    simplify({ simplifier: MeshoptSimplifier, ratio, error: SIMPLIFY_ERROR }),
    dedup(),
    prune({ keepLeaves: true }),
    quantize({ quantizePosition: 14, quantizeNormal: 8, quantizeWeight: 8, quantizeTexcoord: 12 }),
    reorder({ encoder: MeshoptEncoder }),
  );
  out.createExtension(EXTMeshoptCompression).setRequired(true)
    .setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.QUANTIZE });
  out.createExtension(EXTTextureWebP).setRequired(true);
  await writeIO.write(output, out);
  const { size } = await fs.stat(output);
  const triangles = out.getRoot().listMeshes()
    .reduce((n, m) => n + m.listPrimitives().reduce((k, p) => k + p.getIndices().getCount() / 3, 0), 0);
  console.log(`${output}: ${(size / 1024).toFixed(1)} KB, ${triangles} triangles`);
} else {
  await bakeAnims(JSON.parse(await fs.readFile(input, 'utf8')));
}

async function bakeModel(doc) {
  const nodes = doc.getRoot().listNodes().filter((n) => n.getMesh() && n.getSkin());
  if (nodes.length === 0) throw new Error(`No skinned mesh in ${input} (export with --gltf_export_animations)`);
  const skin = nodes[0].getSkin();
  const root = doc.getRoot().listNodes().find((n) => n.getName() === 'root_motion');
  if (!root) throw new Error('No root_motion bone in ' + input);
  const rig = copyTree(root);
  scene.addChild(out.createNode('rig').addChild(rig));

  const byName = nodesByName(rig);
  const outSkin = out.createSkin('body').setInverseBindMatrices(copyAccessor(skin.getInverseBindMatrices()));
  for (const joint of skin.listJoints()) {
    const copy = byName.get(joint.getName());
    if (!copy) throw new Error('Missing joint ' + joint.getName());
    outSkin.addJoint(copy);
  }
  // Both meshes share one joint list, so they can share the skin and merge per material.
  const groups = new Map();
  for (const node of nodes) {
    const joints = node.getSkin().listJoints().map((j) => j.getName()).join();
    if (joints !== skin.listJoints().map((j) => j.getName()).join()) throw new Error('Meshes with different skins');
    for (const prim of node.getMesh().listPrimitives()) {
      const copy = out.createPrimitive().setIndices(copyAccessor(prim.getIndices()));
      for (const semantic of SKINNED_ATTRIBUTES) {
        const attribute = prim.getAttribute(semantic);
        if (!attribute) throw new Error(`Primitive without ${semantic}`);
        copy.setAttribute(semantic, copyAccessor(attribute));
      }
      groups.set(prim.getMaterial(), [...(groups.get(prim.getMaterial()) ?? []), copy]);
    }
  }
  const mesh = out.createMesh('body');
  for (const [material, list] of groups) {
    const primitive = list.length === 1 ? list[0] : joinPrimitives(list);
    if (list.length > 1) for (const copy of list) copy.dispose();
    mesh.addPrimitive(primitive.setMaterial(await bakeMaterial(material)));
  }
  scene.addChild(out.createNode('body').setMesh(mesh).setSkin(outSkin));
}

/** WebP colour map at `textureSize`, half-size ORM and normal maps (placeholders become constant factors). */
async function bakeMaterial(source) {
  const name = source.getName();
  const material = out.createMaterial(name);
  const colorImage = source.getBaseColorTexture()?.getImage();
  const colorSize = colorImage?.length ? await sharp(colorImage).metadata() : null;
  if (!colorSize || Math.min(colorSize.width, colorSize.height) < MIN_MAP_SIZE) {
    // Placeholder colour (e.g. gas mask lenses): a constant factor.
    const { channels } = colorImage?.length ? await sharp(colorImage).stats() : { channels: [{ mean: 128 }, { mean: 128 }, { mean: 128 }] };
    const factor = channels.slice(0, 3).map((c) => srgbToLinear(c.mean / 255));
    material.setBaseColorFactor([...factor, 1]).setRoughnessFactor(DEFAULT_ROUGHNESS).setMetallicFactor(0);
    console.log(`  ${name}: constant colour`);
    return material;
  }
  const scale = Math.min(1, textureSize / Math.max(colorSize.width, colorSize.height));
  const width = Math.max(1, Math.round(colorSize.width * scale));
  const height = Math.max(1, Math.round(colorSize.height * scale));
  const webp = await sharp(colorImage).removeAlpha().resize(width, height, { kernel: 'lanczos3' })
    .webp({ quality: WEBP_QUALITY }).toBuffer();
  material.setBaseColorTexture(out.createTexture(name).setImage(webp).setMimeType('image/webp').setURI(`${name}.webp`));
  let summary = `${width}x${height} webp ${(webp.length / 1024).toFixed(1)} KB`;
  const mapWidth = Math.max(1, Math.round(width * MAP_SCALE));
  const mapHeight = Math.max(1, Math.round(height * MAP_SCALE));
  const ormImage = source.getOcclusionTexture()?.getImage();
  const ormSize = ormImage?.length ? await sharp(ormImage).metadata() : null;
  if (ormSize && Math.min(ormSize.width, ormSize.height) >= MIN_MAP_SIZE) {
    const ormWebp = await sharp(ormImage).removeAlpha().resize(mapWidth, mapHeight).webp(MAP_WEBP).toBuffer();
    const orm = out.createTexture(`${name}_orm`).setImage(ormWebp).setMimeType('image/webp').setURI(`${name}_orm.webp`);
    material.setOcclusionTexture(orm).setMetallicRoughnessTexture(orm).setRoughnessFactor(1).setMetallicFactor(1);
    summary += `, orm ${(ormWebp.length / 1024).toFixed(1)} KB`;
  } else {
    let roughness = DEFAULT_ROUGHNESS;
    let metallic = 0;
    if (ormImage?.length) {
      const { channels } = await sharp(ormImage).stats();
      roughness = channels[1].mean / 255;
      metallic = channels[2].mean / 255;
    }
    material.setRoughnessFactor(Number(roughness.toFixed(3))).setMetallicFactor(Number(metallic.toFixed(3)));
    summary += `, roughness ${roughness.toFixed(2)}, metalness ${metallic.toFixed(2)}`;
  }
  const normalImage = source.getNormalTexture()?.getImage();
  const normalSize = normalImage?.length ? await sharp(normalImage).metadata() : null;
  if (normalSize && Math.min(normalSize.width, normalSize.height) >= MIN_MAP_SIZE) {
    const normalWebp = await sharp(normalImage).removeAlpha().resize(mapWidth, mapHeight).webp(MAP_WEBP).toBuffer();
    material.setNormalTexture(out.createTexture(`${name}_normal`).setImage(normalWebp).setMimeType('image/webp')
      .setURI(`${name}_normal.webp`));
    summary += `, normal ${(normalWebp.length / 1024).toFixed(1)} KB`;
  }
  console.log(`  ${name}: ${summary}`);
  return material;
}

async function bakeAnims(spec) {
  const model = await readSource(spec.skeleton);
  const modelRoot = model.getRoot().listNodes().find((n) => n.getName() === 'root_motion');
  if (!modelRoot) throw new Error('No root_motion bone in ' + spec.skeleton);
  const bones = nodesByName(copyTree(modelRoot));
  const boneNames = [...bones.keys()];
  const boneIndex = new Map(boneNames.map((name, i) => [name, i]));
  const lower = lowerBody(bones);
  const frames = [];
  const rotations = [];
  const translations = [];
  const clips = [];
  let droppedScale = 0;
  for (const clip of spec.clips) {
    const doc = await readSource(clip.file);
    const source = doc.getRoot().listAnimations()[0];
    if (!source) throw new Error('Clip export without animation: ' + clip.file);
    const clipBones = nodesByName(doc.getRoot().listScenes()[0].listChildren().find((n) => !n.getMesh()));
    const fold = clipBones.get('root_motion')?.getRotation() ?? [0, 0, 0, 1];
    const additive = clip.kind === 'additive';
    const keepFingers = additive || IDLE_POSE.test(clip.name);
    const fps = clipFps(source, clip.name);
    const tracks = new Map();
    const trackList = [];
    let duration = 0;
    for (const channel of source.listChannels()) {
      const name = channel.getTargetNode()?.getName();
      const path = channel.getTargetPath();
      const target = bones.get(name);
      if (!target || name === 'root_motion' || (path !== 'rotation' && path !== 'translation' && path !== 'scale')) continue;
      const sampler = channel.getSampler();
      const times = Array.from(sampler.getInput().getArray());
      const values = Array.from(sampler.getOutput().getArray());
      const size = path === 'rotation' ? 4 : 3;
      duration = Math.max(duration, times[times.length - 1]);
      if (clipParentName(doc, channel.getTargetNode()) === 'root_motion') foldTrack(values, path, size, fold, additive);
      if (path === 'rotation') alignHemisphere(values);
      tracks.set(`${name}.${path}`, { times, values });
      const rest = additive ? (path === 'rotation' ? [0, 0, 0, 1] : path === 'scale' ? [1, 1, 1] : [0, 0, 0]) : trs(target, path);
      if (path === 'scale') {
        if (!atRest(rest, values, 3, REST_EPSILON)) droppedScale++;
        continue;
      }
      if (clip.kind === 'upper' && lower.has(name)) continue;
      if (!keepFingers && FINGER.test(name)) continue;
      const restTolerance = path === 'rotation' ? REST_EPSILON : ROOT_BONES.has(name) ? REST_EPSILON : MIN_TRANSLATION;
      if (atRest(rest, values, size, restTolerance)) continue;
      const keep = reduceKeys(times, values, size, path === 'rotation' ? ROTATION_TOLERANCE : TRANSLATION_TOLERANCE);
      const store = path === 'rotation' ? rotations : translations;
      trackList.push(boneIndex.get(name), path === 'rotation' ? 0 : 1, keep.length, frames.length, store.length / size);
      for (const i of keep) {
        const frame = times[i] * fps;
        if (Math.abs(frame - Math.round(frame)) > 0.05) throw new Error(`${clip.name}: key off the ${fps} fps grid at ${times[i]}`);
        frames.push(Math.round(frame));
        for (let k = 0; k < size; k++) store.push(values[i * size + k]);
      }
    }
    const info = { name: clip.name, kind: clip.kind, fps, duration: round(duration, 4), tracks: trackList };
    if (clip.kind === 'move') Object.assign(info, measureGait(bones, tracks, clip.name));
    clips.push(info);
  }
  if (droppedScale) console.log(`  dropped ${droppedScale} animated scale tracks`);

  const timeData = MeshoptEncoder.encodeGltfBuffer(new Uint8Array(Uint32Array.from(frames).buffer), frames.length, 4, 'ATTRIBUTES');
  const rotationCount = rotations.length / 4;
  const rotationData = MeshoptEncoder.encodeGltfBuffer(
    MeshoptEncoder.encodeFilterQuat(Float32Array.from(rotations), rotationCount, 8, 16), rotationCount, 8, 'ATTRIBUTES');
  const translationCount = translations.length / 3;
  const translationData = MeshoptEncoder.encodeGltfBuffer(
    MeshoptEncoder.encodeFilterExp(Float32Array.from(translations), translationCount, 12, 16), translationCount, 12, 'ATTRIBUTES');
  const sections = [];
  let offset = 0;
  const section = (data, count) => {
    const entry = { count, byteOffset: offset, byteLength: data.length };
    sections.push(data);
    offset += align4(data.length);
    return entry;
  };
  const header = {
    format: 'cs2anims',
    version: 1,
    bones: boneNames,
    // Per clip, `tracks` is a flat list of [bone, path (0 rotation, 1 translation), key count, first key, first value].
    clips,
    frames: section(timeData, frames.length),
    rotations: section(rotationData, rotationCount),
    translations: section(translationData, translationCount),
  };
  const json = new TextEncoder().encode(JSON.stringify(header));
  const jsonLength = align4(json.length);
  const file = new Uint8Array(12 + jsonLength + offset);
  const view = new DataView(file.buffer);
  file.set(new TextEncoder().encode('CS2A'), 0);
  view.setUint32(4, 1, true);
  view.setUint32(8, jsonLength, true);
  file.set(json, 12);
  file.fill(0x20, 12 + json.length, 12 + jsonLength);
  let at = 12 + jsonLength;
  for (const data of sections) {
    file.set(data, at);
    at += align4(data.length);
  }
  await fs.writeFile(output, file);
  const moves = clips.filter((c) => c.kind === 'move' && c.name.startsWith('rifle_'));
  console.log('  gait: ' + moves.map((c) => `${c.name.slice(6)}=${c.speed}@${Math.round(c.direction * 180 / Math.PI)}`).join(' '));
  console.log(`${output}: ${(file.length / 1024).toFixed(1)} KB (header ${(jsonLength / 1024).toFixed(1)} KB), ${clips.length} clips, `
    + `${frames.length} keys (${rotationCount} rotation, ${translationCount} translation)`);
}

/** The clip's key rate, from the smallest key spacing (clips are authored at 30, 24, ... fps). */
function clipFps(animation, name) {
  let step = Infinity;
  for (const sampler of animation.listSamplers()) {
    const times = sampler.getInput().getArray();
    for (let i = 1; i < times.length; i++) step = Math.min(step, times[i] - times[i - 1]);
  }
  if (step === Infinity) return 30;
  const fps = Math.round(1 / step);
  if (!(fps > 0 && fps <= 120)) throw new Error(`${name}: unexpected key spacing ${step}`);
  return fps;
}

/** Indices of the keys to keep: linear interpolation of the rest stays within `tolerance` per component. */
function reduceKeys(times, values, size, tolerance) {
  const count = times.length;
  if (count === 1) return [0];
  const keep = [0];
  for (let i = 1; i < count - 1; i++) {
    const a = keep[keep.length - 1];
    const b = i + 1;
    let fits = true;
    for (let j = a + 1; j <= i && fits; j++) {
      const t = (times[j] - times[a]) / (times[b] - times[a]);
      for (let k = 0; k < size; k++) {
        const lerp = values[a * size + k] * (1 - t) + values[b * size + k] * t;
        if (Math.abs(lerp - values[j * size + k]) > tolerance) {
          fits = false;
          break;
        }
      }
    }
    if (!fits) keep.push(i);
  }
  const last = count - 1;
  const constant = keep.length === 1 && values.slice(last * size, last * size + size)
    .every((v, k) => Math.abs(v - values[k]) <= tolerance);
  if (!constant) keep.push(last);
  return keep;
}

function align4(n) {
  return (n + 3) & ~3;
}
/** pelvis and everything under it except the spine: the bones `upper` clips leave to the locomotion layer. */
function lowerBody(bones) {
  const lower = new Set(['pelvis']);
  const visit = (node) => {
    for (const child of node.listChildren()) {
      if (child.getName() === 'spine_0') continue;
      lower.add(child.getName());
      visit(child);
    }
  };
  visit(bones.get('pelvis'));
  return lower;
}

/**
 * Ground speed and direction of a locomotion cycle. Root motion is zero in the clips, so the planted foot
 * slides backwards at the travel speed: take the median ankle velocity while each foot is down.
 */
function measureGait(bones, tracks, name) {
  const times = [...new Set([...tracks.values()].flatMap((t) => t.times))].sort((a, b) => a - b);
  const feet = ['ankle_L', 'ankle_R'].map((foot) => times.map((time) => worldPosition(bones, tracks, foot, time)));
  const vx = [];
  const vz = [];
  for (const positions of feet) {
    const lowest = Math.min(...positions.map((p) => p[1]));
    for (let i = 1; i < times.length; i++) {
      const [a, b] = [positions[i - 1], positions[i]];
      if (a[1] > lowest + CONTACT_HEIGHT || b[1] > lowest + CONTACT_HEIGHT) continue;
      const dt = times[i] - times[i - 1];
      vx.push((b[0] - a[0]) / dt);
      vz.push((b[2] - a[2]) / dt);
    }
  }
  if (vx.length === 0) throw new Error('No foot contact in ' + name);
  // The body moves opposite to the planted foot. glTF +Z is Source forward (+X), glTF +X is Source left (+Y).
  const [x, z] = [-median(vx), -median(vz)];
  return { speed: round(Math.hypot(x, z) * METRES_TO_INCHES, 1), direction: round(Math.atan2(x, z), 4) };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function worldPosition(bones, tracks, name, time) {
  const chain = [];
  for (let node = bones.get(name); node && node.getName() !== 'root_motion'; node = parentOf(node)) chain.unshift(node);
  let position = [0, 0, 0];
  let rotation = [0, 0, 0, 1];
  for (const node of chain) {
    const t = sample(tracks.get(`${node.getName()}.translation`), time, 3) ?? node.getTranslation();
    const r = sample(tracks.get(`${node.getName()}.rotation`), time, 4) ?? node.getRotation();
    const moved = rotate(rotation, t);
    position = [position[0] + moved[0], position[1] + moved[1], position[2] + moved[2]];
    rotation = multiply(rotation, r);
  }
  return position;
}

function sample(track, time, size) {
  if (!track) return null;
  const { times, values } = track;
  let i = 0;
  while (i < times.length - 1 && times[i + 1] <= time) i++;
  if (i === times.length - 1) return Array.from(values.slice(i * size, i * size + size));
  const t = (time - times[i]) / (times[i + 1] - times[i]);
  const result = [];
  for (let k = 0; k < size; k++) result.push(values[i * size + k] * (1 - t) + values[(i + 1) * size + k] * t);
  if (size === 4) {
    const length = Math.hypot(...result);
    return result.map((v) => v / length);
  }
  return result;
}

/** Re-expresses a track of a root_motion child in the model skeleton, where root_motion has no rotation. */
function foldTrack(values, path, size, fold, additive) {
  for (let i = 0; i < values.length; i += size) {
    const value = Array.from(values.slice(i, i + size));
    let result;
    if (path === 'rotation') result = additive ? value : multiply(fold, value);
    else if (path === 'translation') result = rotate(fold, value);
    else result = value;
    for (let k = 0; k < size; k++) values[i + k] = result[k];
  }
}

/** Keeps consecutive quaternion keys in one hemisphere so linear interpolation takes the short way. */
function alignHemisphere(values) {
  for (let i = 4; i < values.length; i += 4) {
    const dot = values[i] * values[i - 4] + values[i + 1] * values[i - 3] + values[i + 2] * values[i - 2] + values[i + 3] * values[i - 1];
    if (dot < 0) for (let k = 0; k < 4; k++) values[i + k] = -values[i + k];
  }
}

function atRest(rest, values, size, tolerance) {
  for (let i = 0; i < values.length; i += size) {
    const sign = size === 4 && values[i] * rest[0] + values[i + 1] * rest[1] + values[i + 2] * rest[2] + values[i + 3] * rest[3] < 0 ? -1 : 1;
    for (let k = 0; k < size; k++) if (Math.abs(sign * values[i + k] - rest[k]) > tolerance) return false;
  }
  return true;
}

function trs(node, path) {
  return path === 'rotation' ? node.getRotation() : path === 'scale' ? node.getScale() : node.getTranslation();
}

function multiply(a, b) {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}

function rotate(q, v) {
  const p = multiply(multiply(q, [v[0], v[1], v[2], 0]), [-q[0], -q[1], -q[2], q[3]]);
  return [p[0], p[1], p[2]];
}

function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function round(value, digits) {
  return Number(value.toFixed(digits));
}

function copyTree(node) {
  const copy = out.createNode(node.getName())
    .setTranslation(node.getTranslation()).setRotation(node.getRotation()).setScale(node.getScale());
  for (const child of node.listChildren()) if (!child.getMesh()) copy.addChild(copyTree(child));
  return copy;
}

function copyAccessor(accessor) {
  return out.createAccessor(accessor.getName())
    .setType(accessor.getType())
    .setArray(accessor.getArray().slice())
    .setNormalized(accessor.getNormalized())
    .setBuffer(buffer);
}

function nodesByName(root) {
  const map = new Map();
  const visit = (node) => {
    map.set(node.getName(), node);
    for (const child of node.listChildren()) visit(child);
  };
  visit(root);
  return map;
}

function parentOf(node) {
  return node.listParents().find((p) => p.propertyType === 'Node') ?? null;
}

function clipParentName(doc, node) {
  return doc.getRoot().listNodes().find((n) => n.listChildren().includes(node))?.getName() ?? null;
}

/** Reads a CLI GLB export; `withImages` loads the texture files next to it, otherwise images stay empty. */
async function readSource(path, withImages = false) {
  const bytes = await fs.readFile(path);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let json = null;
  let bin = null;
  for (let offset = 12; offset < bytes.length;) {
    const length = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    const chunk = bytes.subarray(offset + 8, offset + 8 + length);
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(chunk));
    else if (type === 0x004e4942) bin = new Uint8Array(chunk);
    offset += 8 + length;
  }
  if (!json) throw new Error('Not a GLB: ' + path);
  const resources = bin ? { '@glb.bin': bin } : {};
  for (const image of json.images ?? []) {
    if (!image.uri) continue;
    const file = join(dirname(path), decodeURIComponent(image.uri));
    resources[image.uri] = withImages && existsSync(file) ? readFileSync(file) : new Uint8Array(0);
  }
  return readIO.readJSON({ json, resources });
}
