// Bakes a CS2 first-person viewmodel (skinned weapon + the game's viewmodel animation clips) or the shared
// first-person arms into a textured, meshopt-compressed GLB. Driven by `python -m extractor.cli cs2weapons`.
//
// Usage: node bake-viewmodel.mjs <model-export.glb> <out.glb> --clip=<role>:<clip-export.glb> ... [--texture-size=512]
//        node bake-viewmodel.mjs --arms <arms-export.glb> <out.glb> [--texture-size=512]
//
// Each material is baked to a WebP color map (long side --texture-size, default 512) plus a half-size WebP
// ORM map (occlusion/roughness/metalness, glTF's channel layout) from Source2Viewer's adapted ORM.
// Placeholder ORMs become constant roughness/metalness factors.
//
// Inputs are Source2Viewer-CLI exports made with --gltf_export_animations (that flag is what makes it emit
// the skeleton and skin): the weapon's vmdl, also with --gltf_export_materials --gltf_textures_adapt (the
// material names let decals be skipped, and the adapted textures are baked), and one export per AnimGraph2
// clip (.vnmclip). A clip export holds two
// skeletons: the camera-space viewmodel rig (`root_motion` -> `wpn`, arms, fingers) and the weapon's own
// skeleton (`weapon` -> `bolt`, `clip`, `muzzle`, ...), whose root the game attaches to the `wpn` bone.
//
// Weapon output (scene children):
// - `viewmodel`: the viewmodel rig (no mesh; the client drives the arms from it).
// - `weapon_rig`: the weapon skeleton. The client places this node at the `wpn` bone every frame.
// - the `body_hd` mesh (`body`/`body_legacy` fallback), skinned to the weapon skeleton, with positions,
//   normals, UVs, joints and weights, one primitive per material.
// - one animation per role (`draw`, `idle`, `shoot`, `shoot2`, `reload`) targeting both skeletons.
//   Rest poses are the idle pose; channels that never leave it are dropped, and keys are resampled.
// Units stay glTF metres, Y up, rig forward +Z.
import { Document, Logger, NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS, EXTMeshoptCompression, EXTTextureWebP, KHRMeshQuantization } from '@gltf-transform/extensions';
import { dedup, joinPrimitives, prune, quantize, reorder, resample, weld } from '@gltf-transform/functions';
import { MeshoptEncoder } from 'meshoptimizer';
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import sharp from 'sharp';

const args = process.argv.slice(2);
const clips = args.filter((a) => a.startsWith('--clip=')).map((a) => {
  const value = a.slice('--clip='.length);
  const split = value.indexOf(':');
  return { role: value.slice(0, split), path: value.slice(split + 1) };
});
const arms = args.includes('--arms');
const sizeArg = args.find((a) => a.startsWith('--texture-size='));
/** Long side of each baked base-color texture. */
const textureSize = sizeArg ? Number(sizeArg.split('=')[1]) : 512;
const [input, output] = args.filter((a) => !a.startsWith('--'));
if (!input || !output || (!arms && clips.length === 0) || !(textureSize > 0)) {
  console.error('Usage: node bake-viewmodel.mjs <model-export.glb> <out.glb> --clip=<role>:<clip.glb> ... [--texture-size=512]\n'
    + '       node bake-viewmodel.mjs --arms <arms-export.glb> <out.glb> [--texture-size=512]');
  process.exit(1);
}

const BODY_GROUPS = ['body_hd', 'body', 'body_legacy'];
// Decals and effect cards that aren't part of the solid weapon (same list as bake-weapon-model.mjs).
const SKIP_MATERIALS = /sticker_gaps|flame|chargemete|panorama|digits/i;
const SKINNED_ATTRIBUTES = ['POSITION', 'NORMAL', 'TEXCOORD_0', 'JOINTS_0', 'WEIGHTS_0'];
const REST_EPSILON = 1e-5;
const DEFAULT_ROUGHNESS = 0.7;
const WEBP_QUALITY = 85;
/** The packed occlusion/roughness/metalness map is baked at this fraction of the color map's size. */
const ORM_SCALE = 0.5;
// Lossy WebP mixes channels through YUV; a higher quality and sharp chroma keep the packed ORM usable.
const ORM_WEBP = { quality: 90, smartSubsample: true };
/** Smaller ORM sources are placeholders; their average becomes constant factors instead. */
const MIN_ORM_SIZE = 64;

await MeshoptEncoder.ready;
const readIO = new NodeIO().registerExtensions(ALL_EXTENSIONS);
const writeIO = new NodeIO()
  .registerExtensions([KHRMeshQuantization, EXTMeshoptCompression, EXTTextureWebP])
  .registerDependencies({ 'meshopt.encoder': MeshoptEncoder });

const out = new Document().setLogger(new Logger(Logger.Verbosity.WARN));
const buffer = out.createBuffer();
const scene = out.createScene();
const bakedMaterials = new Map();
const source = await readSource(input, true);

if (arms) await bakeArms(source);
else await bakeWeapon(source);

await out.transform(
  resample({ tolerance: 1e-4 }),
  weld(),
  dedup(),
  prune({ keepLeaves: true }),
  quantize({ quantizePosition: 14, quantizeNormal: 8, quantizeWeight: 8, quantizeTexcoord: 12 }),
  reorder({ encoder: MeshoptEncoder }),
);
out.createExtension(EXTMeshoptCompression).setRequired(true)
  .setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.QUANTIZE });
if (out.getRoot().listTextures().length) out.createExtension(EXTTextureWebP).setRequired(true);
await writeIO.write(output, out);
const meshes = out.getRoot().listMeshes();
const vertices = meshes.reduce((n, m) => n + m.listPrimitives().reduce((k, p) => k + p.getAttribute('POSITION').getCount(), 0), 0);
const animations = out.getRoot().listAnimations()
  .map((a) => `${a.getName()}(${a.listChannels().length}ch)`).join(' ');
const { size } = await import('node:fs').then((fs) => fs.promises.stat(output));
console.log(`${output}: ${(size / 1024).toFixed(1)} KB, ${vertices} verts ${animations}`);

async function bakeArms(doc) {
  const node = doc.getRoot().listNodes().find((n) => n.getMesh() && n.getSkin());
  if (!node) throw new Error('No skinned mesh in ' + input);
  const skin = node.getSkin();
  const root = skin.listJoints().find((joint) => !skin.listJoints().includes(parentOf(doc, joint)));
  const rig = out.createNode('arms_rig');
  scene.addChild(rig);
  rig.addChild(copyTree(root));
  await addSkinnedMesh('arms', node.getMesh().listPrimitives(), skin);
}

async function bakeWeapon(doc) {
  const meshNodes = doc.getRoot().listNodes().filter((n) => n.getMesh());
  const body = BODY_GROUPS.map((group) => meshNodes.find((n) => n.getName().endsWith('.' + group))).find(Boolean);
  if (!body?.getSkin()) throw new Error(`No skinned ${BODY_GROUPS.join('/')} mesh in ${input} (export with --gltf_export_animations)`);
  const prims = body.getMesh().listPrimitives()
    .filter((p) => p.getMaterial()?.getBaseColorTexture() && !SKIP_MATERIALS.test(p.getMaterial().getName()));
  if (prims.length === 0) throw new Error('No textured primitives in ' + input);

  // The idle clip (or the first one) provides both skeletons and the rest pose.
  const ordered = [...clips].sort((a, b) => (b.role === 'idle') - (a.role === 'idle'));
  const clipDocs = await Promise.all(ordered.map((clip) => readSource(clip.path)));
  const skeletonRoot = (clipDoc, pattern) => clipDoc.getRoot().listScenes()[0].listChildren()
    .find((n) => pattern.test(n.getName()) && !n.getMesh());
  const viewRoot = skeletonRoot(clipDocs[0], /characters\/viewmodel\.vnmskel$/);
  const weaponRoot = skeletonRoot(clipDocs[0], /skeletons\/weapons\/.*\.vnmskel$/);
  if (!viewRoot || !weaponRoot) throw new Error('Clip export has no viewmodel/weapon skeleton: ' + ordered[0].path);
  const viewmodel = copyTree(viewRoot).setName('viewmodel');
  const weaponRig = copyTree(weaponRoot).setName('weapon_rig');
  scene.addChild(viewmodel).addChild(weaponRig);

  // Model joints that the animation skeleton lacks (e.g. attachment helpers) hang off their parent.
  const skin = body.getSkin();
  const byName = nodesByName(weaponRig);
  for (const joint of skin.listJoints()) {
    if (byName.has(joint.getName())) continue;
    const parent = byName.get(parentOf(doc, joint)?.getName()) ?? weaponRig;
    const copy = copyNode(joint);
    parent.addChild(copy);
    byName.set(joint.getName(), copy);
  }
  await addSkinnedMesh('weapon', prims, skin, weaponRig);

  const targets = new Map([...nodesByName(viewmodel), ...nodesByName(weaponRig)]);
  ordered.forEach((clip, i) => copyAnimation(clipDocs[i], clip.role, targets, i === 0 && clip.role === 'idle'));
}

/** Adds a skinned mesh made of `prims` (attributes trimmed to what rendering needs) bound to copies of `skin`'s joints. */
async function addSkinnedMesh(name, prims, skin, jointRoot = null) {
  const byName = jointRoot ? nodesByName(jointRoot) : nodesByName(scene.listChildren()[0]);
  const joints = skin.listJoints().map((joint) => {
    const copy = byName.get(joint.getName());
    if (!copy) throw new Error('Missing joint ' + joint.getName());
    return copy;
  });
  const outSkin = out.createSkin(name).setInverseBindMatrices(copyAccessor(skin.getInverseBindMatrices()));
  for (const joint of joints) outSkin.addJoint(joint);
  const copies = prims.map((prim) => {
    const copy = out.createPrimitive().setIndices(copyAccessor(prim.getIndices()));
    for (const semantic of SKINNED_ATTRIBUTES) {
      const attribute = prim.getAttribute(semantic);
      if (!attribute) throw new Error(`Primitive without ${semantic}`);
      copy.setAttribute(semantic, copyAccessor(attribute));
    }
    return copy;
  });
  // One primitive (draw call) per material.
  const groups = new Map();
  prims.forEach((prim, i) => groups.set(prim.getMaterial(), [...(groups.get(prim.getMaterial()) ?? []), copies[i]]));
  const mesh = out.createMesh(name);
  for (const [material, list] of groups) {
    const primitive = list.length === 1 ? list[0] : joinPrimitives(list);
    if (list.length > 1) for (const copy of list) copy.dispose();
    mesh.addPrimitive(primitive.setMaterial(await bakeMaterial(material)));
  }
  scene.addChild(out.createNode(name).setMesh(mesh).setSkin(outSkin));
}

/**
 * One small WebP base-color texture per material, downscaled to `textureSize` on its long side, plus a
 * half-size ORM texture (or constant roughness/metalness factors when the source ORM is a placeholder).
 */
async function bakeMaterial(source) {
  const cached = bakedMaterials.get(source);
  if (cached) return cached;
  const colorImage = source.getBaseColorTexture()?.getImage();
  if (!colorImage?.length) throw new Error(`Material ${source.getName()} has no color image (export with textures)`);
  const { width: srcWidth, height: srcHeight } = await sharp(colorImage).metadata();
  const scale = Math.min(1, textureSize / Math.max(srcWidth, srcHeight));
  const width = Math.max(1, Math.round(srcWidth * scale));
  const height = Math.max(1, Math.round(srcHeight * scale));
  const webp = await sharp(colorImage).removeAlpha().resize(width, height, { kernel: 'lanczos3' })
    .webp({ quality: WEBP_QUALITY }).toBuffer();
  const name = source.getName();
  const texture = out.createTexture(name).setImage(webp).setMimeType('image/webp').setURI(`${name}.webp`);
  const material = out.createMaterial(name).setBaseColorTexture(texture);
  let summary = `${width}x${height} webp ${(webp.length / 1024).toFixed(1)} KB`;
  // Source2Viewer's adapted ORM: R = ambient occlusion, G = roughness, B = metalness (glTF's layout).
  const ormImage = source.getOcclusionTexture()?.getImage();
  const ormSize = ormImage?.length ? await sharp(ormImage).metadata() : null;
  if (ormSize && Math.min(ormSize.width, ormSize.height) >= MIN_ORM_SIZE) {
    const ormWidth = Math.max(1, Math.round(width * ORM_SCALE));
    const ormHeight = Math.max(1, Math.round(height * ORM_SCALE));
    const ormWebp = await sharp(ormImage).removeAlpha().resize(ormWidth, ormHeight).webp(ORM_WEBP).toBuffer();
    const orm = out.createTexture(`${name}_orm`).setImage(ormWebp).setMimeType('image/webp').setURI(`${name}_orm.webp`);
    material.setOcclusionTexture(orm).setMetallicRoughnessTexture(orm).setRoughnessFactor(1).setMetallicFactor(1);
    summary += `, orm ${ormWidth}x${ormHeight} ${(ormWebp.length / 1024).toFixed(1)} KB`;
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
  // Tangent-space normal map (glTF convention), at the ORM's size; placeholders are dropped.
  const normalImage = source.getNormalTexture()?.getImage();
  const normalSize = normalImage?.length ? await sharp(normalImage).metadata() : null;
  if (normalSize && Math.min(normalSize.width, normalSize.height) >= MIN_ORM_SIZE) {
    const normalWidth = Math.max(1, Math.round(width * ORM_SCALE));
    const normalHeight = Math.max(1, Math.round(height * ORM_SCALE));
    const normalWebp = await sharp(normalImage).removeAlpha().resize(normalWidth, normalHeight).webp(ORM_WEBP).toBuffer();
    const normal = out.createTexture(`${name}_normal`).setImage(normalWebp).setMimeType('image/webp')
      .setURI(`${name}_normal.webp`);
    material.setNormalTexture(normal);
    summary += `, normal ${normalWidth}x${normalHeight} ${(normalWebp.length / 1024).toFixed(1)} KB`;
  }
  bakedMaterials.set(source, material);
  console.log(`  ${name}: ${summary}`);
  return material;
}

/**
 * Copies a clip's animation onto `targets` (by node name) as `role`. With `setRest`, the first key of
 * every channel becomes the target's rest value; channels that never leave the rest pose are dropped.
 */
function copyAnimation(clipDoc, role, targets, setRest) {
  const source = clipDoc.getRoot().listAnimations()[0];
  if (!source) throw new Error('Clip export without animation for ' + role);
  const animation = out.createAnimation(role);
  for (const channel of source.listChannels()) {
    const target = targets.get(channel.getTargetNode()?.getName());
    if (!target) continue;
    const path = channel.getTargetPath();
    const sampler = channel.getSampler();
    const values = sampler.getOutput().getArray();
    const size = path === 'rotation' ? 4 : 3;
    if (setRest) setTrs(target, path, Array.from(values.slice(0, size)));
    if (atRest(target, path, values, size)) continue;
    const outSampler = out.createAnimationSampler()
      .setInput(copyAccessor(sampler.getInput()))
      .setOutput(copyAccessor(sampler.getOutput()))
      .setInterpolation(sampler.getInterpolation());
    animation.addSampler(outSampler).addChannel(
      out.createAnimationChannel().setTargetNode(target).setTargetPath(path).setSampler(outSampler));
  }
  // An empty clip (an idle that is exactly the rest pose) still marks the role as present.
  if (animation.listChannels().length === 0) animation.setExtras({ restPose: true });
}

/**
 * Reads a CLI GLB export. `withImages` loads the external texture files next to the GLB (the model
 * export); clip exports only need their skeletons and animation, so their PNGs are left empty.
 */
async function readSource(path, withImages = false) {
  const bytes = await import('node:fs').then((fs) => fs.promises.readFile(path));
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

function atRest(node, path, values, size) {
  const rest = path === 'rotation' ? node.getRotation() : path === 'scale' ? node.getScale() : node.getTranslation();
  for (let i = 0; i < values.length; i += size) {
    // q and -q are the same rotation.
    const sign = size === 4 && values[i + 3] * rest[3] + values[i] * rest[0] + values[i + 1] * rest[1] + values[i + 2] * rest[2] < 0 ? -1 : 1;
    for (let k = 0; k < size; k++) if (Math.abs(sign * values[i + k] - rest[k]) > REST_EPSILON) return false;
  }
  return true;
}

function setTrs(node, path, value) {
  if (path === 'rotation') node.setRotation(value);
  else if (path === 'scale') node.setScale(value);
  else node.setTranslation(value);
}

function copyNode(node) {
  return out.createNode(node.getName())
    .setTranslation(node.getTranslation()).setRotation(node.getRotation()).setScale(node.getScale());
}

function copyTree(node) {
  const copy = copyNode(node);
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

function parentOf(doc, node) {
  return doc.getRoot().listNodes().find((n) => n.listChildren().includes(node)) ?? null;
}
