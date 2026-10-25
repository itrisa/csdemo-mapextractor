// Ring-aware pre-reduction for hard-surface weapon meshes. Driven by `python -m extractor.cli cs2weapons`.
//
// Generic edge-collapse simplifiers know nothing about cylinders: a 32-sided silencer gets reduced to an
// irregular, lopsided polygon whose caps tilt. CS2 weapon meshes are built from exact vertex rings
// (circles in an axis-aligned plane), so this pass finds them and reduces them in a controlled way:
//
// 1. Detect rings: for every vertex, circumcenters through pairs of its same-plane neighbors vote for a
//    (axis, plane, center, radius) circle; circles with enough evenly spread vertices are rings.
// 2. Group rings sharing an axis line (a tube with its caps, ribs, inner surface, ...) and reduce every
//    ring of the group to the same N evenly spaced angles (one vertex kept on top), so neighboring
//    rings stay aligned and the cross-section stays symmetric.
// 3. Profile reduction: a ring whose (axial, radius) profile point lies within `tolerance` of the line
//    between its two neighbor rings (e.g. the middle of a straight tube) is collapsed into the nearer
//    neighbor, vertex by vertex at matching angles.
// 4. The surviving ring vertices are returned as locked, so the general simplifier that runs afterwards
//    cannot bend them.

const PLANE_EPSILON = 2e-3; // inches; vertices of one ring share this axial coordinate
const CENTER_QUANTUM = 0.01; // inches; circle center/radius vote buckets
const MIN_RING_VERTICES = 8;
const MIN_RADIUS = 0.08; // inches; smaller circles (screw heads, pins) are left to the general simplifier

/** Default side count for a tube of this radius (inches). */
export function sidesForRadius(radius) {
  if (radius < 0.3) return 6;
  if (radius < 1.5) return 8;
  return 12;
}

/**
 * @param {Uint32Array} indices
 * @param {Float32Array} positions
 * @param {{ tolerance?: number, sides?: (radius: number) => number, ringBudget?: number, minLockRadius?: number }} options
 *   tolerance: max profile deviation (inches) for dropping a ring. ringBudget: max triangles the locked
 *   rings may cost; rings are locked largest radius first until it's used up. Rings that aren't locked
 *   (budget exhausted, or radius below minLockRadius) are still side-reduced, just not protected.
 * @returns {{ indices: Uint32Array, locked: Uint8Array, stats: object }}
 */
export function reduceRings(indices, positions, options = {}) {
  const sides = options.sides ?? sidesForRadius;
  const vertexCount = positions.length / 3;
  const neighbors = Array.from({ length: vertexCount }, () => new Set());
  for (let i = 0; i < indices.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const a = indices[i + k];
      const b = indices[i + ((k + 1) % 3)];
      neighbors[a].add(b);
      neighbors[b].add(a);
    }
  }

  const rings = detectRings(positions, neighbors);
  const groups = groupRings(rings);

  const sideRemap = Int32Array.from({ length: vertexCount }, (_, i) => i);
  for (const group of groups) {
    group.sides = sides(group.maxRadius);
    for (const ring of group.rings) reduceRingSides(ring, group, positions, sideRemap);
  }
  const afterSides = remapTriangles(indices, sideRemap);
  const result = reduceProfiles(afterSides, groups, positions, vertexCount, options.tolerance ?? 0.03);

  // Lock the most visible rings first. Each locked ring costs about 2 triangles per vertex (the quads
  // to its neighbor ring).
  const minLockRadius = options.minLockRadius ?? 0.15;
  const budget = options.ringBudget ?? Infinity;
  const candidates = result.rings
    .filter(({ node }) => node.ring.radius >= minLockRadius)
    .sort((a, b) => b.group.maxRadius - a.group.maxRadius || b.node.ring.radius - a.node.ring.radius);
  const locked = new Uint8Array(vertexCount);
  let ringVertices = 0;
  let lockedRings = 0;
  for (const { node } of candidates) {
    if ((ringVertices + node.verts.size) * 2 > budget) continue;
    for (const v of node.verts) locked[v] = 1;
    ringVertices += node.verts.size;
    lockedRings++;
  }
  return {
    indices: result.indices,
    locked,
    stats: {
      rings: rings.length,
      groups: groups.length,
      afterProfile: result.rings.length,
      lockedRings,
      lockedVertices: ringVertices,
    },
  };
}

function axesFor(axis) {
  // Right-handed (u, w) plane for each axis, so angles are consistent.
  return [[1, 2], [2, 0], [0, 1]][axis];
}

function detectRings(positions, neighbors) {
  const votes = new Map();
  const p = (v, c) => positions[v * 3 + c];
  for (let axis = 0; axis < 3; axis++) {
    const [u, w] = axesFor(axis);
    for (let v = 0; v < neighbors.length; v++) {
      const plane = [...neighbors[v]].filter((n) => Math.abs(p(n, axis) - p(v, axis)) < PLANE_EPSILON);
      if (plane.length < 2) continue;
      const limit = Math.min(plane.length, 8);
      const seen = new Set();
      for (let i = 0; i < limit; i++) {
        for (let j = i + 1; j < limit; j++) {
          const c = circumcenter(p(v, u), p(v, w), p(plane[i], u), p(plane[i], w), p(plane[j], u), p(plane[j], w));
          if (!c) continue;
          const radius = Math.hypot(p(v, u) - c[0], p(v, w) - c[1]);
          if (radius < MIN_RADIUS || radius > 20) continue;
          const key = [axis, Math.round(p(v, axis) / PLANE_EPSILON), Math.round(c[0] / CENTER_QUANTUM),
            Math.round(c[1] / CENTER_QUANTUM), Math.round(radius / CENTER_QUANTUM)].join(',');
          if (seen.has(key)) continue;
          seen.add(key);
          let set = votes.get(key);
          if (!set) votes.set(key, (set = new Set()));
          set.add(v);
        }
      }
    }
  }

  // Each candidate circle: refit the center from its member vertices and require even coverage.
  const candidates = [];
  for (const [key, members] of votes) {
    if (members.size < MIN_RING_VERTICES) continue;
    const [axis] = key.split(',').map(Number);
    const [u, w] = axesFor(axis);
    const verts = [...members];
    let cu = 0;
    let cw = 0;
    for (const v of verts) {
      cu += p(v, u);
      cw += p(v, w);
    }
    cu /= verts.length;
    cw /= verts.length;
    const radius = verts.reduce((sum, v) => sum + Math.hypot(p(v, u) - cu, p(v, w) - cw), 0) / verts.length;
    if (verts.some((v) => Math.abs(Math.hypot(p(v, u) - cu, p(v, w) - cw) - radius) > Math.max(0.01, radius * 0.03))) continue;
    const angles = verts.map((v) => Math.atan2(p(v, w) - cw, p(v, u) - cu)).sort((a, b) => a - b);
    let maxGap = angles[0] + 2 * Math.PI - angles[angles.length - 1];
    for (let i = 1; i < angles.length; i++) maxGap = Math.max(maxGap, angles[i] - angles[i - 1]);
    if (maxGap > (2 * Math.PI / verts.length) * 1.6) continue;
    candidates.push({ axis, axial: p(verts[0], axis), center: [cu, cw], radius, verts });
  }
  // A vertex belongs to at most one ring: prefer rings with more vertices.
  candidates.sort((a, b) => b.verts.length - a.verts.length);
  const owner = new Map();
  const rings = [];
  for (const ring of candidates) {
    if (ring.verts.some((v) => owner.has(v))) continue;
    for (const v of ring.verts) owner.set(v, ring);
    const [u, w] = axesFor(ring.axis);
    ring.angleOf = new Map(ring.verts.map((v) => [v, Math.atan2(p(v, w) - ring.center[1], p(v, u) - ring.center[0])]));
    rings.push(ring);
  }
  return rings;
}

function circumcenter(ax, ay, bx, by, cx, cy) {
  const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  if (Math.abs(d) < 1e-9) return null;
  const a2 = ax * ax + ay * ay;
  const b2 = bx * bx + by * by;
  const c2 = cx * cx + cy * cy;
  return [(a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d, (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d];
}

/** Rings on the same axis line (same axis and center within a small tolerance). */
function groupRings(rings) {
  const groups = [];
  for (const ring of rings) {
    let group = groups.find((g) => g.axis === ring.axis
      && Math.hypot(g.center[0] - ring.center[0], g.center[1] - ring.center[1]) < 0.02);
    if (!group) groups.push((group = { axis: ring.axis, center: ring.center, rings: [], maxRadius: 0 }));
    group.rings.push(ring);
    ring.group = group;
    group.maxRadius = Math.max(group.maxRadius, ring.radius);
  }
  return groups;
}

const angleDistance = (a, b) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));

/** Keeps the ring vertex nearest each of `sides` evenly spaced target angles (one at the top). */
function reduceRingSides(ring, group, positions, remap) {
  const n = group.sides;
  if (ring.verts.length <= n) {
    ring.kept = ring.verts;
    ring.reduced = false;
    return;
  }
  // Angle 90 deg is "up" (+w) for the X axis (w = Z); a vertex on top keeps the silhouette symmetric.
  const targets = Array.from({ length: n }, (_, j) => Math.PI / 2 + (j * 2 * Math.PI) / n);
  const kept = [];
  for (const target of targets) {
    let best = -1;
    for (const v of ring.verts) {
      if (kept.includes(v)) continue;
      if (best < 0 || angleDistance(ring.angleOf.get(v), target) < angleDistance(ring.angleOf.get(best), target)) best = v;
    }
    kept.push(best);
  }
  for (const v of ring.verts) {
    if (kept.includes(v)) continue;
    let best = kept[0];
    for (const k of kept) {
      if (angleDistance(ring.angleOf.get(v), ring.angleOf.get(k)) < angleDistance(ring.angleOf.get(v), ring.angleOf.get(best))) best = k;
    }
    remap[v] = best;
  }
  ring.kept = kept;
  ring.reduced = true;
}

function remapTriangles(indices, remap) {
  const out = [];
  const seen = new Map();
  for (let i = 0; i < indices.length; i += 3) {
    const a = resolve(remap, indices[i]);
    const b = resolve(remap, indices[i + 1]);
    const c = resolve(remap, indices[i + 2]);
    if (a === b || b === c || a === c) continue;
    // Collapses can fold two triangles onto the same vertices; drop such pairs.
    const key = [a, b, c].sort((x, y) => x - y).join(',');
    if (seen.has(key)) {
      const other = seen.get(key);
      if (other >= 0) {
        out[other] = out[other + 1] = out[other + 2] = -1;
        seen.set(key, -1);
      }
      continue;
    }
    seen.set(key, out.length);
    out.push(a, b, c);
  }
  return Uint32Array.from(out.filter((x) => x >= 0));
}

function resolve(remap, v) {
  while (remap[v] !== v) v = remap[v];
  return v;
}

function reduceProfiles(indices, groups, positions, vertexCount, tolerance) {
  const remap = Int32Array.from({ length: vertexCount }, (_, i) => i);
  const neighbors = Array.from({ length: vertexCount }, () => new Set());
  for (let i = 0; i < indices.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      neighbors[indices[i + k]].add(indices[i + ((k + 1) % 3)]);
      neighbors[indices[i + ((k + 1) % 3)]].add(indices[i + k]);
    }
  }
  const lockedRings = [];
  for (const group of groups) {
    // Only rings reduced to the group's angle set can be merged into each other.
    const live = group.rings.filter((r) => r.reduced || r.verts.length === group.sides).map((r) => ({
      ring: r,
      verts: new Set(r.kept),
      point: [r.axial, r.radius],
    }));
    const ringOf = new Map();
    for (const node of live) for (const v of node.verts) ringOf.set(v, node);
    const linksOf = (node) => {
      const links = new Set();
      let foreign = false;
      for (const v of node.verts) {
        for (const n of neighbors[v]) {
          const other = ringOf.get(n);
          if (other === node) continue;
          if (other) links.add(other);
          else foreign = true;
        }
      }
      return { links: [...links], foreign };
    };
    for (let changed = true; changed;) {
      changed = false;
      let best = null;
      for (const node of live) {
        if (node.removed) continue;
        const { links, foreign } = linksOf(node);
        if (foreign || links.length !== 2) continue;
        const deviation = segmentDistance(node.point, links[0].point, links[1].point);
        if (deviation <= tolerance && (!best || deviation < best.deviation)) best = { node, links, deviation };
      }
      if (!best) break;
      const { node, links } = best;
      const target = Math.hypot(links[0].point[0] - node.point[0], links[0].point[1] - node.point[1])
        <= Math.hypot(links[1].point[0] - node.point[0], links[1].point[1] - node.point[1]) ? links[0] : links[1];
      for (const v of node.verts) {
        const angle = node.ring.angleOf.get(v);
        let match = null;
        for (const t of target.verts) {
          if (!match || angleDistance(target.ring.angleOf.get(t), angle) < angleDistance(target.ring.angleOf.get(match), angle)) match = t;
        }
        remap[v] = match;
        for (const n of neighbors[v]) {
          if (n === match) continue;
          neighbors[n].delete(v);
          neighbors[n].add(match);
          neighbors[match].add(n);
        }
        neighbors[match].delete(v);
        neighbors[match].delete(match);
        ringOf.delete(v);
      }
      node.removed = true;
      changed = true;
    }
    for (const node of live) if (!node.removed) lockedRings.push({ node, group });
  }
  return { indices: remapTriangles(indices, remap), rings: lockedRings };
}

function segmentDistance(point, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const length2 = dx * dx + dy * dy;
  const t = length2 ? Math.max(0, Math.min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2)) : 0;
  return Math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy);
}
