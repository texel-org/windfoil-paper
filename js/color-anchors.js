// Joint color+alpha parametrization as a softmax over the 9 extreme points of
// the premultiplied style body {(alpha*color, alpha)}: the 8 sRGB cube corners
// at alpha=1 plus fully-transparent. Every decode is a convex combination, so
// layer styles are displayable by construction — no gamut clamps, masks, or
// leaks exist anywhere in this VJP. Gradients reach the weights at
// premultiplied strength (the 1/alpha in d color/d p cancels the alpha in the
// renderer's color cotangent), so nearly-transparent shapes still learn what
// color to become.

const ANCHOR_COUNT = 9;
const TRANSPARENT = 8; // anchors 0..7 are cube corners: bit c of k = channel c
const ALPHA_MAX = 0.999;
const ALPHA_EPS = 1e-4;
// Init keeps every weight at least this far from zero so no logit starts in
// softmax saturation (the same born-frozen failure as a logit of +-9).
const WEIGHT_FLOOR = 0.02;

const w = new Float64Array(ANCHOR_COUNT); // softmax scratch
const g = new Float64Array(ANCHOR_COUNT); // per-anchor cotangent scratch
const p = [0, 0, 0];

const clamp01 = (v) => Math.max(0, Math.min(1, v));

function softmax(logits, at, out = w, outAt = 0) {
  let max = -Infinity;
  for (let k = 0; k < ANCHOR_COUNT; k++) max = Math.max(max, logits[at + k]);
  let sum = 0;
  for (let k = 0; k < ANCHOR_COUNT; k++) {
    const value = Math.exp(logits[at + k] - max);
    out[outAt + k] = value;
    sum += value;
  }
  for (let k = 0; k < ANCHOR_COUNT; k++) out[outAt + k] /= sum;
}

// Premultiplied color: p_c = sum of corner weights whose corner has bit c set.
function premultiplied(weights, at, out) {
  // Cube-corner indices encode RGB in bits 0..2. Each channel is the sum of
  // the four opaque anchors with that bit set.
  out[0] = weights[at + 1] + weights[at + 3] + weights[at + 5] + weights[at + 7];
  out[1] = weights[at + 2] + weights[at + 3] + weights[at + 6] + weights[at + 7];
  out[2] = weights[at + 4] + weights[at + 5] + weights[at + 6] + weights[at + 7];
  return out;
}

// Overwrite shape i's logits to decode as (rgb, alpha): corner weights are the
// trilinear coordinates of rgb scaled by alpha, floored for mobility.
function writeAnchors(logits, i, rgb, alpha) {
  const at = ANCHOR_COUNT * i;
  const r = clamp01(rgb[0]), g = clamp01(rgb[1]), b = clamp01(rgb[2]);
  const a = clamp01(alpha);
  const norm = 1 + ANCHOR_COUNT * WEIGHT_FLOOR;
  for (let k = 0; k < TRANSPARENT; k++) {
    const t = (k & 1 ? r : 1 - r) * (k & 2 ? g : 1 - g) * (k & 4 ? b : 1 - b);
    logits[at + k] = Math.log((a * t + WEIGHT_FLOOR) / norm);
  }
  logits[at + TRANSPARENT] = Math.log((1 - a + WEIGHT_FLOOR) / norm);
}

export function decodeAnchors(logits, i, outColor) {
  softmax(logits, ANCHOR_COUNT * i);
  return decodeWeights(w, 0, outColor);
}

function decodeWeights(weights, at, outColor) {
  const a = 1 - weights[at + TRANSPARENT];
  const safe = Math.max(a, ALPHA_EPS);
  premultiplied(weights, at, p);
  outColor[0] = clamp01(p[0] / safe);
  outColor[1] = clamp01(p[1] / safe);
  outColor[2] = clamp01(p[2] / safe);
  return ALPHA_MAX * a;
}

// Accumulate dLoss/dlogits for shape i from the renderer's cotangents at
// cot[o..o+3] = (dColor, dAlpha). u = (p, a) is linear in the softmax weights
// (anchor matrix), so the chain is per-anchor cotangents -> softmax VJP.
function pullbackAnchors(logits, i, cot, o, out) {
  const at = ANCHOR_COUNT * i;
  softmax(logits, at);
  pullbackWeights(w, 0, cot, o, out, at);
}

function pullbackWeights(weights, at, cot, o, out, outAt) {
  const a = 1 - weights[at + TRANSPARENT];
  const safe = Math.max(a, ALPHA_EPS);
  premultiplied(weights, at, p);
  const dp0 = cot[o] / safe;
  const dp1 = cot[o + 1] / safe;
  const dp2 = cot[o + 2] / safe;
  let da = ALPHA_MAX * cot[o + 3];
  if (a > ALPHA_EPS) {
    da -= (cot[o] * p[0] + cot[o + 1] * p[1] + cot[o + 2] * p[2]) / (safe * safe);
  }
  let mean = 0;
  for (let k = 0; k < TRANSPARENT; k++) {
    let gk = da;
    if (k & 1) gk += dp0;
    if (k & 2) gk += dp1;
    if (k & 4) gk += dp2;
    g[k] = gk;
    mean += weights[at + k] * gk;
  }
  g[TRANSPARENT] = 0; // its anchor vector is the origin
  for (let k = 0; k < ANCHOR_COUNT; k++) {
    out[outAt + k] += weights[at + k] * (g[k] - mean);
  }
}

// The default style codec for trainable shape color and opacity.
export function anchorStyle({ lr = 0.05 } = {}) {
  return {
    kind: 'anchor',
    // Every decode is a convex combination of sRGB cube corners.
    range: 'unit',
    trainsAlpha: true,
    groups: ['colorAnchor'],
    lrs: { colorAnchor: { lr } },
    createScratch(params) {
      return {
        weights: new Float64Array(params.colorAnchor.length),
        valid: new Uint8Array(params.colorAnchor.length / ANCHOR_COUNT),
      };
    },
    invalidateScratch(scratch) {
      scratch?.valid.fill(0);
    },
    init(colors, alphas) {
      const colorAnchor = new Float32Array(ANCHOR_COUNT * colors.length);
      for (let i = 0; i < colors.length; i++) {
        writeAnchors(colorAnchor, i, colors[i], alphas[i] ?? 0.9);
      }
      return { colorAnchor };
    },
    decode(params, colorIndex, alphaIndex, outColor, scratch = null) {
      if (alphaIndex !== colorIndex) {
        throw new Error('anchor style fuses color and alpha; indices must match');
      }
      if (scratch) {
        const at = ANCHOR_COUNT * colorIndex;
        softmax(params.colorAnchor, at, scratch.weights, at);
        scratch.valid[colorIndex] = 1;
        return decodeWeights(scratch.weights, at, outColor);
      }
      return decodeAnchors(params.colorAnchor, colorIndex, outColor);
    },
    pullback(params, colorIndex, alphaIndex, cot, o, grads, scratch = null) {
      if (alphaIndex !== colorIndex) {
        throw new Error('anchor style fuses color and alpha; indices must match');
      }
      if (scratch?.valid[colorIndex]) {
        const at = ANCHOR_COUNT * colorIndex;
        pullbackWeights(scratch.weights, at, cot, o, grads.colorAnchor, at);
        return;
      }
      pullbackAnchors(params.colorAnchor, colorIndex, cot, o, grads.colorAnchor);
    },
  };
}
