// Constrained-palette color+alpha codec. A small shared palette of `count`
// trainable colors that the optimizer discovers, plus a per-shape softmax that
// assigns each shape to a blend of those colors. This is the fixed anchor style
// generalized: the anchors are no longer the 8 sRGB cube corners but N learned,
// shared colors, and each palette color is itself a softmax over the corners so
// it stays in gamut. Every decode is a convex combination of in-gamut colors,
// so results are displayable by construction — no clamps, masks, or leaks.
//
// Opaque mode assigns over the N colors alone: every shape is fully opaque and,
// once hardened, is exactly one of N colors (an N-color dither). Translucent
// mode adds a transparent anchor so shapes can fade and blend the background,
// while the base ink still snaps to one of N colors.
//
// Two batch hooks keep the shared palette efficient and finite-difference safe:
// prepare() decodes the N palette colors once per decode/pullback pass, and
// finalize() folds the palette-color gradient accumulated across all shapes
// back through each color's corner softmax.

import { convert, OKLab, sRGB } from '@texel/color';

const CORNERS = 8; // sRGB cube corners; bit c of k is channel c of corner k
const ALPHA_MAX = 0.999;
const ALPHA_EPS = 1e-4;
const WEIGHT_FLOOR = 0.02; // mobility floor: keep no logit in softmax saturation

const clamp01 = (v) => Math.max(0, Math.min(1, v));

// Deterministic sub-random logit jitter so shapes start assigned differently
// (broken symmetry) without an informed, target-derived palette init.
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// A spread of distinct starting hues (S/V fixed) so the N entries begin apart.
function hueColor(i, count) {
  const h = (i / count) * 6;
  const x = 1 - Math.abs((h % 2) - 1);
  const [r, g, b] =
    h < 1 ? [1, x, 0] : h < 2 ? [x, 1, 0] : h < 3 ? [0, 1, x]
    : h < 4 ? [0, x, 1] : h < 5 ? [x, 0, 1] : [1, 0, x];
  const s = 0.7, v = 0.9;
  return [v * (1 - s + s * r), v * (1 - s + s * g), v * (1 - s + s * b)];
}

// Corner logits whose softmax decodes (near) rgb, via trilinear cube weights.
// A zero floor writes exact logits (zero corners pinned deep) — used for fixed
// palettes, where mobility is irrelevant because the entries never train.
function writeCornerLogits(palette, j, rgb, floor = WEIGHT_FLOOR) {
  const r = clamp01(rgb[0]), g = clamp01(rgb[1]), b = clamp01(rgb[2]);
  const norm = 1 + CORNERS * floor;
  for (let k = 0; k < CORNERS; k++) {
    const t = (k & 1 ? r : 1 - r) * (k & 2 ? g : 1 - g) * (k & 4 ? b : 1 - b);
    palette[j * CORNERS + k] =
      floor > 0 ? Math.log((t + floor) / norm) : t > 0 ? Math.log(t) : -12;
  }
}

function softmax(logits, at, len, out, outAt) {
  let max = -Infinity;
  for (let k = 0; k < len; k++) max = Math.max(max, logits[at + k]);
  let sum = 0;
  for (let k = 0; k < len; k++) {
    const value = Math.exp(logits[at + k] - max);
    out[outAt + k] = value;
    sum += value;
  }
  for (let k = 0; k < len; k++) out[outAt + k] /= sum;
}

// Decode palette entry j's corner softmax into an rgb color, writing the corner
// weights into `weights` (for the finalize VJP) when provided.
function paletteColor(paletteLogits, j, out3, weights = null) {
  const w = weights ?? new Float64Array(CORNERS);
  softmax(paletteLogits, j * CORNERS, CORNERS, w, weights ? j * CORNERS : 0);
  const at = weights ? j * CORNERS : 0;
  let c0 = 0, c1 = 0, c2 = 0;
  for (let k = 0; k < CORNERS; k++) {
    const v = w[at + k];
    if (k & 1) c0 += v;
    if (k & 2) c1 += v;
    if (k & 4) c2 += v;
  }
  out3[0] = c0;
  out3[1] = c1;
  out3[2] = c2;
}

export function learnedPaletteStyle({
  count,
  // Fixed palette (--colors): rgb triples the entries are pinned to exactly —
  // the palette group's lr is zero so only assignments (and, translucent,
  // opacity) train. count is taken from the list.
  fixedColors = null,
  opaque = false,
  seed = 1,
  assignLr = 0.05,
  paletteLr = 0.02,
  fidelity = 0,
} = {}) {
  const fixed = Array.isArray(fixedColors) && fixedColors.length > 0;
  if (fixed) count = fixedColors.length;
  if (!Number.isInteger(count) || count < (fixed ? 1 : 2)) {
    throw new Error(
      fixed
        ? "--colors needs at least one color"
        : "learned palette count must be an integer >= 2",
    );
  }
  const anchors = opaque ? count : count + 1; // per-shape softmax width
  const TRANSPARENT = count; // valid only when !opaque
  const a = new Float64Array(anchors); // per-shape assignment scratch
  let hard = false;
  // Optional palette-fidelity regularizer: pull each learned color toward the
  // perceptual (OKLab) centroid of the target colors of the shapes assigned to
  // it, so the palette drifts toward the image's actual colors instead of the
  // saturated primaries that merely fit best. Uses each shape's one pre-sampled
  // target color (n samples, not the pixels) — cheap and off unless fidelity>0.
  let targetLab = null; // OKLab of each shape's sampled target color

  const decodeColor = (params, colorIndex, colors, outColor, hardMode) => {
    softmax(params.assign, colorIndex * anchors, anchors, a, 0);
    let argmax = 0;
    if (hardMode) {
      for (let j = 1; j < count; j++) if (a[j] > a[argmax]) argmax = j;
    }
    let p0 = 0, p1 = 0, p2 = 0;
    for (let j = 0; j < count; j++) {
      const w = hardMode ? (j === argmax ? 1 : 0) : a[j];
      p0 += w * colors[j * 3];
      p1 += w * colors[j * 3 + 1];
      p2 += w * colors[j * 3 + 2];
    }
    if (opaque) {
      outColor[0] = clamp01(p0);
      outColor[1] = clamp01(p1);
      outColor[2] = clamp01(p2);
      return 1;
    }
    const alpha = 1 - a[TRANSPARENT];
    if (hardMode) {
      // Hard weights are unit, not premultiplied: the blend already IS the
      // argmax color, so dividing by alpha would over-brighten it.
      outColor[0] = clamp01(p0);
      outColor[1] = clamp01(p1);
      outColor[2] = clamp01(p2);
      return ALPHA_MAX * alpha;
    }
    const safe = Math.max(alpha, ALPHA_EPS);
    outColor[0] = clamp01(p0 / safe);
    outColor[1] = clamp01(p1 / safe);
    outColor[2] = clamp01(p2 / safe);
    return ALPHA_MAX * alpha;
  };

  return {
    kind: "learned-palette",
    // Palette entries and assignments are softmax blends of cube corners.
    range: "unit",
    trainsAlpha: !opaque,
    opaque,
    count,
    groups: ["palette", "assign"],
    lrs: { palette: { lr: fixed ? 0 : paletteLr }, assign: { lr: assignLr } },

    // The palette itself is uninformed — a spread of hues the optimizer is free
    // to move anywhere. Only the per-shape assignments are seeded, biasing each
    // shape toward whichever starting hue is nearest its local target color.
    // That breaks symmetry so entries specialize instead of all collapsing to
    // the mean color; the palette colors are still discovered, not seeded.
    init(colors) {
      const n = colors.length;
      const palette = new Float32Array(count * CORNERS);
      const hues = [];
      for (let j = 0; j < count; j++) {
        const rgb = fixed ? fixedColors[j] : hueColor(j, count);
        hues.push(rgb);
        writeCornerLogits(palette, j, rgb, fixed ? 0 : WEIGHT_FLOOR);
      }
      const random = mulberry32(seed);
      const assign = new Float32Array(n * anchors);
      for (let i = 0; i < n; i++) {
        const base = i * anchors;
        for (let k = 0; k < anchors; k++) assign[base + k] = (random() - 0.5) * 0.1;
        const rgb = colors[i];
        if (rgb) assign[base + nearest(hues, rgb)] += 3;
      }
      // Pre-sample each shape's target color into OKLab, once, for the fidelity
      // regularizer (skipped entirely when it is off).
      if (fidelity > 0) {
        targetLab = new Float64Array(n * 3);
        for (let i = 0; i < n; i++) {
          const lab = convert(colors[i] ?? [0, 0, 0], sRGB, OKLab);
          targetLab[i * 3] = lab[0];
          targetLab[i * 3 + 1] = lab[1];
          targetLab[i * 3 + 2] = lab[2];
        }
      }
      return { palette, assign };
    },

    createScratch() {
      return {
        colors: new Float64Array(count * 3),
        weights: new Float64Array(count * CORNERS),
        colorGrad: new Float64Array(count * 3),
        // Fidelity centroid accumulators (unused when fidelity is 0).
        centroidNum: new Float64Array(count * 3),
        centroidDen: new Float64Array(count),
      };
    },

    // Decode the shared palette once per decode/pullback pass. Called by the
    // model before its per-shape loop, so palette reads stay fresh (correct
    // finite differences) without recomputing per shape.
    prepare(params, scratch) {
      for (let j = 0; j < count; j++) {
        const out = [0, 0, 0];
        paletteColor(params.palette, j, out, scratch.weights);
        scratch.colors[j * 3] = out[0];
        scratch.colors[j * 3 + 1] = out[1];
        scratch.colors[j * 3 + 2] = out[2];
      }
    },

    harden() {
      hard = true;
    },

    // The learned palette as rgb triples (for reporting the discovered colors).
    paletteColors(params) {
      const out = [];
      for (let j = 0; j < count; j++) {
        const rgb = [0, 0, 0];
        paletteColor(params.palette, j, rgb);
        out.push(rgb);
      }
      return out;
    },

    decode(params, colorIndex, alphaIndex, outColor, scratch) {
      if (alphaIndex !== colorIndex) {
        throw new Error("learned palette fuses color and alpha; indices must match");
      }
      const colors = scratch ? scratch.colors : freshColors(params, count);
      return decodeColor(params, colorIndex, colors, outColor, hard);
    },

    pullback(params, colorIndex, alphaIndex, cot, o, grads, scratch) {
      if (alphaIndex !== colorIndex) {
        throw new Error("learned palette fuses color and alpha; indices must match");
      }
      const colors = scratch.colors;
      softmax(params.assign, colorIndex * anchors, anchors, a, 0);
      // Accumulate this shape's target color toward each entry's OKLab centroid,
      // weighted by its assignment (soft responsibility).
      if (fidelity > 0 && targetLab) {
        const at = colorIndex * 3;
        const t0 = targetLab[at], t1 = targetLab[at + 1], t2 = targetLab[at + 2];
        for (let j = 0; j < count; j++) {
          const w = a[j];
          scratch.centroidNum[j * 3] += w * t0;
          scratch.centroidNum[j * 3 + 1] += w * t1;
          scratch.centroidNum[j * 3 + 2] += w * t2;
          scratch.centroidDen[j] += w;
        }
      }
      let p0 = 0, p1 = 0, p2 = 0;
      for (let j = 0; j < count; j++) {
        p0 += a[j] * colors[j * 3];
        p1 += a[j] * colors[j * 3 + 1];
        p2 += a[j] * colors[j * 3 + 2];
      }
      const alpha = opaque ? 1 : 1 - a[TRANSPARENT];
      const safe = Math.max(alpha, ALPHA_EPS);
      const dp0 = cot[o] / safe, dp1 = cot[o + 1] / safe, dp2 = cot[o + 2] / safe;
      let da = 0;
      if (!opaque) {
        da = ALPHA_MAX * cot[o + 3];
        if (alpha > ALPHA_EPS) {
          da -= (cot[o] * p0 + cot[o + 1] * p1 + cot[o + 2] * p2) / (safe * safe);
        }
      }
      // Cotangent on each assignment weight, then the softmax VJP into logits.
      let mean = 0;
      const g = new Float64Array(count);
      for (let j = 0; j < count; j++) {
        g[j] = da + dp0 * colors[j * 3] + dp1 * colors[j * 3 + 1] + dp2 * colors[j * 3 + 2];
        mean += a[j] * g[j];
      }
      const base = colorIndex * anchors;
      for (let k = 0; k < anchors; k++) {
        const gk = k < count ? g[k] : 0; // transparent anchor's vector is origin
        grads.assign[base + k] += a[k] * (gk - mean);
      }
      // Cotangent on each palette color, accumulated across shapes for finalize.
      for (let j = 0; j < count; j++) {
        scratch.colorGrad[j * 3] += dp0 * a[j];
        scratch.colorGrad[j * 3 + 1] += dp1 * a[j];
        scratch.colorGrad[j * 3 + 2] += dp2 * a[j];
      }
    },

    // Fold the accumulated palette-color gradient back through each color's
    // corner softmax, then clear it for the next pullback. When the fidelity
    // regularizer is on, first add a pull toward each entry's OKLab centroid,
    // scaled so its total magnitude is `fidelity` times the fit's palette
    // gradient — so fidelity reads as strength relative to the fit and stays in
    // range regardless of image, mark count, or fit scale.
    finalize(params, grads, scratch) {
      if (fidelity > 0) {
        let fitMag = 0;
        for (let i = 0; i < count * 3; i++) fitMag += scratch.colorGrad[i] * scratch.colorGrad[i];
        fitMag = Math.sqrt(fitMag);
        const pull = new Float64Array(count * 3);
        let pullMag = 0;
        for (let j = 0; j < count; j++) {
          const den = scratch.centroidDen[j];
          if (den <= 1e-6) continue;
          const centroid = convert(
            [
              scratch.centroidNum[j * 3] / den,
              scratch.centroidNum[j * 3 + 1] / den,
              scratch.centroidNum[j * 3 + 2] / den,
            ],
            OKLab,
            sRGB,
          );
          for (let c = 0; c < 3; c++) {
            const r = scratch.colors[j * 3 + c] - centroid[c];
            pull[j * 3 + c] = r;
            pullMag += r * r;
          }
        }
        pullMag = Math.sqrt(pullMag);
        if (pullMag > 1e-9) {
          const scale = (fidelity * fitMag) / pullMag;
          for (let i = 0; i < count * 3; i++) scratch.colorGrad[i] += scale * pull[i];
        }
      }
      for (let j = 0; j < count; j++) {
        scratch.centroidNum[j * 3] = 0;
        scratch.centroidNum[j * 3 + 1] = 0;
        scratch.centroidNum[j * 3 + 2] = 0;
        scratch.centroidDen[j] = 0;
        const c0 = scratch.colorGrad[j * 3];
        const c1 = scratch.colorGrad[j * 3 + 1];
        const c2 = scratch.colorGrad[j * 3 + 2];
        let meanH = 0;
        const h = new Float64Array(CORNERS);
        for (let k = 0; k < CORNERS; k++) {
          h[k] = (k & 1 ? c0 : 0) + (k & 2 ? c1 : 0) + (k & 4 ? c2 : 0);
          meanH += scratch.weights[j * CORNERS + k] * h[k];
        }
        for (let k = 0; k < CORNERS; k++) {
          grads.palette[j * CORNERS + k] +=
            scratch.weights[j * CORNERS + k] * (h[k] - meanH);
        }
        scratch.colorGrad[j * 3] = 0;
        scratch.colorGrad[j * 3 + 1] = 0;
        scratch.colorGrad[j * 3 + 2] = 0;
      }
    },
  };
}

// Index of the nearest color in `list` to `rgb` (squared distance).
function nearest(list, rgb) {
  let best = 0, bestD = Infinity;
  for (let j = 0; j < list.length; j++) {
    const d =
      (list[j][0] - rgb[0]) ** 2 +
      (list[j][1] - rgb[1]) ** 2 +
      (list[j][2] - rgb[2]) ** 2;
    if (d < bestD) { bestD = d; best = j; }
  }
  return best;
}

// Palette colors without scratch (used only by scratch-free decode paths).
function freshColors(params, count) {
  const colors = new Float64Array(count * 3);
  for (let j = 0; j < count; j++) {
    const rgb = [0, 0, 0];
    paletteColor(params.palette, j, rgb);
    colors[j * 3] = rgb[0];
    colors[j * 3 + 1] = rgb[1];
    colors[j * 3 + 2] = rgb[2];
  }
  return colors;
}
