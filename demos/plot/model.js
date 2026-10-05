import { GenericModel } from "../../js/generic/model.js";
import { circle, roundCapsule } from "../../js/generic/primitives.js";
import { learnedPaletteStyle } from "../../js/learned-palette.js";
import { plotMarkersToSVG } from "../util/svg.js";
import { parseColor, rgbToHex } from "../util/color.js";
import { toGrayscale } from "../util/image.js";
import { rng } from "../util/random.js";
import {
  arg,
  argList,
  opaqueMode,
  paletteCount,
  paletteFidelity,
  positiveArg,
} from "../util/runtime.js";

// A fixed set of pen colors. Every marker is assigned one palette entry at init
// and keeps it — geometry is trained, color is not — so the markers of each pen
// migrate to wherever that ink best reduces the error. A single-entry palette
// (the default black) reproduces a one-pen plot.
function paletteStyle(palette) {
  return {
    kind: "palette",
    groups: [],
    lrs: {},
    init: () => ({}),
    decode(_params, colorIndex, _alphaIndex, outColor) {
      const color = palette[colorIndex];
      outColor[0] = color[0];
      outColor[1] = color[1];
      outColor[2] = color[2];
      return 1;
    },
    pullback() {},
  };
}

// The number of round segments used to approximate a stipple disc. Points are
// small (a pen nib), so a coarse ring is plenty.
const POINT_SEGMENTS = 12;

// Pen-plotter markers in two flavours, chosen by `mode`:
//   line  — straight round-capped strokes of a fixed pen width, each trained as
//           a center point, an angle, and a length softly bounded to
//           [minLenCm, maxLenCm];
//   point — stipple discs the radius of the pen, trained only in position.
// Each marker draws in one of the palette colors (default black). Parameters
// live in centimeters; the build folds the cm-to-pixel scale into the tape so
// gradients arrive in cm.
export function buildPlotModel({
  n,
  width,
  height,
  mode = "line",
  canvasCm = 18,
  penMm = 0.45,
  minLenCm = 0.1,
  maxLenCm = 1,
  colors = null,
  colorCount = 1,
  opaque = true,
  fidelity = 0,
  pad = 0,
  seed = 1,
  target = null,
}) {
  if (!Number.isInteger(n) || n < 1)
    throw new Error("n must be a positive integer");
  if (mode !== "line" && mode !== "point")
    throw new Error('mode must be "line" or "point"');
  if (mode === "line" && !(minLenCm > 0 && maxLenCm >= minLenCm))
    throw new Error("line mode requires 0 < min-len <= max-len");
  const palette = (colors && colors.length ? colors : [[0, 0, 0]]).map((c) =>
    c.slice(0, 3),
  );
  // width/height are the padded optimization canvas; the paper is the visible
  // window inside it, and the pen and canvas sizes anchor to the paper. Marker
  // coordinates are stored in paper centimeters (so pad marks land outside
  // [0, widthCm] and the SVG paper viewBox clips them), and the build folds a
  // +pad pixel offset in so they render into the padded canvas.
  const visibleWidth = width - 2 * pad;
  const visibleHeight = height - 2 * pad;
  const pxPerCm = Math.max(visibleWidth, visibleHeight) / canvasCm;
  const widthCm = visibleWidth / pxPerCm;
  const heightCm = visibleHeight / pxPerCm;
  const penWidthCm = penMm / 10;
  const radiusPx = 0.5 * penWidthCm * pxPerCm;
  const halfMinCm = 0.5 * minLenCm;
  const halfMaxCm = 0.5 * maxLenCm;
  const random = rng(seed);
  // Paper cm -> padded pixels: multiply by the scale, then shift past the pad.
  const toPixels = (tape, value) => tape.addc(tape.mulc(value, pxPerCm), pad);

  // Every marker starts at an ink-worthy point and is painted with the palette
  // color nearest the target there, so each pen seeds where it fits best.
  const px = new Float32Array(n);
  const py = new Float32Array(n);
  const pen = new Int32Array(n);
  const sampled = new Array(n);
  for (let i = 0; i < n; i++) {
    const sample = samplePoint(random, target, width, height, pad, pxPerCm);
    px[i] = sample.x;
    py[i] = sample.y;
    pen[i] = nearestPalette(palette, sample.color);
    sampled[i] = sample.color;
  }

  // Constrained mode (--color-count > 1) discovers the palette: a learned,
  // shared set of pen colors plus a per-shape assignment, so each shape snaps to
  // one of N optimized inks. Otherwise the palette is fixed and each shape keeps
  // its seeded pen. Plot ink is always opaque.
  const constrained = colorCount > 1;
  const style = constrained
    ? learnedPaletteStyle({ count: colorCount, opaque, seed, fidelity })
    : paletteStyle(palette);
  const styleParams = constrained ? style.init(sampled) : {};
  const colorIndexOf = constrained ? (i) => i : (i) => pen[i];
  // Translucent assignments carry an extra transparent anchor, widening each
  // shape's assignment row.
  const assignStride = opaque ? colorCount : colorCount + 1;
  // Per-shape SVG fill: the discovered palette entry each shape snaps to, or its
  // fixed pen. Reads live params so it reflects the trained result.
  const markerColors = () => {
    if (!constrained) return (i) => palette[pen[i]];
    const learned = style.paletteColors(model.params);
    return (i) => learned[argmaxAssign(model.params.assign, i, colorCount, assignStride)];
  };
  let params, build, lrs, markers;
  if (mode === "line") {
    const theta = new Float32Array(n);
    const len = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      theta[i] = random() * 2 * Math.PI;
      const lengthCm = minLenCm + (maxLenCm - minLenCm) * random();
      // Invert the length squash so the initial stroke is exactly lengthCm.
      const t = (0.5 * lengthCm - halfMinCm) / (halfMaxCm - halfMinCm + 1e-12);
      len[i] = squashInverse(t);
    }
    params = { px, py, theta, len, ...styleParams };
    build = (tape, p) =>
      Array.from({ length: n }, (_, i) => {
        const cx = toPixels(tape, p.px[i]);
        const cy = toPixels(tape, p.py[i]);
        const halfPx = halfLength(tape, p.len[i], halfMinCm, halfMaxCm, pxPerCm);
        const ex = tape.mul(tape.cos(p.theta[i]), halfPx);
        const ey = tape.mul(tape.sin(p.theta[i]), halfPx);
        return {
          curves: roundCapsule(
            tape,
            tape.sub(cx, ex),
            tape.sub(cy, ey),
            tape.add(cx, ex),
            tape.add(cy, ey),
            tape.const_(radiusPx),
          ),
          color: colorIndexOf(i),
          alpha: colorIndexOf(i),
        };
      });
    lrs = {
      px: { lr: (0.5 * widthCm) / 128 },
      py: { lr: (0.5 * heightCm) / 128 },
      theta: { lr: 0.03 },
      len: { lr: 0.05 },
      ...style.lrs,
    };
    markers = () => {
      const colorFor = markerColors();
      return Array.from({ length: n }, (_, i) => {
        const half = halfLengthCm(len[i], halfMinCm, halfMaxCm);
        const ex = Math.cos(theta[i]) * half;
        const ey = Math.sin(theta[i]) * half;
        return {
          mode: "line",
          x0: px[i] - ex,
          y0: py[i] - ey,
          x1: px[i] + ex,
          y1: py[i] + ey,
          color: colorFor(i),
        };
      });
    };
  } else {
    params = { px, py, ...styleParams };
    build = (tape, p) =>
      Array.from({ length: n }, (_, i) => ({
        curves: circle(
          tape,
          toPixels(tape, p.px[i]),
          toPixels(tape, p.py[i]),
          tape.const_(radiusPx),
          POINT_SEGMENTS,
        ),
        color: colorIndexOf(i),
        alpha: colorIndexOf(i),
      }));
    lrs = {
      px: { lr: (0.5 * widthCm) / 128 },
      py: { lr: (0.5 * heightCm) / 128 },
      ...style.lrs,
    };
    markers = () => {
      const colorFor = markerColors();
      return Array.from({ length: n }, (_, i) => ({
        mode: "point",
        x: px[i],
        y: py[i],
        color: colorFor(i),
      }));
    };
  }

  const model = new GenericModel({ params, style, build });
  return {
    model,
    lrs,
    mode,
    minLenCm,
    maxLenCm,
    constrained,
    colorCount,
    opaque,
    palette,
    // The effective palette: the discovered colors (constrained) or the fixed
    // pens; read live so it reflects the trained result.
    paletteColors: () =>
      constrained ? style.paletteColors(model.params) : palette,
    canvas: { widthCm, heightCm, penWidthCm },
    markers,
  };
}

// Index of the palette color a shape's assignment softmax peaks at (over the
// `count` color anchors; `stride` is the full row width including transparent).
function argmaxAssign(assign, i, count, stride) {
  const base = i * stride;
  let best = 0;
  for (let j = 1; j < count; j++) if (assign[base + j] > assign[base + best]) best = j;
  return best;
}

// CLI descriptor for the pen-plotter model: owns its flags, the grayscale
// rule, the white paper backdrop, band-limited tone matching, and the
// plotter-ready SVG, so the shared runner needs no plot-specific branches.
export const plotCli = {
  defaults: { n: 4000, steps: 800, size: 512 },
  // A fixed pen expresses tone only as mark density, so with --blur above 1
  // the target is box-filtered like the render and never fully sharpens.
  bandLimited: true,
  supportsInit: false,
  parse(options) {
    if ("colors" in options && "color-count" in options) {
      throw new Error(
        "--colors fixes the palette and --color-count learns one; pass only one",
      );
    }
    return {
      k: undefined,
      mode: plotMode(options),
      // Flags override the model defaults only when given explicitly.
      canvasCm: "canvas" in options ? positiveArg(options, "canvas") : undefined,
      penMm: "pen" in options ? positiveArg(options, "pen") : undefined,
      minLenCm: "min-len" in options ? positiveArg(options, "min-len") : undefined,
      maxLenCm: "max-len" in options ? positiveArg(options, "max-len") : undefined,
      // No --colors fits a single black pen; a list names the pen colors.
      colors:
        "colors" in options ? argList(options, "colors").map(parseColor) : null,
      // --color-count > 1 discovers that many pen colors instead of fixing them.
      colorCount: paletteCount(options),
      // Plot ink is opaque by default; --translucent lets marks fade and blend.
      opaque: opaqueMode(options, true),
      // --palette-fidelity pulls discovered colors toward the image's actual ones.
      fidelity: paletteFidelity(options),
      grayscale: "grayscale" in options,
    };
  },
  // A single black pen fits a b&w target. Fixed or discovered color palettes
  // keep the target in color, unless --grayscale is explicitly requested.
  prepareTarget(target, config) {
    const singlePen = !config.colors && !(config.colorCount > 1);
    if (singlePen || config.grayscale) toGrayscale(target);
  },
  // Ink sits on white paper regardless of the target's mean tone.
  background({ options }) {
    return parseColor(arg(options, "bg", "white"));
  },
  build({ config, width, height, pad, target }) {
    const plot = buildPlotModel({
      n: config.n,
      width,
      height,
      mode: config.mode,
      canvasCm: config.canvasCm,
      penMm: config.penMm,
      minLenCm: config.minLenCm,
      maxLenCm: config.maxLenCm,
      colors: config.colors,
      colorCount: config.colorCount,
      opaque: config.opaque,
      fidelity: config.fidelity,
      pad,
      seed: config.seed,
      target: target?.rgba,
    });
    return { model: plot.model, lrs: plot.lrs, built: plot };
  },
  // Marker geometry comes from the model; color and opacity come from the final
  // (hardened) decode, so translucent marks carry their trained alpha. The paper
  // matches the fit's --bg (white by default).
  toSVG(built, { shapes, background }) {
    const markers = built.markers().map((m, i) => ({
      ...m,
      color: shapes[i].color,
      opacity: shapes[i].alpha,
    }));
    return plotMarkersToSVG(markers, {
      widthCm: built.canvas.widthCm,
      heightCm: built.canvas.heightCm,
      penWidthCm: built.canvas.penWidthCm,
      background: rgbToHex(background),
    });
  },
  outputFields(built) {
    return {
      mode: built.mode,
      canvasCm: [built.canvas.widthCm, built.canvas.heightCm],
      penMm: built.canvas.penWidthCm * 10,
      minLenCm: built.mode === "line" ? built.minLenCm : undefined,
      maxLenCm: built.mode === "line" ? built.maxLenCm : undefined,
      colorCount: built.constrained ? built.colorCount : undefined,
      opaque: built.constrained ? built.opaque : undefined,
      colors: built.paletteColors().map(rgbToHex),
    };
  },
  primitive(built) {
    return built.mode === "point" ? "stipple-disc" : "round-capsule";
  },
};

// --primitive selects the marker geometry (unified with the other commands'
// geometry flag): "line" trains round-capped strokes, "point" trains stipple
// discs. Kept as the model's internal `mode` downstream.
function plotMode(options) {
  const value = String(arg(options, "primitive", "line"));
  if (value !== "line" && value !== "point")
    throw new Error('--primitive must be "line" or "point"');
  return value;
}


// Smooth 0..1 sigmoid built from the tape's algebraic ops (no exp available):
// 0.5 + 0.5 * len / sqrt(len^2 + 1). Maps len 0 -> 0.5, +-inf -> 1/0.
function squash(tape, len) {
  const denom = tape.sqrt(tape.addc(tape.mul(len, len), 1));
  return tape.addc(tape.mulc(tape.div(len, denom), 0.5), 0.5);
}

// CPU mirror of squash() and its inverse, used for init and SVG readout.
function squashCpu(len) {
  return 0.5 + (0.5 * len) / Math.sqrt(len * len + 1);
}
function squashInverse(t) {
  const clamped = Math.max(0.02, Math.min(0.98, t));
  const u = 2 * clamped - 1;
  return u / Math.sqrt(1 - u * u);
}

// Trained half-length in pixels, softly bounded to [halfMin, halfMax] in cm.
function halfLength(tape, len, halfMinCm, halfMaxCm, pxPerCm) {
  const s = squash(tape, len);
  return tape.addc(
    tape.mulc(s, (halfMaxCm - halfMinCm) * pxPerCm),
    halfMinCm * pxPerCm,
  );
}
function halfLengthCm(len, halfMinCm, halfMaxCm) {
  return halfMinCm + (halfMaxCm - halfMinCm) * squashCpu(len);
}

// Index of the palette color nearest an RGB target sample (squared distance).
function nearestPalette(palette, rgb) {
  let best = 0;
  let bestDistance = Infinity;
  for (let k = 0; k < palette.length; k++) {
    const c = palette[k];
    const distance =
      (c[0] - rgb[0]) ** 2 + (c[1] - rgb[1]) ** 2 + (c[2] - rgb[2]) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = k;
    }
  }
  return best;
}

// Rejection-sample a marker center toward dark (ink-worthy) target regions of
// the padded canvas, returning the point in paper centimeters (pad regions land
// outside the paper) and the target color there for pen assignment.
function samplePoint(random, target, width, height, pad, pxPerCm) {
  const paper = (u, v) => ({
    x: (u * width - pad) / pxPerCm,
    y: (v * height - pad) / pxPerCm,
  });
  for (let tries = 0; tries < 40; tries++) {
    const u = random();
    const v = random();
    if (!target) return { ...paper(u, v), color: [0, 0, 0] };
    const cx = Math.max(0, Math.min(width - 1, Math.round(u * width)));
    const cy = Math.max(0, Math.min(height - 1, Math.round(v * height)));
    const at = 4 * (cy * width + cx);
    const rgb = [target[at], target[at + 1], target[at + 2]];
    const darkness =
      1 - (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]);
    if (random() < darkness) return { ...paper(u, v), color: rgb };
  }
  const u = random();
  const v = random();
  const cx = Math.max(0, Math.min(width - 1, Math.round(u * width)));
  const cy = Math.max(0, Math.min(height - 1, Math.round(v * height)));
  const at = target ? 4 * (cy * width + cx) : 0;
  const color = target
    ? [target[at], target[at + 1], target[at + 2]]
    : [0, 0, 0];
  return { ...paper(u, v), color };
}
