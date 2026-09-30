import { GenericModel } from '../../js/generic/model.js';
import { circle, rectStroke, roundCapsule } from '../../js/generic/primitives.js';
import { rng } from '../util/random.js';
import {
  colorsArg,
  constrainedBuilt,
  meanBackground,
  paletteOutputFields,
  rawStyleArg,
  sampleColor,
  sceneSvg,
  selectStyle,
} from '../util/model.js';
import { arg, opaqueMode, paletteCount, paletteFidelity } from '../util/runtime.js';

// The stroke primitives this model can train, keyed by --primitive. k is the
// curve count each element contributes (the renderer capacity unit); `stroke`
// primitives train two endpoints, `point` trains a position and radius only.
export const LINE_PRIMITIVES = {
  capsule: { k: 6, name: 'round-capsule' },
  line: { k: 4, name: 'rect-stroke' },
  point: { k: 8, name: 'point' },
};

export function buildLineModel({
  n,
  size = null,
  width = size,
  height = size,
  seed = 1,
  target = null,
  primitive = 'capsule',
  colorCount = 1,
  palette = null,
  opaque = false,
  fidelity = 0,
  raw = null,
  learnBlur = false,
  blurInit = 7,
  blurFloor = 1,
}) {
  if (!LINE_PRIMITIVES[primitive]) {
    throw new Error(
      `--primitive must be one of shape, ${Object.keys(LINE_PRIMITIVES).join(', ')}`,
    );
  }
  const point = primitive === 'point';
  const random = rng(seed);
  const x0 = new Float32Array(n), y0 = new Float32Array(n);
  const x1 = point ? null : new Float32Array(n), y1 = point ? null : new Float32Array(n);
  const strokeWidth = new Float32Array(n);
  const colors = [], alphas = [];
  const scale = Math.min(width, height);
  // Width trains unbounded but decodes through an algebraic softplus, so the
  // rendered thickness stays strictly positive: a negative width flips the
  // capsule's cap offsets (and a disc's winding) and turns the outline inside
  // out. beta — the softplus curvature scale — matches the smallest seeded
  // width, keeping the decode near-identity over the seeded range while raw
  // values glide toward zero instead of crossing it.
  const widthBeta = 0.003 * scale;
  for (let i = 0; i < n; i++) {
    const x = random() * width, y = random() * height;
    const angle = random() * Math.PI * 2;
    const length = scale * (0.03 + 0.08 * random());
    if (point) {
      x0[i] = x;
      y0[i] = y;
    } else {
      x0[i] = x - Math.cos(angle) * length * 0.5;
      y0[i] = y - Math.sin(angle) * length * 0.5;
      x1[i] = x + Math.cos(angle) * length * 0.5;
      y1[i] = y + Math.sin(angle) * length * 0.5;
    }
    strokeWidth[i] = softplusInverse(scale * (0.003 + 0.012 * random()), widthBeta);
    colors.push(sampleColor(target, width, height, x, y, random));
    alphas.push(0.45 + 0.4 * random());
  }
  const style = selectStyle({ colorCount, palette, opaque, seed, fidelity, raw });
  const styleParams = style.init(colors, alphas);
  const stroke = primitive === 'line' ? rectStroke : roundCapsule;
  // Learnable per-shape blur self-anneals: every stroke starts at the global
  // anneal's initial width and its gradient decides how crisp it ends up. The
  // per-shape filter overrides the annealed global one, so the renderer needs
  // no schedule coupling; the floor keeps sigma positive and bounded away
  // from a degenerate filter.
  const blur = learnBlur
    ? new Float32Array(n).fill(Math.log(Math.max(blurInit - blurFloor, 1e-3)))
    : null;
  const params = {
    x0, y0,
    ...(point ? {} : { x1, y1 }),
    width: strokeWidth,
    ...(blur ? { blur } : {}),
    ...styleParams,
  };
  const model = new GenericModel({
    params,
    style,
    blurFloor,
    build: (tape, p) => Array.from({ length: n }, (_, i) => {
      const w = softplus(tape, p.width[i], widthBeta);
      return {
        curves: point
          ? circle(tape, p.x0[i], p.y0[i], w)
          : stroke(tape, p.x0[i], p.y0[i], p.x1[i], p.y1[i], w),
        color: i,
        alpha: i,
      };
    }),
  });
  return {
    model,
    lrs: {
      x0: { lr: 0.5 * width / 128 }, y0: { lr: 0.5 * height / 128 },
      ...(point
        ? {}
        : { x1: { lr: 0.5 * width / 128 }, y1: { lr: 0.5 * height / 128 } }),
      width: { lr: 0.06 * scale / 128 },
      ...(blur ? { blur: { lr: 0.1 } } : {}),
      ...style.lrs,
    },
  };
}

// CLI descriptor for the JavaScript-only stroke/point model family.
export const lineCli = {
  defaults: { n: 512, steps: 500, size: 128 },
  blurFloor: 1,
  bandLimited: false,
  supportsInit: false,
  parse(options) {
    const primitive = String(arg(options, 'primitive', 'capsule'));
    if (!LINE_PRIMITIVES[primitive]) {
      throw new Error(
        `--primitive must be one of shape, ${Object.keys(LINE_PRIMITIVES).join(', ')}`,
      );
    }
    const palette = colorsArg(options);
    const raw = rawStyleArg(options);
    return {
      k: LINE_PRIMITIVES[primitive].k,
      ...(primitive !== 'capsule' ? { primitive } : {}),
      colorCount: paletteCount(options),
      opaque: opaqueMode(options, false),
      fidelity: paletteFidelity(options),
      ...(palette ? { palette } : {}),
      ...(raw ? { raw } : {}),
      // --learn-blur trains a per-stroke filter size (starts at --blur,
      // floored at --blur-floor) instead of following the global anneal.
      learnBlur: 'learn-blur' in options,
    };
  },
  prepareTarget() {},
  background: meanBackground,
  async build({ config, width, height, target }) {
    const { model, lrs } = buildLineModel({
      n: config.n,
      width,
      height,
      seed: config.seed,
      target: target?.rgba,
      primitive: config.primitive ?? 'capsule',
      colorCount: config.colorCount,
      palette: config.palette,
      opaque: config.opaque,
      fidelity: config.fidelity,
      raw: config.raw,
      learnBlur: config.learnBlur,
      blurInit: config.blur,
      blurFloor: config.blurFloor,
    });
    return {
      model,
      lrs,
      built: {
        palette: constrainedBuilt(model, config),
        primitive: config.primitive ?? 'capsule',
      },
    };
  },
  toSVG: sceneSvg,
  outputFields(built) {
    return paletteOutputFields(built?.palette);
  },
  primitive(built) {
    return LINE_PRIMITIVES[built?.primitive ?? 'capsule'].name;
  },
};

// Smooth positive map built from the tape's algebraic ops (no exp available):
// 0.5 * (x + sqrt(x^2 + beta^2)). Behaves like softplus — identity for
// x >> beta, -> 0+ for x -> -inf — and its derivative is an algebraic sigmoid
// in (0, 1), so the gradient never changes sign.
function softplus(tape, x, beta) {
  const root = tape.sqrt(tape.addc(tape.mul(x, x), beta * beta));
  return tape.mulc(tape.add(x, root), 0.5);
}

// Exact inverse for positive widths, so seeds decode to the width they were
// sampled at.
function softplusInverse(width, beta) {
  return width - (beta * beta) / (4 * width);
}
