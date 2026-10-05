import { LoopModel } from '../../js/loop-model.js';
import { anchorStyle } from '../../js/color-anchors.js';
import { learnedPaletteStyle } from '../../js/learned-palette.js';
import { rawStyle } from '../../js/raw-style.js';
import { rng } from './random.js';
import { blurParameter } from '../../js/blur.js';
import { sceneToSVG } from './svg.js';
import { parseColor, rgbToHex } from './color.js';
import {
  arg,
  integerArg,
  opaqueMode,
  paletteCount,
  paletteFidelity,
  positiveArg,
  readText,
} from './runtime.js';

export const BLUR_ANNEAL_FRACTION = 0.55;

export function createInit({
  n,
  size = null,
  width = size,
  height = size,
  k = 8,
  seed = 1,
  target = null,
  background = [1, 1, 1],
}) {
  if (!Number.isInteger(n) || n < 1) throw new Error('n must be a positive integer');
  if (!Number.isInteger(k) || k < 3) throw new Error('k must be at least 3');
  if (!Number.isInteger(width) || width < 1 || !Number.isInteger(height) || height < 1) {
    throw new Error('width and height must be positive integers');
  }
  const random = rng(seed);
  const ax = new Float32Array(n * k), ay = new Float32Array(n * k);
  const cx = new Float32Array(n * k), cy = new Float32Array(n * k);
  const colors = [], alphas = [];
  const nx = Math.max(1, Math.min(n, Math.round(Math.sqrt(n * width / height))));
  const ny = Math.ceil(n / nx);
  const cell = Math.min(width / nx, height / ny);
  const sec = 1 / Math.cos(Math.PI / k);

  for (let shape = 0; shape < n; shape++) {
    const gx = shape % nx, gy = Math.floor(shape / nx);
    const x = (gx + 0.2 + 0.6 * random()) * width / nx;
    const y = (gy + 0.2 + 0.6 * random()) * height / ny;
    const base = cell * (0.35 + 0.35 * random());
    const radii = new Float64Array(k);
    for (let j = 0; j < k; j++) {
      const at = shape * k + j;
      const angle = j * 2 * Math.PI / k;
      radii[j] = base * (0.75 + 0.5 * random());
      ax[at] = x + radii[j] * Math.cos(angle);
      ay[at] = y + radii[j] * Math.sin(angle);
    }
    for (let j = 0; j < k; j++) {
      const at = shape * k + j;
      const angle = (j + 0.5) * 2 * Math.PI / k;
      const radius = 0.5 * (radii[j] + radii[(j + 1) % k]) * sec;
      cx[at] = x + radius * Math.cos(angle);
      cy[at] = y + radius * Math.sin(angle);
    }
    colors.push(sampleColor(target, width, height, x, y, random));
    alphas.push(0.55 + 0.35 * random());
  }
  return {
    n,
    size: width === height ? width : undefined,
    width,
    height,
    k,
    background,
    params: { ax, ay, cx, cy },
    colors,
    alphas,
  };
}

export function sampleColor(target, width, height, x, y, random) {
  if (!target) {
    const value = 0.4 + 0.2 * random();
    return [value, value, value];
  }
  const px = Math.max(0, Math.min(width - 1, Math.round(x)));
  const py = Math.max(0, Math.min(height - 1, Math.round(y)));
  const at = 4 * (py * width + px);
  return [target[at], target[at + 1], target[at + 2]];
}

// The color codec: the raw linear codec when --style=raw, a fixed palette
// when --colors is given, the discovery palette when --color-count > 1,
// otherwise the default free per-shape anchor style. Fixed palettes keep
// per-shape opacity learnable (unless --opaque), so --colors=#ffffff
// --bg=black is white translucent ink.
export function selectStyle({
  colorCount = 1,
  palette = null,
  opaque = false,
  seed = 1,
  fidelity = 0,
  raw = null,
} = {}) {
  if (raw) {
    if (palette?.length || colorCount > 1) {
      throw new Error('--style=raw is exclusive with --colors and --color-count');
    }
    return rawStyle(raw);
  }
  if (palette?.length) {
    if (colorCount > 1) {
      throw new Error('--colors and --color-count are mutually exclusive');
    }
    return learnedPaletteStyle({ fixedColors: palette, opaque, seed });
  }
  return colorCount > 1
    ? learnedPaletteStyle({ count: colorCount, opaque, seed, fidelity })
    : anchorStyle();
}

// --style=raw selects the raw linear codec; its knobs are --channels
// (gray|rgb), --transfer (identity|softplus|sigmoid -- the codomain IS the
// range declaration), --alpha (a fixed opacity, default 1, or 'learned'),
// --color-lr, and --alpha-lr. The raw-only flags are rejected without
// --style=raw so a typo cannot silently fall back to the anchor codec.
export function rawStyleArg(options) {
  const style = arg(options, 'style', null);
  if (style === null || style === 'anchor') {
    for (const flag of ['channels', 'transfer', 'alpha', 'color-lr', 'alpha-lr']) {
      if (flag in options) throw new Error(`--${flag} requires --style=raw`);
    }
    return null;
  }
  if (style !== 'raw') throw new Error("--style must be 'anchor' or 'raw'");
  const alpha = String(arg(options, 'alpha', '1'));
  return {
    channels: String(arg(options, 'channels', 'gray')),
    transfer: String(arg(options, 'transfer', 'identity')),
    alpha: alpha === 'learned' ? 'learned' : Number(alpha),
    lr: 'color-lr' in options ? positiveArg(options, 'color-lr') : 0.05,
    alphaLr: 'alpha-lr' in options ? positiveArg(options, 'alpha-lr') : 0.05,
  };
}

// --colors=#hex,#hex,...: a fixed palette for any anchor-based model (the plot
// model has its own pen handling of the same flag).
export function colorsArg(options) {
  if (!('colors' in options)) return null;
  const list = String(arg(options, 'colors')).split(',').filter(Boolean).map(parseColor);
  if (!list.length) throw new Error('--colors needs at least one color');
  return list;
}

export function buildModel(initial) {
  const style = selectStyle(initial);
  const styleParams = style.init(initial.colors, initial.alphas);
  // Learnable per-loop blur mirrors the stroke model: shapes start at the
  // anneal's initial width and self-anneal, floored at blurFloor.
  const blurFloor = initial.blurFloor ?? 1;
  const blurCeiling = initial.blurCeiling ?? null;
  const blur = initial.learnBlur
    ? new Float32Array(initial.n).fill(
        blurParameter(initial.blurInit ?? 7, blurFloor, blurCeiling))
    : null;
  const model = new LoopModel({
    ...initial.params,
    n: initial.n,
    k: initial.k,
    style,
    styleParams,
    blur,
    blurFloor,
    blurCeiling,
  });
  const width = initial.width ?? initial.size;
  const height = initial.height ?? initial.size;
  const lrs = {
    ax: { lr: 0.5 * width / 128 }, ay: { lr: 0.5 * height / 128 },
    cx: { lr: 0.5 * width / 128 }, cy: { lr: 0.5 * height / 128 },
    ...(blur ? { blur: { lr: 0.1 } } : {}),
    ...style.lrs,
  };
  return { model, lrs };
}

// Shared background: an explicit --bg always wins (blend modes want their
// natural base -- dark for add/screen, light for multiply -- even when a
// target exists); otherwise opaque images tint the backdrop with the target
// mean, and prompt-only (no target) runs fall back to white.
export function meanBackground({ options, target }) {
  if ('bg' in options) return parseColor(arg(options, 'bg'));
  return target?.mean ?? parseColor('white');
}

// A constrained-palette build handle: exposes the discovered colors (read live,
// after training) for the SVG/output. Null in the free-color default.
export function constrainedBuilt(model, config) {
  const fixed = config.palette?.length ?? 0;
  if (!(config.colorCount > 1 || fixed)) return null;
  return {
    constrained: true,
    colorCount: fixed || config.colorCount,
    fixed: fixed > 0,
    opaque: config.opaque,
    paletteColors: () => model.style.paletteColors(model.params),
  };
}

// Model-specific output fields shared by the anchor-based demos: the discovered
// palette when constrained, nothing otherwise.
export function paletteOutputFields(built) {
  if (!built?.constrained) return {};
  return {
    colorCount: built.colorCount,
    opaque: built.opaque,
    colors: built.paletteColors().map(rgbToHex),
  };
}

// Shared SVG: filled quadratic loops over the flat --bg backdrop, clipped to the
// visible frame when the fit was padded. The plot model overrides this with its
// own plotter-ready emitter.
export function sceneSvg(_built, { shapes, width, height, pad = 0, background }) {
  return sceneToSVG(shapes, width, height, {
    background: rgbToHex(background),
    viewBox: { x: pad, y: pad, width, height },
  });
}

// CLI descriptor for the default quadratic-loop shape model — the one the l2
// and clip demos share, and the only model that accepts a shared --init.
export const shapeCli = {
  defaults: { n: 512, steps: 500, size: 512 },
  bandLimited: false,
  supportsInit: true,
  parse(options) {
    const palette = colorsArg(options);
    const raw = rawStyleArg(options);
    return {
      k: integerArg(options, 'k', 8),
      colorCount: paletteCount(options),
      opaque: opaqueMode(options, false),
      fidelity: paletteFidelity(options),
      ...(palette ? { palette } : {}),
      ...(raw ? { raw } : {}),
      // --learn-blur trains a per-loop filter size (starts at --blur-start,
      // never below --blur) instead of following the global anneal.
      learnBlur: 'learn-blur' in options,
    };
  },
  prepareTarget() {},
  background: meanBackground,
  async build({ config, width, height, target, background }) {
    const initial = config.initPath
      ? deserializeInit(JSON.parse(await readText(config.initPath)))
      : createInit({
          n: config.n,
          width,
          height,
          k: config.k,
          seed: config.seed,
          target: target?.rgba,
          background,
        });
    const initialWidth = initial.width ?? initial.size;
    const initialHeight = initial.height ?? initial.size;
    if (
      initialWidth !== width ||
      initialHeight !== height ||
      initial.n !== config.n ||
      initial.k !== config.k
    ) {
      throw new Error('shared init does not match --opt-size, --n, and --k');
    }
    initial.background = background;
    initial.colorCount = config.colorCount;
    initial.palette = config.palette;
    initial.opaque = config.opaque;
    initial.fidelity = config.fidelity;
    initial.raw = config.raw;
    initial.seed = config.seed;
    initial.learnBlur = config.learnBlur;
    initial.blurInit = config.blurStart;
    initial.blurFloor = config.blur;
    const { model, lrs } = buildModel(initial);
    return { model, lrs, built: constrainedBuilt(model, config) };
  },
  toSVG: sceneSvg,
  outputFields: paletteOutputFields,
  primitive() {
    return 'quadratic-loop';
  },
};

// Coarse-to-fine filter schedule: the box-filter width eases from `start` to
// `floor` pixels over the first BLUR_ANNEAL_FRACTION of `steps`, then holds.
export function annealedBlur(step, steps, start, floor) {
  const t = Math.min(step / Math.max(1, steps * BLUR_ANNEAL_FRACTION), 1);
  return floor + (start - floor) * (1 - t) ** 2;
}

export function serializeInit(initial) {
  return {
    ...initial,
    params: Object.fromEntries(Object.entries(initial.params).map(([key, value]) => [key, Array.from(value)])),
  };
}

export function deserializeInit(value) {
  return {
    ...value,
    params: Object.fromEntries(Object.entries(value.params).map(([key, data]) => [key, Float32Array.from(data)])),
  };
}
