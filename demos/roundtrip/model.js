import { packScene } from '../../js/prep.js';

function finite(value, label) {
  if (!Number.isFinite(value)) throw new Error(`${label} must be finite`);
  return value;
}

function fillRule(value, shape) {
  const rule = value ?? 'nonzero';
  if (rule !== 'nonzero' && rule !== 'evenodd') {
    throw new Error(`shape ${shape}: fillRule must be "nonzero" or "evenodd"`);
  }
  return rule;
}

function copyStyle(shape, index) {
  if (!shape.color || shape.color.length !== 3) {
    throw new Error(`shape ${index}: color must contain three channels`);
  }
  const color = Array.from(shape.color, (value, channel) =>
    finite(value, `shape ${index} color[${channel}]`));
  const alpha = finite(shape.alpha ?? 1, `shape ${index} alpha`);
  if (alpha < 0 || alpha > 1) throw new Error(`shape ${index}: alpha must be between 0 and 1`);
  return { color, alpha, fillRule: fillRule(shape.fillRule, index) };
}

/**
 * Convert parsed closed quadratic paths into shared anchors and one control per
 * segment. The returned values do not alias the parsed SVG scene.
 */
export function roundtripGeometry(shapes) {
  if (!Array.isArray(shapes) || !shapes.length) {
    throw new Error('roundtrip requires at least one filled path');
  }
  let curveCount = 0;
  for (let shape = 0; shape < shapes.length; shape++) {
    const curves = shapes[shape].curves;
    if (!curves || curves.length === 0 || curves.length % 6 !== 0) {
      throw new Error(`shape ${shape}: curves must contain closed quadratic segments`);
    }
    curveCount += curves.length / 6;
  }

  const ax = new Float64Array(curveCount);
  const ay = new Float64Array(curveCount);
  const cx = new Float64Array(curveCount);
  const cy = new Float64Array(curveCount);
  const layout = [];
  let start = 0;

  for (let shape = 0; shape < shapes.length; shape++) {
    const input = shapes[shape];
    const curves = input.curves;
    const count = curves.length / 6;
    const contours = [];
    let contourStart = 0;
    for (let segment = 0; segment < count; segment++) {
      const at = start + segment;
      const offset = 6 * segment;
      const endX = finite(curves[offset + 4], `shape ${shape} segment ${segment} end x`);
      const endY = finite(curves[offset + 5], `shape ${shape} segment ${segment} end y`);
      ax[at] = finite(curves[offset], `shape ${shape} segment ${segment} x`);
      ay[at] = finite(curves[offset + 1], `shape ${shape} segment ${segment} y`);
      cx[at] = finite(curves[offset + 2], `shape ${shape} segment ${segment} control x`);
      cy[at] = finite(curves[offset + 3], `shape ${shape} segment ${segment} control y`);

      const nextSegment = segment + 1;
      const continues = nextSegment < count &&
        endX === finite(curves[6 * nextSegment], `shape ${shape} segment ${segment} next x`) &&
        endY === finite(curves[6 * nextSegment + 1], `shape ${shape} segment ${segment} next y`);
      if (!continues) {
        const first = 6 * contourStart;
        if (endX !== curves[first] || endY !== curves[first + 1]) {
          throw new Error(`shape ${shape}: every contour must be continuous and explicitly closed`);
        }
        contours.push({ start: start + contourStart, count: segment - contourStart + 1 });
        contourStart = nextSegment;
      }
    }
    if (input.contours !== undefined) {
      if (!Array.isArray(input.contours) ||
          input.contours.some((length) => !Number.isInteger(length) || length < 1) ||
          input.contours.reduce((sum, length) => sum + length, 0) !== count) {
        throw new Error(`shape ${shape}: contours must partition its curves`);
      }
      contours.length = 0;
      let localStart = 0;
      for (const length of input.contours) {
        for (let segment = 0; segment < length; segment++) {
          const local = localStart + segment;
          const next = localStart + (segment + 1) % length;
          if (curves[6 * local + 4] !== curves[6 * next] ||
              curves[6 * local + 5] !== curves[6 * next + 1]) {
            throw new Error(`shape ${shape}: every contour must be continuous and explicitly closed`);
          }
        }
        contours.push({ start: start + localStart, count: length });
        localStart += length;
      }
    }
    layout.push({ start, count, contours, ...copyStyle(input, shape) });
    start += count;
  }
  return { params: { ax, ay, cx, cy }, layout, curveCount };
}

/** Return a translated deep copy of a geometry parameter object. */
export function translateRoundtripParams(params, translation = [0, 0]) {
  if (!translation || translation.length !== 2) {
    throw new Error('translation must contain x and y offsets');
  }
  const dx = finite(Number(translation[0]), 'translation x');
  const dy = finite(Number(translation[1]), 'translation y');
  const out = {
    ax: Float64Array.from(params.ax),
    ay: Float64Array.from(params.ay),
    cx: Float64Array.from(params.cx),
    cy: Float64Array.from(params.cy),
  };
  for (let i = 0; i < out.ax.length; i++) {
    out.ax[i] += dx;
    out.ay[i] += dy;
    out.cx[i] += dx;
    out.cy[i] += dy;
  }
  return out;
}

// Fixed-topology, ragged quadratic loops. Source fill styles and rules are
// intentionally fixed so the unshifted source remains exactly representable.
export class RoundtripModel {
  constructor(shapes, { translation = [0, 0] } = {}) {
    const geometry = roundtripGeometry(shapes);
    this.params = translateRoundtripParams(geometry.params, translation);
    this.layout = geometry.layout;
    this.n = this.layout.length;
    this.maxShapes = this.n;
    this.maxCurves = geometry.curveCount;
    this.maxPieces = this.maxCurves * 3;
    this.scratch = {};
    this.shapes = this.layout.map(({ count, contours, color, alpha, fillRule: rule }) => ({
      curves: new Float64Array(count * 6),
      color: color.slice(),
      alpha,
      fillRule: rule,
      contours: contours.map((contour) => contour.count),
      s: null,
    }));
    this.grads = Object.fromEntries(
      Object.entries(this.params).map(([name, values]) => [name, new Float64Array(values.length)]),
    );
  }

  decode() {
    const { ax, ay, cx, cy } = this.params;
    for (let shape = 0; shape < this.layout.length; shape++) {
      const { start, contours } = this.layout[shape];
      const curves = this.shapes[shape].curves;
      for (const contour of contours) {
        for (let segment = 0; segment < contour.count; segment++) {
          const at = contour.start + segment;
          const next = contour.start + (segment + 1) % contour.count;
          const offset = 6 * (at - start);
          curves[offset] = ax[at];
          curves[offset + 1] = ay[at];
          curves[offset + 2] = cx[at];
          curves[offset + 3] = cy[at];
          curves[offset + 4] = ax[next];
          curves[offset + 5] = ay[next];
        }
      }
    }
    return { shapes: this.shapes, scene: packScene(this.shapes, this.scratch) };
  }

  pullback({ curveGrads, shapeGrads }) {
    if (curveGrads?.length !== this.maxCurves * 6) {
      throw new Error('curve gradient size mismatch');
    }
    if (shapeGrads?.length !== this.n * 4) {
      throw new Error('shape gradient size mismatch');
    }
    for (const values of Object.values(this.grads)) values.fill(0);
    for (const { contours } of this.layout) {
      for (const contour of contours) {
        for (let segment = 0; segment < contour.count; segment++) {
          const at = contour.start + segment;
          const previous = contour.start + (segment + contour.count - 1) % contour.count;
          const currentOffset = 6 * at;
          const previousOffset = 6 * previous;
          this.grads.ax[at] = curveGrads[currentOffset] + curveGrads[previousOffset + 4];
          this.grads.ay[at] = curveGrads[currentOffset + 1] + curveGrads[previousOffset + 5];
          this.grads.cx[at] = curveGrads[currentOffset + 2];
          this.grads.cy[at] = curveGrads[currentOffset + 3];
        }
      }
    }
    return this.grads;
  }
}

export function roundtripLearningRates(width, height, base = 0.5) {
  finite(width, 'coordinate width');
  finite(height, 'coordinate height');
  finite(base, 'geometry learning-rate base');
  if (!(width > 0) || !(height > 0) || !(base > 0)) {
    throw new Error('coordinate dimensions and geometry learning rate must be positive');
  }
  return {
    ax: { lr: base * width / 128 },
    ay: { lr: base * height / 128 },
    cx: { lr: base * width / 128 },
    cy: { lr: base * height / 128 },
  };
}

export function buildRoundtripModel(shapes, {
  translation = [0, 0],
  width,
  height,
  learningRate = 0.5,
} = {}) {
  return {
    model: new RoundtripModel(shapes, { translation }),
    lrs: roundtripLearningRates(width, height, learningRate),
  };
}
