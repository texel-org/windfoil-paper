import { blurSlope, blurWidth } from './blur.js';
import { packScene } from './prep.js';
import { anchorStyle } from './color-anchors.js';

// Trainable closed quadratic loops with one color and alpha per loop.
// style/styleParams select the color+alpha codec (default: anchor softmax).
// An optional blur group trains each loop's filter size exactly like
// GenericModel; an optional ceiling bounds the per-shape filter.
export class LoopModel {
  constructor({ ax, ay, cx, cy, colorAnchor, n, k, style = null, styleParams = null,
    blur = null, blurFloor = 1e-3, blurCeiling = null }) {
    this.style = style ?? anchorStyle();
    this.params = { ax, ay, cx, cy, ...(blur ? { blur } : {}), ...(styleParams ?? { colorAnchor }) };
    this.blurFloor = blurFloor;
    this.blurCeiling = blurCeiling;
    // Per-model scratch lets a style reuse decode intermediates in pullback.
    this.styleScratch = this.style.createScratch?.(this.params) ?? null;
    this.n = n;
    this.k = k;
    this.maxShapes = n;
    this.maxCurves = n * k;
    this.maxPieces = this.maxCurves * 3;
    this.scratch = {};
    const coords = new ax.constructor(n * k * 6); // contiguous backing for all loops
    this.shapes = Array.from({ length: n }, (_, i) => ({
      curves: coords.subarray(i * k * 6, (i + 1) * k * 6),
      color: [0, 0, 0],
      alpha: 1,
      s: null,
    }));
    this.grads = Object.fromEntries(
      Object.entries(this.params).map(([name, values]) => [name, new values.constructor(values.length)]),
    );
  }

  harden() {
    this.style.harden?.();
  }

  decode() {
    const { ax, ay, cx, cy } = this.params;
    this.style.prepare?.(this.params, this.styleScratch);
    for (let shape = 0; shape < this.n; shape++) {
      const out = this.shapes[shape];
      const base = shape * this.k;
      for (let j = 0; j < this.k; j++) {
        const at = base + j;
        const next = j + 1 < this.k ? at + 1 : base;
        const o = j * 6;
        out.curves[o] = ax[at];
        out.curves[o + 1] = ay[at];
        out.curves[o + 2] = cx[at];
        out.curves[o + 3] = cy[at];
        out.curves[o + 4] = ax[next];
        out.curves[o + 5] = ay[next];
      }
      out.alpha = this.style.decode(this.params, shape, shape, out.color, this.styleScratch);
      if (this.params.blur) out.s = blurWidth(this.params.blur[shape], this.blurFloor, this.blurCeiling);
    }
    return { shapes: this.shapes, scene: packScene(this.shapes, this.scratch) };
  }

  pullback({ curveGrads, shapeGrads, blurGrads }) {
    // A renderer with a frozen parameter group omits that array entirely, so
    // presence is optional but a supplied array must still be the right size.
    if (curveGrads !== undefined && curveGrads?.length !== this.maxCurves * 6) {
      throw new Error('curve gradient size mismatch');
    }
    if (shapeGrads !== undefined && shapeGrads?.length !== this.n * 4) {
      throw new Error('shape gradient size mismatch');
    }
    if (blurGrads !== undefined && blurGrads?.length !== this.n * 2) {
      throw new Error('blur gradient size mismatch');
    }
    const { ax, ay, cx, cy } = this.grads;
    for (const group of this.style.groups) this.grads[group].fill(0); // style pullback accumulates
    this.style.prepare?.(this.params, this.styleScratch);
    for (let shape = 0; shape < this.n; shape++) {
      const base = shape * this.k;
      for (let j = 0; j < this.k; j++) {
        const at = base + j;
        const current = 6 * at;
        const previous = 6 * (j ? at - 1 : base + this.k - 1);
        ax[at] = curveGrads ? curveGrads[current] + curveGrads[previous + 4] : 0;
        ay[at] = curveGrads ? curveGrads[current + 1] + curveGrads[previous + 5] : 0;
        cx[at] = curveGrads ? curveGrads[current + 2] : 0;
        cy[at] = curveGrads ? curveGrads[current + 3] : 0;
      }
      if (shapeGrads) {
        this.style.pullback(
          this.params, shape, shape, shapeGrads, shape * 4, this.grads, this.styleScratch,
        );
      }
      if (blurGrads && this.params.blur) {
        const scale = blurSlope(this.shapes[shape].s, this.blurFloor, this.blurCeiling);
        this.grads.blur[shape] = scale * (blurGrads[2 * shape] + blurGrads[2 * shape + 1]);
      }
    }
    this.style.finalize?.(this.params, this.grads, this.styleScratch);
    this.style.invalidateScratch?.(this.styleScratch);
    return this.grads;
  }
}
