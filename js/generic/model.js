import { packScene } from '../prep.js';
import { anchorStyle } from '../color-anchors.js';
import { Tape } from './autodiff.js';

// Fixed-topology geometry described as ordinary scalar tape expressions.
// style selects the color+alpha codec (default: anchor softmax).
//
// An optional params.blur group (one value per shape) trains each shape's
// filter size: sigma = blurFloor + exp(blur[i]), written to the shape's
// per-shape `s`, which overrides the global (annealed) filter in the
// renderer. Initialize blur at the anneal start and shapes learn their own
// coarse-to-fine schedule; the exp keeps sigma above the floor.
export class GenericModel {
  constructor({ params, build, style = null, blurFloor = 1e-3 }) {
    this.style = style ?? anchorStyle();
    this.params = params;
    this.blurFloor = blurFloor;
    // Per-model scratch lets a style reuse decode intermediates in pullback.
    this.styleScratch = this.style.createScratch?.(params) ?? null;
    this.scratch = {};
    const styleGroups = new Set(this.style.groups);
    const tape = new Tape();
    const nodes = {};
    for (const [group, values] of Object.entries(params)) {
      if (styleGroups.has(group) || group === 'blur') continue;
      nodes[group] = Array.from(values, (value, index) => tape.input(value, group, index));
    }
    const descriptions = build(tape, nodes);
    tape.seal();
    this.tape = tape;
    this.curveNodes = descriptions.map(({ curves }) => Int32Array.from(curves));
    this.styles = descriptions.map(({ color, alpha }) => ({ color, alpha }));
    this.shapes = descriptions.map(({ curves }) => ({
      curves: new Float64Array(curves.length),
      color: [0, 0, 0],
      alpha: 1,
      s: null,
    }));
    this.n = this.shapes.length;
    this.maxShapes = this.n;
    this.maxCurves = this.curveNodes.reduce((sum, curves) => sum + curves.length / 6, 0);
    this.maxPieces = this.maxCurves * 3;
    this.grads = Object.fromEntries(
      Object.entries(params).map(([group, values]) => [group, new values.constructor(values.length)]),
    );
  }

  harden() {
    this.style.harden?.();
  }

  decode() {
    this.tape.load(this.params);
    this.tape.forward();
    this.style.prepare?.(this.params, this.styleScratch);
    const blur = this.params.blur ?? null;
    for (let shape = 0; shape < this.n; shape++) {
      const style = this.styles[shape];
      const nodes = this.curveNodes[shape];
      const out = this.shapes[shape];
      for (let i = 0; i < nodes.length; i++) out.curves[i] = this.tape.value[nodes[i]];
      out.alpha = this.style.decode(
        this.params, style.color, style.alpha, out.color, this.styleScratch,
      );
      if (blur) out.s = this.blurFloor + Math.exp(blur[shape]);
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
    for (const values of Object.values(this.grads)) values.fill(0);
    this.tape.grad.fill(0);
    let offset = 0;
    if (curveGrads) {
      for (const nodes of this.curveNodes) {
        for (let i = 0; i < nodes.length; i++) this.tape.grad[nodes[i]] += curveGrads[offset++];
      }
    }
    this.tape.backward();
    this.tape.readGrads(this.grads);

    this.style.prepare?.(this.params, this.styleScratch);
    if (shapeGrads) {
      for (let shape = 0; shape < this.n; shape++) {
        const style = this.styles[shape];
        this.style.pullback(
          this.params, style.color, style.alpha, shapeGrads, shape * 4, this.grads,
          this.styleScratch,
        );
      }
    }
    if (blurGrads && this.params.blur) {
      // Isotropic sigma feeds both filter axes, so dL/dsigma = dsx + dsy;
      // chain through sigma = floor + exp(u) via the decoded value.
      for (let shape = 0; shape < this.n; shape++) {
        const scale = this.shapes[shape].s - this.blurFloor;
        this.grads.blur[shape] = scale * (blurGrads[2 * shape] + blurGrads[2 * shape + 1]);
      }
    }
    this.style.finalize?.(this.params, this.grads, this.styleScratch);
    this.style.invalidateScratch?.(this.styleScratch);
    return this.grads;
  }
}
