// Browser optimization engine with generation-safe resets and a continuous step loop.

import { Renderer } from '../../js/renderer.js';
import { OptimizationSession } from '../../js/optimize.js';
import { buildModel, createInit } from './model.js';
import { buildLineModel } from '../lines/model.js';

// Dispatch the shape / line model builders behind one signature.
export function buildFitModel({
  mode, n, width, height, seed, target = null,
  background = [1, 1, 1], colorCount = 1, opaque = false, learnBlur = false, lrScale = 1,
}) {
  const colorOptions = { colorCount, opaque, learnBlur, blurInit: 7, blurFloor: 1, blurCeiling: 32,
    ...(opaque && colorCount === 1 ? { raw: { channels: 'rgb', transfer: 'sigmoid', alpha: 1 } } : {}),
  };
  const built = mode !== 'shape'
    ? buildLineModel({ n, width, height, seed, target, primitive: mode, ...colorOptions })
    : buildModel({ ...createInit({ n, width, height, k: 8, seed, target, background }), ...colorOptions, seed });
  if (lrScale !== 1) {
    built.lrs = Object.fromEntries(Object.entries(built.lrs).map(([key, rate]) => [
      key, { ...rate, lr: rate.lr * lrScale },
    ]));
  }
  return built;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export class Engine {
  // build:    () => ({ model, lrs }) using the caller's current mode/count/seed
  // settings: (step) => renderer settings ({ s, scale, origin, bg })
  // onFrame:  called after every accepted step and reset so the page can repaint
  // onError:  surfaces a fatal step error
  constructor(device, { width, height, blend = 'src-over', build, settings, onFrame = null, onError = null }) {
    this.device = device;
    this.width = width;
    this.height = height;
    this.blend = blend;
    this.build = build;
    this.settings = settings;
    this.onFrame = onFrame;
    this.onError = onError;

    this.renderer = null;
    this.capacity = { shapes: 0, pieces: 0, curves: 0 };
    this.model = null;
    this.learningRates = null;
    this.session = null;

    this.target = null;
    this.background = [1, 1, 1];
    this.step = 0;
    this.loss = null;
    this.averageMs = 0;

    this.playing = false;
    this.generation = 0;
    this.activeStep = null;
  }

  setTarget(target, background = null) {
    this.target = target;
    if (background) this.background = background;
  }

  // Push a new target into the running session without rebuilding it.
  updateTarget(target, background = null) {
    this.setTarget(target, background);
    // A camera can deliver frames while a resized renderer is being built.
    if (this.session?.renderer.width === this.width && this.session.renderer.height === this.height) {
      this.session.setTarget(target);
    }
  }

  // Rebuild the model (fresh) or just the session (fresh=false), growing the
  // renderer only when capacity demands it. Superseded resets bail out via the
  // generation ticket so concurrent callers never clobber live state.
  async reset(fresh = true) {
    if (!this.target) return;
    const ticket = ++this.generation;
    const currentTarget = this.target;
    const previousRenderer = this.renderer;
    this.playing = false;
    const pending = this.activeStep;
    if (pending) {
      try {
        await pending;
      } catch {
        // A superseded step's error is ignored by the loop as well.
      }
    }
    if (ticket !== this.generation) return;

    let nextModel = this.model;
    let nextRates = this.learningRates;
    let nextRenderer = this.renderer;
    let nextCapacity = this.capacity;
    let createdRenderer = false;
    const resized = nextRenderer && (nextRenderer.width !== this.width || nextRenderer.height !== this.height);
    const blendChanged = nextRenderer && nextRenderer.blend !== this.blend;
    if (fresh || !nextModel || resized || blendChanged) {
      ({ model: nextModel, lrs: nextRates } = this.build());
      const alphaChanged = nextRenderer && nextRenderer.train?.alpha !== nextModel.style.trainsAlpha;
      const blurChanged = nextRenderer && !!nextRenderer.train?.blur !== !!nextModel.params.blur;
      const grow = !nextRenderer || resized || blendChanged || alphaChanged || blurChanged ||
        nextModel.maxShapes > this.capacity.shapes ||
        nextModel.maxPieces > this.capacity.pieces ||
        nextModel.maxCurves > this.capacity.curves;
      if (grow) {
        nextRenderer = await Renderer.create(this.device, {
          width: this.width,
          height: this.height,
          blend: this.blend,
          train: { alpha: nextModel.style.trainsAlpha, blur: !!nextModel.params.blur },
          maxShapes: nextModel.maxShapes,
          maxPieces: nextModel.maxPieces,
          maxCurves: nextModel.maxCurves,
        });
        createdRenderer = true;
        nextCapacity = {
          shapes: nextModel.maxShapes,
          pieces: nextModel.maxPieces,
          curves: nextModel.maxCurves,
        };
      }
    }
    if (ticket !== this.generation) {
      if (createdRenderer) nextRenderer.destroy();
      return;
    }

    this.model = nextModel;
    this.learningRates = nextRates;
    this.renderer = nextRenderer;
    this.capacity = nextCapacity;
    this.session = new OptimizationSession({
      renderer: this.renderer,
      model: this.model,
      lrs: this.learningRates,
      target: currentTarget,
      settings: this.settings,
    });
    if (createdRenderer && previousRenderer && previousRenderer !== this.renderer) {
      this.#retireRenderer(previousRenderer);
    }
    this.step = 0;
    this.loss = null;
    this.averageMs = 0;
    this.playing = true;
    this.onFrame?.(this);
  }

  // Stop optimizing and forget the target, e.g. when the page has no image.
  async clear() {
    this.playing = false;
    this.generation++;
    await this.activeStep?.catch(() => {});
    this.target = null;
    this.session = null;
    this.model = null;
    this.step = 0;
    this.loss = null;
    this.averageMs = 0;
  }

  snapshot(includeShapes = false) {
    if (!this.model || !this.target) return null;
    const decoded = this.model.decode();
    return {
      scene: structuredClone(decoded.scene),
      ...(includeShapes ? { shapes: structuredClone(decoded.shapes) } : {}),
      width: this.width, height: this.height, blend: this.blend, learnBlur: !!this.model.params.blur,
      background: [...this.background],
      maxShapes: this.model.maxShapes, maxPieces: this.model.maxPieces,
      maxCurves: this.model.maxCurves,
    };
  }

  #retireRenderer(value) {
    const pending = this.activeStep;
    if (pending) pending.then(() => value.destroy(), () => value.destroy());
    else value.destroy();
  }

  // Single owner of the optimization cadence; start once after construction.
  async run() {
    for (;;) {
      if (!this.playing || !this.session) {
        await sleep(20);
        continue;
      }
      const started = performance.now();
      const active = this.session;
      const ticket = this.generation;
      const pending = active.step(this.step);
      this.activeStep = pending;
      try {
        const update = await pending;
        if (active !== this.session || ticket !== this.generation) continue;
        this.loss = update.loss;
        this.step++;
        const elapsed = performance.now() - started;
        this.averageMs = this.averageMs ? this.averageMs * 0.85 + elapsed * 0.15 : elapsed;
        this.onFrame?.(this);
      } catch (error) {
        if (active !== this.session || ticket !== this.generation) continue;
        this.playing = false;
        this.onError?.(error);
      } finally {
        if (this.activeStep === pending) this.activeStep = null;
      }
    }
  }
}
