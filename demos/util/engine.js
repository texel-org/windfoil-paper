// Browser optimization engine: renderer lifecycle, generation-safe resets, and
// a single always-live step loop. The L2 page drives it with a finite step
// budget and webcam targets.

import { Renderer } from '../../js/renderer.js';
import { OptimizationSession } from '../../js/optimize.js';
import { buildModel, createInit } from './model.js';
import { buildLineModel } from '../lines/model.js';

// Dispatch the shape / line model builders behind one signature.
export function buildFitModel({ mode, n, size, seed, target = null, background = [1, 1, 1] }) {
  if (mode === 'line') return buildLineModel({ n, size, seed, target });
  return buildModel(createInit({ n, size, k: 8, seed, target, background }));
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export class Engine {
  // build:    () => ({ model, lrs }) using the caller's current mode/count/seed
  // settings: (step) => renderer settings ({ s, scale, origin, bg })
  // onFrame:  called after every accepted step and reset so the page can repaint
  // onError:  surfaces a fatal step error
  constructor(device, { size, build, settings, onFrame = null, onError = null }) {
    this.device = device;
    this.size = size;
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
    this.shapes = null;

    this.playing = false;
    this.continuous = true; // when false the loop pauses once step >= steps
    this.steps = Infinity;
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
    this.session?.setTarget(target);
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
    if (fresh || !nextModel) {
      ({ model: nextModel, lrs: nextRates } = this.build());
      const grow = !nextRenderer ||
        nextModel.maxShapes > this.capacity.shapes ||
        nextModel.maxPieces > this.capacity.pieces ||
        nextModel.maxCurves > this.capacity.curves;
      if (grow) {
        nextRenderer = await Renderer.create(this.device, {
          width: this.size,
          height: this.size,
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
    this.shapes = this.model.decode().shapes;
    this.playing = true;
    this.onFrame?.(this);
  }

  #retireRenderer(value) {
    const pending = this.activeStep;
    if (pending) pending.then(() => value.destroy(), () => value.destroy());
    else value.destroy();
  }

  // Single owner of the optimization cadence; start once after construction.
  async run() {
    for (;;) {
      if (!this.playing || !this.session || (!this.continuous && this.step >= this.steps)) {
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
        this.shapes = update.shapes;
        this.step++;
        const elapsed = performance.now() - started;
        this.averageMs = this.averageMs ? this.averageMs * 0.85 + elapsed * 0.15 : elapsed;
        if (!this.continuous && this.step >= this.steps) this.playing = false;
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
