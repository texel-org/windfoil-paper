import { Adam } from './adam.js';
import { tonemapChain, tonemapImage } from './tonemap.js';

export class OptimizationSession {
  // target is either an image or a per-step image schedule (step) => image,
  // re-uploaded only when the returned image changes identity.
  constructor({ renderer, model, lrs, settings, target = null, loss = null,
    views = () => [{ scale: 1, origin: [0, 0], weight: 1 }], exposure = null }) {
    if (!target && !loss) throw new Error('target or loss is required');
    this.renderer = renderer;
    this.model = model;
    this.settings = settings;
    this.targetFn = typeof target === 'function' ? target : null;
    this.target = this.targetFn ? this.targetFn(0) : target;
    this.loss = loss;
    this.views = views;
    this.adam = new Adam(model.params, lrs);
    // A tonemapped renderer trains its exposure as one scene-level scalar,
    // in log space so k stays positive; a plain scalar Adam alongside the
    // model's. The white-normalized operators train their white point W the
    // same way (omega = log W). The fused L2 path returns dL/dk (and dL/dW)
    // from the kernel; the external (CLIP) path gets them from the host-side
    // chain in externalStep.
    this.exposure = renderer.tonemapCode
      ? {
          kappa: Math.log(exposure?.k ?? 1),
          lr: exposure?.lr ?? 0.05,
          m: 0, v: 0, t: 0,
          scratch: null,
          ...(renderer.hasWhite
            ? {
                omega: Math.log(exposure?.w ?? 1),
                wLr: exposure?.wLr ?? exposure?.lr ?? 0.05,
                wm: 0, wv: 0, wt: 0,
              }
            : {}),
        }
      : null;
    if (this.target) renderer.uploadTarget(this.target);
  }

  get exposureK() {
    return this.exposure ? Math.exp(this.exposure.kappa) : null;
  }

  get exposureW() {
    return this.exposure?.omega === undefined ? null : Math.exp(this.exposure.omega);
  }

  #stepExposure(kGrad, wGrad) {
    const e = this.exposure;
    const g = kGrad * Math.exp(e.kappa); // d/dkappa via k = exp(kappa)
    e.t++;
    e.m = 0.9 * e.m + 0.1 * g;
    e.v = 0.999 * e.v + 0.001 * g * g;
    const mHat = e.m / (1 - 0.9 ** e.t);
    const vHat = e.v / (1 - 0.999 ** e.t);
    e.kappa -= e.lr * mHat / (Math.sqrt(vHat) + 1e-8);
    if (e.omega === undefined || !Number.isFinite(wGrad)) return;
    const gw = wGrad * Math.exp(e.omega); // d/domega via W = exp(omega)
    e.wt++;
    e.wm = 0.9 * e.wm + 0.1 * gw;
    e.wv = 0.999 * e.wv + 0.001 * gw * gw;
    const wmHat = e.wm / (1 - 0.9 ** e.wt);
    const wvHat = e.wv / (1 - 0.999 ** e.wt);
    e.omega -= e.wLr * wmHat / (Math.sqrt(wvHat) + 1e-8);
  }

  setTarget(target) {
    this.target = target;
    this.renderer.uploadTarget(target);
  }

  async step(step) {
    if (this.targetFn) {
      const scheduled = this.targetFn(step);
      if (scheduled !== this.target) this.setTarget(scheduled);
    }
    if (this.exposure) {
      this.renderer.setExposure(
        Math.exp(this.exposure.kappa),
        this.exposure.omega === undefined ? undefined : Math.exp(this.exposure.omega));
    }
    const decoded = this.model.decode();
    let result;
    if (this.target) {
      this.renderer.uploadScene(decoded.scene, this.settings(step));
      result = await this.renderer.stepGpuLoss();
    } else {
      result = await externalStep(
        this.renderer, decoded.scene, this.loss, this.views(step), step, this.exposure);
    }
    if (!Number.isFinite(result.loss)) throw new Error(`non-finite loss at step ${step}`);
    const grads = this.model.pullback(result, decoded.shapes, decoded.scene);
    this.adam.step(this.model.params, grads);
    if (this.exposure && Number.isFinite(result.kGrad)) {
      this.#stepExposure(result.kGrad, result.wGrad);
    }
    return { loss: result.loss, shapes: decoded.shapes };
  }
}

export async function optimize({
  renderer,
  model,
  lrs,
  settings,
  steps = Infinity,
  seconds = null,
  target = null,
  loss = null,
  views = () => [{ scale: 1, origin: [0, 0], weight: 1 }],
  onStep = null,
  exposure = null,
}) {
  const session = new OptimizationSession({
    renderer, model, lrs, settings, target, loss, views, exposure,
  });
  const losses = [];
  const timeline = [];
  const started = performance.now();
  // Wall-clock epoch of the first optimiser step, so a report can place this
  // run's trace on an axis whose origin is when the *command* was launched
  // rather than when optimisation began. Interpreter start, module loading,
  // device init and warmup all live to the left of it.
  const startedEpochMs = Date.now();
  const budgetMs = seconds == null ? Infinity : seconds * 1000;
  let callbackMs = 0;
  let step = 0;

  const elapsed = () => performance.now() - started - callbackMs;
  while (step < steps && elapsed() < budgetMs) {
    const result = await session.step(step);
    losses.push(result.loss);
    const elapsedMs = elapsed();
    timeline.push({ step: step + 1, elapsedMs, loss: result.loss });
    if (onStep) {
      const callbackStarted = performance.now();
      await onStep({ step: step + 1, elapsedMs, loss: result.loss });
      callbackMs += performance.now() - callbackStarted;
    }
    step++;
  }

  const optimizeMs = elapsed();
  return {
    steps: step,
    optimizeMs,
    optimizeStartEpochMs: startedEpochMs,
    ...(onStep ? { callbackMs } : {}),
    msPerStep: step ? optimizeMs / step : null,
    losses,
    timeline,
    ...(session.exposureK == null ? {} : { exposureK: session.exposureK }),
    ...(session.exposureW == null ? {} : { exposureW: session.exposureW }),
  };
}

// With a tonemapped renderer, the external loss sees the display image and
// its cotangent is chained back to linear light (plus dL/dk and, for the
// white-point operators, dL/dW) on the host -- the composite backward stays
// linear either way.
async function lossOnView(renderer, loss, image, step, exposure) {
  if (!exposure) {
    const current = await loss(image, step);
    return { loss: current.loss, dLdI: current.dLdI, kGrad: null, wGrad: null };
  }
  const k = Math.exp(exposure.kappa);
  const w = exposure.omega === undefined ? null : Math.exp(exposure.omega);
  if (!exposure.scratch || exposure.scratch.length !== image.length) {
    exposure.scratch = new Float32Array(image.length);
  }
  const current = await loss(tonemapImage(renderer.tonemap, image, k, w, exposure.scratch), step);
  const chain = tonemapChain(renderer.tonemap, image, k, w, current.dLdI);
  return { loss: current.loss, dLdI: current.dLdI, kGrad: chain.kGrad, wGrad: chain.wGrad };
}

async function externalStep(renderer, scene, loss, viewList, step, exposure = null) {
  const singleWeight = viewList?.[0]?.weight ?? 1;
  if (Array.isArray(viewList) && viewList.length === 1 &&
      Number.isFinite(singleWeight) && singleWeight !== 0) {
    const view = viewList[0];
    renderer.uploadScene(scene, {
      ...view,
      s: view.s ?? [view.scale ?? 1, view.scale ?? 1],
    });
    const image = await renderer.forward();
    const current = await lossOnView(renderer, loss, image, step, exposure);
    const grad = await renderer.backward(current.dLdI);
    return {
      ...grad,
      loss: current.loss,
      ...(current.kGrad == null ? {} : { kGrad: current.kGrad }),
      ...(current.wGrad == null ? {} : { wGrad: current.wGrad }),
    };
  }

  let curveGrads = null;
  let shapeGrads = null;
  let totalLoss = 0;
  let totalWeight = 0;
  let totalKGrad = 0;
  let totalWGrad = 0;
  for (const view of viewList) {
    const weight = view.weight ?? 1;
    renderer.uploadScene(scene, {
      ...view,
      s: view.s ?? [view.scale ?? 1, view.scale ?? 1],
    });
    const image = await renderer.forward();
    const current = await lossOnView(renderer, loss, image, step, exposure);
    const grad = await renderer.backward(current.dLdI);
    curveGrads ??= new Float64Array(grad.curveGrads.length);
    shapeGrads ??= new Float64Array(grad.shapeGrads.length);
    addScaled(curveGrads, grad.curveGrads, weight);
    addScaled(shapeGrads, grad.shapeGrads, weight);
    totalLoss += current.loss * weight;
    if (current.kGrad != null) totalKGrad += current.kGrad * weight;
    if (current.wGrad != null) totalWGrad += current.wGrad * weight;
    totalWeight += weight;
  }
  const inv = 1 / totalWeight;
  scale(curveGrads, inv);
  scale(shapeGrads, inv);
  return {
    curveGrads,
    shapeGrads,
    loss: totalLoss * inv,
    ...(exposure ? { kGrad: totalKGrad * inv } : {}),
    ...(exposure && renderer.hasWhite ? { wGrad: totalWGrad * inv } : {}),
  };
}

function addScaled(out, input, scale) {
  for (let i = 0; i < out.length; i++) out[i] += input[i] * scale;
}

function scale(values, amount) {
  for (let i = 0; i < values.length; i++) values[i] *= amount;
}
