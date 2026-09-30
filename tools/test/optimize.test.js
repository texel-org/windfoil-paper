import assert from 'node:assert/strict';
import test from 'node:test';

import { optimize } from '../../js/optimize.js';

function fixture() {
  const renderer = {
    uploadTarget() {},
    uploadScene() {},
    async stepGpuLoss() {
      return { loss: 0.25, curveGrads: new Float32Array(), shapeGrads: new Float32Array() };
    },
  };
  const model = {
    params: { value: Float64Array.of(0) },
    decode: () => ({ scene: {}, shapes: [] }),
    pullback: () => ({ value: Float64Array.of(0) }),
  };
  return {
    renderer,
    model,
    lrs: { value: { lr: 1 } },
    settings: () => ({}),
    target: Float32Array.of(0),
  };
}

test('optimizer excludes step callback time', async () => {
  const started = performance.now();
  const result = await optimize({
    ...fixture(),
    steps: 2,
    onStep: () => new Promise((resolve) => setTimeout(resolve, 20)),
  });
  const wallMs = performance.now() - started;
  assert.equal(result.steps, 2);
  assert.ok(result.callbackMs >= 30, `callbackMs=${result.callbackMs}`);
  assert.ok(wallMs - result.optimizeMs >= 30, `wall=${wallMs}, optimize=${result.optimizeMs}`);
  assert.ok(result.timeline.at(-1).elapsedMs <= result.optimizeMs);
});

test('optimizer keeps the callback-free timing path', async () => {
  const result = await optimize({ ...fixture(), steps: 2 });
  assert.equal(result.steps, 2);
  assert.equal('callbackMs' in result, false);
  assert.equal(result.msPerStep, result.optimizeMs / 2);
});

test('optimizer re-uploads scheduled targets only when they change', async () => {
  const shared = fixture();
  const uploads = [];
  shared.renderer.uploadTarget = (target) => uploads.push(target);
  const early = Float32Array.of(0);
  const late = Float32Array.of(1);
  const result = await optimize({
    ...shared,
    target: (step) => (step < 2 ? early : late),
    steps: 4,
  });
  assert.equal(result.steps, 4);
  assert.deepEqual(uploads, [early, late]);
});

test('single-view external loss returns renderer gradients without copies', async () => {
  const curveGrads = Float32Array.of(1, 2);
  const shapeGrads = Float32Array.of(3, 4);
  let pulledBack;
  const renderer = {
    uploadScene() {},
    async forward() { return Float32Array.of(0); },
    async backward() { return { curveGrads, shapeGrads }; },
  };
  const model = {
    params: {},
    decode: () => ({ scene: {}, shapes: [] }),
    pullback: (result) => { pulledBack = result; return {}; },
  };

  await optimize({
    renderer,
    model,
    lrs: {},
    settings: () => ({}),
    loss: async () => ({ loss: 0.5, dLdI: Float32Array.of(1) }),
    views: () => [{ scale: 1, origin: [0, 0], weight: 0.25 }],
    steps: 1,
  });

  assert.strictEqual(pulledBack.curveGrads, curveGrads);
  assert.strictEqual(pulledBack.shapeGrads, shapeGrads);
  assert.equal(pulledBack.loss, 0.5);
});
