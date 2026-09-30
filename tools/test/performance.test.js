import assert from 'node:assert/strict';
import test, { after } from 'node:test';
import { Renderer, requestDevice } from '../../js/renderer.js';
import { LoopModel } from '../../js/loop-model.js';
import { rawStyle } from '../../js/raw-style.js';
import { renderChunked } from '../../demos/render/tiled.js';
import { blitRgba8 } from '../../demos/util/image.js';

const width = 35, height = 29, n = 19, k = 5;
const target = Float32Array.from({ length: width * height * 4 }, (_, i) => (i % 4 === 3) ? 1 : (i % 97) / 97);
const settings = (step) => ({ s: [1.1 + step * 0.03, 1.4 + step * 0.02],
  origin: [-0.2, 0.3], scale: 1, bg: [0.83, 0.91, 0.97] });
let device;
try { device = await requestDevice(); } catch { /* Optional real-GPU checks. */ }
after(() => { device?.destroy(); setTimeout(() => process.exit(), 100); });

function modelFor() {
  const ax = new Float32Array(n * k), ay = new Float32Array(n * k);
  const cx = new Float32Array(n * k), cy = new Float32Array(n * k);
  for (let shape = 0; shape < n; shape++) {
    const x = (shape * 7.37) % width, y = (shape * 11.31) % height;
    const r = 2.7 + shape % 7;
    for (let j = 0; j < k; j++) {
      const a = j * 2 * Math.PI / k + 0.137;
      const at = shape * k + j;
      ax[at] = x + Math.cos(a) * r; ay[at] = y + Math.sin(a) * r;
      cx[at] = x + Math.cos(a + Math.PI / k) * r * 1.7;
      cy[at] = y + Math.sin(a + Math.PI / k) * r * 1.3;
    }
  }
  const style = rawStyle({ channels: 'rgb', transfer: 'sigmoid', alpha: 0.8 });
  const colors = Array.from({ length: n }, (_, i) => [0.17 + (i % 7) * 0.07, 0.37, 0.58]);
  const model = new LoopModel({ ax, ay, cx, cy, n, k, style,
    styleParams: style.init(colors, new Float32Array(n).fill(0.8)) });
  model.shapes.forEach((shape, i) => { shape.fillRule = i % 3 ? 'nonzero' : 'evenodd'; });
  return model;
}
async function rendererFor(model, opts = {}) {
  return Renderer.create(device, { width, height, maxShapes: n, maxCurves: n * k,
    maxPieces: n * k * 3, train: { alpha: model.style.trainsAlpha }, colorRange: model.style.range, ...opts });
}
test('dense gather preserves tile order, images, and all gradients exactly', { skip: !device }, async () => {
  const model = modelFor();
  const packed = model.decode().scene;
  const original = process.env.WF_SORT_CAPACITY;
  const run = async (capacity) => {
    process.env.WF_SORT_CAPACITY = capacity;
    const r = await rendererFor(model);
    try {
      r.uploadScene(packed, settings(0)); r.uploadTarget(target);
      const grads = await r.stepGpuLoss();
      const image = await r.readImage();
      return { curve: Array.from(grads.curveGrads), shape: Array.from(grads.shapeGrads),
        loss: grads.loss, image: Array.from(image) };
    } finally { r.destroy(); }
  };
  device.pushErrorScope('validation');
  try { assert.deepEqual(await run('2'), await run('2048')); }
  finally {
    if (original === undefined) delete process.env.WF_SORT_CAPACITY;
    else process.env.WF_SORT_CAPACITY = original;
    assert.equal(await device.popErrorScope(), null);
  }
});

test('cached code survives resolution/capacity changes and specializes training', { skip: !device }, async () => {
  const model = modelFor();
  const a = await rendererFor(model);
  const b = await Renderer.create(device, { width: 8, height: 11,
    maxShapes: 1, maxCurves: 1, maxPieces: 3, train: { alpha: false } });
  const c = await rendererFor(model, { train: { geometry: false } });
  const f = await rendererFor(model, { forwardOnly: true });
  try {
    assert.equal(a.fwdPipe, b.fwdPipe);
    assert.equal(a.bwdPipe, b.bwdPipe);
    assert.notEqual(a.bwdPipe, c.bwdPipe);
    assert.equal(f.bwdPipe, null); assert.equal(f.dLdImageBuf, null);
    assert.equal(f.pieceGradsBuf, null); assert.equal(f.shapeGradsBuf, null);
    await assert.rejects(() => f.stepGpuLoss(), /cannot train/);
    f.uploadScene(model.decode().scene, settings(0));
    a.uploadScene(model.decode().scene, settings(0));
    assert.deepEqual(await f.forward(), await a.forward());
  } finally { a.destroy(); b.destroy(); c.destroy(); f.destroy(); }
});

test('chunked forward rendering reuses geometry without changing pixels', { skip: !device }, async () => {
  const model = modelFor(), packed = model.decode().scene;
  const r = await rendererFor(model, { forwardOnly: true });
  device.pushErrorScope('validation');
  try {
    r.uploadScene(packed, settings(0));
    const expected = new Uint8Array(width * height * 4);
    blitRgba8(expected, width, 0, 0, await r.forward(), width, width, height);
    const chunked = await renderChunked(device, packed, { width, height,
      scale: settings(0).scale, origin: settings(0).origin }, { s: settings(0).s,
      bg: settings(0).bg, chunk: 16 });
    assert.deepEqual(chunked.rgba, expected);
  } finally { r.destroy(); assert.equal(await device.popErrorScope(), null); }
});
