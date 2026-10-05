// Freezing a parameter group must not perturb the groups still trained. The
// fixed-point accumulation is exact, so curve gradients and the loss must come
// back bit-identical to a full run. Shape gradients get a tolerance: each
// specialization compiles separately, and once a frozen group's code is dead
// the compiler may fuse or reorder the per-pixel f32 math differently, moving a
// rare pixel across a rounding step (seen on Metal under macOS 13, a few ulps).
// The bound sits far below what a fault in the shared fixed-point path
// (setFixedScale, SLOTS, the slot reduction) would produce.
//
// Needs a real adapter, so it skips where WebGPU is unavailable.

import assert from 'node:assert/strict';
import test, { after } from 'node:test';

import { packScene } from '../../js/prep.js';
import { Renderer, requestDevice } from '../../js/renderer.js';

const WIDTH = 96;
const HEIGHT = 72;
const N = 40;
// Relative to the largest full-run shape gradient.
const SHAPE_GRAD_TOLERANCE = 1e-6;

function scene() {
  let seed = 7;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  const loop = (anchors, controls) => anchors.flatMap((a, i) =>
    [...a, ...controls[i], ...anchors[(i + 1) % anchors.length]]);
  return Array.from({ length: N }, () => {
    const cx = rand() * WIDTH;
    const cy = rand() * HEIGHT;
    const r = 4 + rand() * 18;
    const k = 5;
    const anchors = Array.from({ length: k }, (_, i) => {
      const t = (i / k) * 2 * Math.PI;
      return [cx + r * Math.cos(t), cy + r * Math.sin(t)];
    });
    const controls = anchors.map((a, i) => {
      const b = anchors[(i + 1) % anchors.length];
      return [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
    });
    return {
      curves: Float64Array.from(loop(anchors, controls)),
      color: [rand(), rand(), rand()],
      alpha: 0.4 + 0.5 * rand(),
      fillRule: 'nonzero',
    };
  });
}

const packed = packScene(scene());
const target = new Float32Array(WIDTH * HEIGHT * 4);
for (let i = 0; i < target.length; i++) target[i] = (i % 4 === 3) ? 1 : (i % 97) / 97;

async function step(train) {
  const device = await requestDevice();
  let renderer;
  try {
    renderer = await Renderer.create(device, {
      width: WIDTH, height: HEIGHT, maxPieces: packed.pieceCount,
      maxShapes: N, maxCurves: packed.curveCount, train,
    });
    renderer.uploadScene(packed, { s: [1.2, 1.2], scale: 1, origin: [0, 0], bg: [1, 1, 1] });
    renderer.uploadTarget(target);
    const grads = await renderer.stepGpuLoss();
    return {
      train: renderer.train,
      frozen: renderer.frozen,
      curveGrads: grads.curveGrads ? Array.from(grads.curveGrads) : undefined,
      shapeGrads: grads.shapeGrads ? Array.from(grads.shapeGrads) : undefined,
      blurGrads: grads.blurGrads ? Array.from(grads.blurGrads) : undefined,
      loss: grads.loss,
      bytes: {
        pieceGrads: renderer.pieceGradsBuf?.size ?? 0,
        curveGrads: renderer.curveGradsBuf?.size ?? 0,
        curveMeta: renderer.curveMetaBuf?.size ?? 0,
        shapeGrads: renderer.shapeGradsBuf?.size ?? 0,
      },
    };
  } finally {
    renderer?.destroy();
    device.destroy();
  }
}

// Node + Dawn can hold the event loop open after the last device is destroyed
// (macOS 13/14), so end the process explicitly once every test has reported.
after(() => setTimeout(() => process.exit(), 100));

let available = true;
try {
  (await requestDevice()).destroy();
} catch {
  available = false;
}

test('frozen parameter groups leave the trained ones unchanged', { skip: !available }, async () => {
  const full = await step(undefined);
  assert.deepEqual(full.frozen, []);
  const shapeTolerance = SHAPE_GRAD_TOLERANCE * Math.max(...full.shapeGrads.map(Math.abs));

  for (const geometry of [true, false]) {
    for (const colour of [true, false]) {
      for (const alpha of [true, false]) {
        if (!geometry && !colour && !alpha) continue;
        const label = `geometry=${geometry} colour=${colour} alpha=${alpha}`;
        const got = await step({ geometry, colour, alpha });

        // A frozen group is omitted, never zero-filled: a zero fill would keep
        // the readback this exists to avoid.
        if (geometry) {
          assert.deepEqual(got.curveGrads, full.curveGrads, `curve grads drift: ${label}`);
        } else {
          assert.equal(got.curveGrads, undefined, `curve grads present: ${label}`);
          assert.equal(got.bytes.curveGrads, 0, `curveGrads allocated: ${label}`);
          assert.equal(got.bytes.curveMeta, 0, `curveMeta allocated: ${label}`);
          assert.ok(got.bytes.pieceGrads <= 4, `pieceGrads not reduced: ${label}`);
        }

        if (colour || alpha) {
          // The stride stays 4 floats whatever is frozen, so every model's
          // pullback keeps one layout; a frozen lane is simply zero.
          assert.equal(got.shapeGrads.length, N * 4, `shape stride changed: ${label}`);
          for (let i = 0; i < got.shapeGrads.length; i++) {
            const trained = (i % 4 === 3) ? alpha : colour;
            if (trained) {
              const drift = Math.abs(got.shapeGrads[i] - full.shapeGrads[i]);
              assert.ok(drift <= shapeTolerance, `shape grad ${i} drift ${drift}: ${label}`);
            } else {
              assert.equal(got.shapeGrads[i], 0, `frozen shape grad ${i} nonzero: ${label}`);
            }
          }
        } else {
          assert.equal(got.shapeGrads, undefined, `shape grads present: ${label}`);
          assert.ok(got.bytes.shapeGrads <= 4, `shapeGrads not reduced: ${label}`);
        }

        assert.equal(got.loss, full.loss, `loss drift: ${label}`);
      }
    }
  }
});

test('blur training must not perturb the other groups', { skip: !available }, async () => {
  const full = await step(undefined);
  // Widening the shape-gradient stride to carry (dsx, dsy) interleaves extra
  // atomics, but integer accumulation commutes: every other gradient family
  // and the loss must come back bit-identical to a blur-off run.
  const blurred = await step({ blur: true });
  assert.deepEqual(blurred.frozen, []);
  assert.deepEqual(blurred.curveGrads, full.curveGrads, 'curve grads drift with blur on');
  assert.deepEqual(blurred.shapeGrads, full.shapeGrads, 'shape grads drift with blur on');
  assert.equal(blurred.loss, full.loss, 'loss drift with blur on');
  // The scene sets no per-shape filters, so blur grads are w.r.t. the global
  // filter every shape shares: present, sized 2 per shape, and generally
  // nonzero (the filter genuinely affects the loss).
  assert.equal(blurred.blurGrads.length, N * 2);
  assert.ok(blurred.blurGrads.some((value) => value !== 0), 'blur grads all zero');
});

test('training nothing is rejected', { skip: !available }, async () => {
  await assert.rejects(() => step({ geometry: false, colour: false, alpha: false }),
    /at least one parameter group/);
});
