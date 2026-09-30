import assert from 'node:assert/strict';
import test from 'node:test';

import { rawStyle } from '../../js/raw-style.js';

const COLORS = [[0.2, 0.4, 0.8], [0.9, 0.1, 0.3]];
const ALPHAS = [0.7, 0.5];

// Scalar objective L = sum over shapes of dot(cot_rgb, color) + cot_a * alpha,
// exactly what pullback differentiates through the transfer.
function lossOf(style, params, cot) {
  let total = 0;
  const color = [0, 0, 0];
  for (let i = 0; i < COLORS.length; i++) {
    const alpha = style.decode(params, i, i, color);
    const o = 4 * i;
    total += cot[o] * color[0] + cot[o + 1] * color[1] + cot[o + 2] * color[2];
    total += cot[o + 3] * alpha;
  }
  return total;
}

function grads(style, params) {
  return Object.fromEntries(
    style.groups.map((group) => [group, new Float64Array(params[group].length)]),
  );
}

const VARIANTS = [];
for (const channels of ['gray', 'rgb']) {
  for (const transfer of ['identity', 'softplus', 'sigmoid']) {
    for (const alpha of [1, 0.8, 'learned']) VARIANTS.push({ channels, transfer, alpha });
  }
}

test('raw style pullback matches finite differences for every variant', () => {
  const h = 1e-5;
  const cot = Float32Array.of(0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8);
  for (const variant of VARIANTS) {
    const style = rawStyle(variant);
    const params = style.init(COLORS, ALPHAS);
    const analytic = grads(style, params);
    for (let i = 0; i < COLORS.length; i++) {
      style.pullback(params, i, i, cot, 4 * i, analytic);
    }
    for (const group of style.groups) {
      for (let j = 0; j < params[group].length; j++) {
        const saved = params[group][j];
        params[group][j] = saved + h;
        const hi = params[group][j]; // realized f32 step, not the nominal one
        const plus = lossOf(style, params, cot);
        params[group][j] = saved - h;
        const lo = params[group][j];
        const minus = lossOf(style, params, cot);
        params[group][j] = saved;
        const fd = (plus - minus) / (hi - lo);
        assert.ok(
          Math.abs(analytic[group][j] - fd) < 1e-5,
          `${variant.channels}/${variant.transfer}/${variant.alpha} ` +
            `${group}[${j}] ${analytic[group][j]} vs fd ${fd}`,
        );
      }
    }
  }
});

test('init inverts the transfer so decode reproduces the seeded colors', () => {
  for (const transfer of ['identity', 'softplus', 'sigmoid']) {
    const style = rawStyle({ channels: 'rgb', transfer, alpha: 'learned' });
    const params = style.init(COLORS, ALPHAS);
    const color = [0, 0, 0];
    for (let i = 0; i < COLORS.length; i++) {
      const alpha = style.decode(params, i, i, color);
      for (let c = 0; c < 3; c++) {
        assert.ok(Math.abs(color[c] - COLORS[i][c]) < 1e-5,
          `${transfer} color[${c}] ${color[c]} vs ${COLORS[i][c]}`);
      }
      assert.ok(Math.abs(alpha - ALPHAS[i]) < 1e-3, `${transfer} alpha ${alpha}`);
    }
  }
});

test('gray channels decode neutrally and project through the channel mean', () => {
  const style = rawStyle({ channels: 'gray', transfer: 'identity' });
  const params = style.init(COLORS, ALPHAS);
  assert.equal(params.rawColor.length, COLORS.length);
  const color = [0, 0, 0];
  const alpha = style.decode(params, 0, 0, color);
  const mean = (COLORS[0][0] + COLORS[0][1] + COLORS[0][2]) / 3;
  assert.ok(Math.abs(color[0] - mean) < 1e-7);
  assert.equal(color[0], color[1]);
  assert.equal(color[1], color[2]);
  assert.equal(alpha, 1); // fixed default
});

test('identity transfer is exactly linear: signed values round-trip unclamped', () => {
  const style = rawStyle({ channels: 'rgb', transfer: 'identity' });
  const params = style.init([[-1.4, 0, 2.7]], []);
  const color = [0, 0, 0];
  style.decode(params, 0, 0, color);
  assert.deepEqual(color, [params.rawColor[0], params.rawColor[1], params.rawColor[2]]);
  assert.ok(color[0] < 0 && color[2] > 1);
});

test('fixed alpha creates no alpha group', () => {
  const fixed = rawStyle({ channels: 'gray', transfer: 'softplus', alpha: 0.6 });
  assert.deepEqual(fixed.groups, ['rawColor']);
  assert.equal(fixed.trainsAlpha, false);
  assert.equal(fixed.fixedAlpha, 0.6);
  const params = fixed.init(COLORS, ALPHAS);
  assert.equal(params.rawAlpha, undefined);
  assert.equal(fixed.decode(params, 0, 0, [0, 0, 0]), 0.6);

  const learned = rawStyle({ channels: 'gray', transfer: 'softplus', alpha: 'learned' });
  assert.deepEqual(learned.groups, ['rawColor', 'rawAlpha']);
  assert.ok(learned.trainsAlpha);
  assert.equal(learned.init(COLORS, ALPHAS).rawAlpha.length, COLORS.length);
});

test('range follows the transfer codomain and bad options are rejected', () => {
  assert.equal(rawStyle({ transfer: 'identity' }).range, 'signed');
  assert.equal(rawStyle({ transfer: 'softplus' }).range, 'nonneg');
  assert.equal(rawStyle({ transfer: 'sigmoid' }).range, 'unit');
  assert.throws(() => rawStyle({ transfer: 'tanh' }), /transfer must be one of/);
  assert.throws(() => rawStyle({ channels: 'cmyk' }), /channels must be/);
  assert.throws(() => rawStyle({ alpha: 0 }), /alpha must be/);
  assert.throws(() => rawStyle({ alpha: 1.5 }), /alpha must be/);
  assert.throws(() => rawStyle({ alpha: 'auto' }), /alpha must be/);
});
