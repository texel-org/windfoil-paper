import assert from 'node:assert/strict';
import test from 'node:test';

import {
  TONEMAP_DOMAINS,
  TONEMAP_HAS_WHITE,
  TONEMAP_MODES,
  tonemapChain,
  tonemapImage,
} from '../../js/tonemap.js';

const K = 1.7;
const W = 2.3;

// Small deterministic linear image (RGBA); values span the HDR range and,
// for the signed cases, cross zero.
function linearImage(signed) {
  const values = [];
  for (let i = 0; i < 6; i++) {
    const base = [0.03 + 0.61 * i, 1.9 - 0.83 * i, 0.24 * i * i];
    for (const v of base) values.push(signed ? v - 0.9 : Math.abs(v));
    values.push(0.5 + 0.1 * i); // alpha lane, must pass through untouched
  }
  return Float32Array.from(values);
}

function cotangent(length) {
  const cot = new Float32Array(length);
  for (let i = 0; i < length; i++) cot[i] = 0.017 * Math.sin(1.3 * i + 0.4) + 0.005;
  return cot;
}

// Scalar loss L = sum over rgb lanes of cot * T(x), the exact objective the
// chain functions differentiate.
function lossOf(mode, linear, cot, k, w) {
  const display = tonemapImage(mode, linear, k, w);
  let total = 0;
  for (let i = 0; i < linear.length; i += 4) {
    for (let c = 0; c < 3; c++) total += cot[i + c] * display[i + c];
  }
  return total;
}

const MODES = [
  { mode: 'reinhard', w: null, signed: false },
  { mode: 'reinhard-white', w: W, signed: false },
  { mode: 'smooth', w: W, signed: false },
  { mode: 'smooth', w: W, signed: true },
];

test('tonemap chains match finite differences of the operators', () => {
  const h = 1e-4;
  for (const { mode, w, signed } of MODES) {
    const linear = linearImage(signed);
    const cot = cotangent(linear.length);
    const dLdDisplay = Float32Array.from(cot);
    const chain = tonemapChain(mode, linear, K, w, dLdDisplay);

    // dL/dx per rgb lane (alpha lanes must be zeroed).
    for (let i = 0; i < linear.length; i++) {
      if (i % 4 === 3) {
        assert.equal(dLdDisplay[i], 0);
        continue;
      }
      const plus = Float32Array.from(linear);
      const minus = Float32Array.from(linear);
      plus[i] += h;
      minus[i] -= h;
      const fd = (lossOf(mode, plus, cot, K, w) - lossOf(mode, minus, cot, K, w)) / (2 * h);
      assert.ok(Math.abs(dLdDisplay[i] - fd) < 1e-4,
        `${mode} dL/dx[${i}] ${dLdDisplay[i]} vs fd ${fd}`);
    }

    // dL/dk and dL/dW.
    const fdK = (lossOf(mode, linear, cot, K + h, w) - lossOf(mode, linear, cot, K - h, w)) / (2 * h);
    assert.ok(Math.abs(chain.kGrad - fdK) < 1e-4, `${mode} kGrad ${chain.kGrad} vs fd ${fdK}`);
    if (w === null) {
      assert.equal(chain.wGrad, null);
    } else {
      const fdW = (lossOf(mode, linear, cot, K, w + h) - lossOf(mode, linear, cot, K, w - h)) / (2 * h);
      assert.ok(Math.abs(chain.wGrad - fdW) < 1e-4, `${mode} wGrad ${chain.wGrad} vs fd ${fdW}`);
    }
  }
});

test('white-point operators reach display 1 at W and tend to x/W as k -> 0', () => {
  for (const mode of ['reinhard-white', 'smooth']) {
    const atWhite = tonemapImage(mode, Float32Array.of(W, W, W, 1), K, W);
    for (let c = 0; c < 3; c++) assert.ok(Math.abs(atWhite[c] - 1) < 1e-6, `${mode} T(W)`);
    const nearLinear = tonemapImage(mode, Float32Array.of(0.4, 0.4, 0.4, 1), 1e-5, 1.6);
    for (let c = 0; c < 3; c++) {
      assert.ok(Math.abs(nearLinear[c] - 0.4 / 1.6) < 1e-4, `${mode} k->0 limit`);
    }
  }
});

test('smooth is odd and monotone across zero; reinhard family rejects mode misuse', () => {
  const xs = Float32Array.of(-10, -1, -0.25, 0, 0.25, 1, 10, 0);
  const display = tonemapImage('smooth', xs, K, W);
  const mirrored = tonemapImage('smooth', Float32Array.from(xs, (v) => -v), K, W);
  for (let c = 0; c < 3; c++) {
    assert.ok(Math.abs(display[c] + mirrored[c]) < 1e-7, 'odd symmetry');
  }
  for (let c = 0; c < 2; c++) assert.ok(display[c] < display[c + 1], 'monotone');
  assert.throws(() => tonemapImage('none', xs, K, W), /no tonemap operator/);
  assert.throws(() => tonemapImage('aces', xs, K, W), /no tonemap operator/);
  assert.throws(() => tonemapImage('reinhard', xs, 0, null), /exposure must be positive/);
  assert.throws(() => tonemapImage('smooth', xs, K, 0), /white point must be positive/);
  assert.throws(() => tonemapChain('smooth', xs, K, W, new Float32Array(4)), /size mismatch/);
});

test('mode tables agree on codes, domains, and white points', () => {
  assert.deepEqual(TONEMAP_MODES, {
    'none': 0, 'reinhard': 1, 'reinhard-white': 2, 'smooth': 3,
  });
  assert.equal(TONEMAP_DOMAINS.reinhard, 'nonneg');
  assert.equal(TONEMAP_DOMAINS['reinhard-white'], 'nonneg');
  assert.equal(TONEMAP_DOMAINS.smooth, 'signed');
  assert.equal(TONEMAP_HAS_WHITE.reinhard, undefined);
  assert.ok(TONEMAP_HAS_WHITE['reinhard-white']);
  assert.ok(TONEMAP_HAS_WHITE.smooth);
});
