import assert from 'node:assert/strict';
import test from 'node:test';

import { anchorStyle } from '../../js/color-anchors.js';

test('anchor style reuses exact decode weights in pullback', () => {
  const style = anchorStyle();
  const params = style.init(
    [[0.2, 0.4, 0.8], [0.9, 0.1, 0.3]],
    [0.7, 0.5],
  );
  const scratch = style.createScratch(params);
  const cot = Float32Array.of(0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8);
  const cached = new Float64Array(params.colorAnchor.length);
  const recomputed = new Float64Array(params.colorAnchor.length);

  for (let i = 0; i < 2; i++) {
    const cachedColor = [0, 0, 0];
    const recomputedColor = [0, 0, 0];
    const cachedAlpha = style.decode(params, i, i, cachedColor, scratch);
    const recomputedAlpha = style.decode(params, i, i, recomputedColor);
    assert.deepEqual(cachedColor, recomputedColor);
    assert.equal(cachedAlpha, recomputedAlpha);
    style.pullback(params, i, i, cot, 4 * i, { colorAnchor: cached }, scratch);
    style.pullback(params, i, i, cot, 4 * i, { colorAnchor: recomputed });
  }

  assert.deepEqual(cached, recomputed);
  assert.notEqual(style.createScratch(params), scratch);
});

test('anchor scratch falls back safely before decode and after invalidation', () => {
  const style = anchorStyle();
  const params = style.init([[0.2, 0.4, 0.8]], [0.7]);
  const scratch = style.createScratch(params);
  const cot = Float32Array.of(0.1, -0.2, 0.3, 0.4);
  const cached = new Float64Array(params.colorAnchor.length);
  const recomputed = new Float64Array(params.colorAnchor.length);

  style.pullback(params, 0, 0, cot, 0, { colorAnchor: cached }, scratch);
  style.pullback(params, 0, 0, cot, 0, { colorAnchor: recomputed });
  assert.deepEqual(cached, recomputed);

  cached.fill(0);
  recomputed.fill(0);
  style.decode(params, 0, 0, [0, 0, 0], scratch);
  style.invalidateScratch(scratch);
  params.colorAnchor[0] += 0.25;
  style.pullback(params, 0, 0, cot, 0, { colorAnchor: cached }, scratch);
  style.pullback(params, 0, 0, cot, 0, { colorAnchor: recomputed });
  assert.deepEqual(cached, recomputed);
});
