import assert from 'node:assert/strict';
import test from 'node:test';

import {
  parseOffset,
  parseRoundtripArgs,
  resolveRoundtripOffset,
  roundtripAnnealSettings,
  roundtripRenderSettings,
  summarizeLosses,
} from '../../demos/roundtrip/cli.js';

test('roundtrip CLI has strict, portable defaults and parses its complete surface', () => {
  assert.deepEqual(parseRoundtripArgs([]), {
    svg: 'demos/roundtrip/star-evenodd.svg',
    optSize: 512,
    steps: 100,
    blur: 7,
    offset: null,
    saveEvery: null,
    out: null,
    quiet: false,
  });
  assert.deepEqual(parseRoundtripArgs([
    '--svg=shape.svg', '--opt-size', '256', '--steps=40', '--blur', '3.5',
    '--offset=-2, 4.25', '--save-every=5', '--out', 'output/case', '--quiet',
  ]), {
    svg: 'shape.svg',
    optSize: 256,
    steps: 40,
    blur: 3.5,
    offset: [-2, 4.25],
    saveEvery: 5,
    out: 'output/case',
    quiet: true,
  });

  assert.throws(() => parseRoundtripArgs(['shape.svg']), /unexpected argument/);
  assert.throws(() => parseRoundtripArgs(['--target=x']), /unknown option/);
  assert.throws(() => parseRoundtripArgs(['--steps=1', '--steps=2']), /duplicate option/);
  assert.throws(() => parseRoundtripArgs(['--opt-size=max']), /positive integer/);
  assert.throws(() => parseRoundtripArgs(['--quiet=true']), /does not take a value/);
  assert.throws(() => parseOffset('1'), /two comma-separated/);
});

test('roundtrip offset and filter widths scale with arbitrary SVG viewBoxes', () => {
  assert.deepEqual(resolveRoundtripOffset(null, { width: 1, height: 2 }), [1 / 16, 2 / 16]);
  assert.deepEqual(resolveRoundtripOffset([3, -4], { width: 1, height: 2 }), [3, -4]);
  const raster = { scale: 0.25, origin: [-2, 7] };
  const background = [1, 0.5, 0];
  assert.deepEqual(roundtripRenderSettings(raster, background, 3), {
    s: [0.75, 0.75],
    scale: 0.25,
    origin: [-2, 7],
    bg: background,
  });
  assert.deepEqual(roundtripAnnealSettings(0, 100, raster, background, 7).s, [1.75, 1.75]);
  assert.deepEqual(roundtripAnnealSettings(100, 100, raster, background, 7).s, [0.25, 0.25]);
});

test('roundtrip loss summaries report the best one-based step', () => {
  assert.deepEqual(summarizeLosses([]), {
    start: null, end: null, min: null, bestStep: null,
  });
  assert.deepEqual(summarizeLosses([0.4, 0.2, 0.3]), {
    start: 0.4, end: 0.3, min: 0.2, bestStep: 2,
  });
});
