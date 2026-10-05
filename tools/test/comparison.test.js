import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  cropRGBA,
  cropWindow,
  diffRGBA,
  grayRGBA,
  upscaleCrop,
} from '../../demos/comparison/images.js';
import { comparisonOptions, DEFAULT_SCENES } from '../../demos/comparison/options.js';
import { resolveScene, sceneKind, translateQuads } from '../../demos/comparison/scenes.js';
import { bandQuads, buildSlugScene } from '../../demos/comparison/slug.js';
import { coverageStats, pointCoverage, windingAt } from '../../demos/comparison/truth.js';
import { parseArgs } from '../../demos/util/runtime.js';

const line = (x0, y0, x1, y1) => [x0, y0, (x0 + x1) / 2, (y0 + y1) / 2, x1, y1];
const polygon = (points) => points.flatMap((p, i) => line(...p, ...points[(i + 1) % points.length]));
const rect = (x0, y0, x1, y1) => polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);

test('defaults reproduce the paper figure', () => {
  const options = comparisonOptions(parseArgs([]));
  assert.deepEqual(options.scenes, [...DEFAULT_SCENES]);
  assert.deepEqual(options.scenes, ['glyph:@', 'svg:demos/comparison/rosette.svg']);
  assert.equal(options.size, 512);
  assert.equal(options.samples, 64);
  assert.equal(options.amp, 15);
  assert.equal(options.offset, 0);
  assert.equal(options.fit, 'viewbox');
  assert.deepEqual(options.crop, { zoom: 2, inset: 40, corner: 'tr' });
  assert.equal(options.out, null);
});

test('options parse, repeat scenes, and reject bad values', () => {
  const options = comparisonOptions(parseArgs([
    '--scene', 'glyph:,', '--scene=svg:a.svg', '--size', '256', '--samples=8', '--amp', '1',
    '--offset', '0.5', '--fit', 'INK', '--zoom', '4', '--inset', '0', '--corner', 'BL', '--out', 'x',
  ]));
  assert.deepEqual(options.scenes, ['glyph:,', 'svg:a.svg']);
  assert.equal(options.size, 256);
  assert.equal(options.samples, 8);
  assert.equal(options.amp, 1);
  assert.equal(options.offset, 0.5);
  assert.equal(options.fit, 'ink');
  assert.deepEqual(options.crop, { zoom: 4, inset: 0, corner: 'bl' });
  assert.equal(options.out, 'x');

  const bad = (argv, pattern) => assert.throws(() => comparisonOptions(parseArgs(argv)), pattern);
  bad(['--size', '0'], /--size must be a positive integer/);
  bad(['--samples', '1.5'], /--samples must be a positive integer/);
  bad(['--amp', '-2'], /--amp must be positive/);
  bad(['--corner', 'mid'], /--corner must be one of/);
  bad(['--fit', 'cover'], /--fit must be one of/);
  bad(['--scene'], /--scene requires a value/);
  bad(['--out'], /--out requires a directory/);
  bad(['--sise', '512'], /unknown option --sise/);
});

test('scene specs resolve by prefix or shorthand', () => {
  assert.deepEqual(sceneKind('glyph:@'), { kind: 'glyph', rest: '@' });
  assert.deepEqual(sceneKind('glyph::'), { kind: 'glyph', rest: ':' });
  assert.deepEqual(sceneKind('svg:art/a.svg'), { kind: 'svg', rest: 'art/a.svg' });
  assert.deepEqual(sceneKind('art/a:b.SVG'), { kind: 'svg', rest: 'art/a:b.SVG' });
  assert.deepEqual(sceneKind('Q'), { kind: 'glyph', rest: 'Q' });
  assert.throws(() => sceneKind('shape:circle'), /unknown scene/);
});

test('the rosette fixture resolves to its paper geometry', async () => {
  const rosette = fileURLToPath(new URL('../../demos/comparison/rosette.svg', import.meta.url));
  const scene = await resolveScene(`svg:${rosette}`, { font: null, size: 512 });
  assert.equal(scene.slug, 'svg_rosette');
  assert.equal(scene.key, 'rosette');
  assert.equal(scene.quads.length / 6, 1716);
  assert.equal(scene.evenodd, false);
  assert.deepEqual(scene.warnings, []);
  const shifted = translateQuads(scene.quads, 0.5);
  assert.equal(shifted[0], scene.quads[0] + 0.5);
  assert.equal(shifted[1], scene.quads[1] + 0.5);
});

test('ray-cast winding counts nested and reversed contours', () => {
  const outer = rect(0, 0, 10, 10);
  const inner = rect(2, 2, 8, 8);
  assert.deepEqual(windingAt(5, 5, outer), { W: 1, K: 1 });
  assert.deepEqual(windingAt(11, 5, outer), { W: 0, K: 0 });
  assert.equal(windingAt(5, 5, [...outer, ...inner]).W, 2);
  // Reversing the traversal flips the sign.
  assert.equal(windingAt(5, 5, polygon([[0, 0], [0, 10], [10, 10], [10, 0]])).W, -1);
});

test('point-sampled truth is exact on pixel-aligned and half-pixel edges', () => {
  // [1, 3] x [1, 2.5] in a 4 px frame: whole pixels, then a half-covered row.
  const cov = pointCoverage(rect(1, 1, 3, 2.5), { size: 4, samples: 8 });
  assert.deepEqual([...cov], [
    0, 0, 0, 0,
    0, 1, 1, 0,
    0, 0.5, 0.5, 0,
    0, 0, 0, 0,
  ]);
  // One sample per pixel is the aliased pixel-centre fill.
  const binary = pointCoverage(rect(0.6, 0.6, 2.4, 2.4), { size: 3, samples: 1 });
  assert.deepEqual([...binary], [0, 0, 0, 0, 1, 0, 0, 0, 0]);
});

test('truth follows the fill rule through overlapping contours', () => {
  const twice = [...rect(0, 0, 2, 2), ...rect(0, 0, 2, 2)];
  assert.deepEqual([...pointCoverage(twice, { size: 2, samples: 4 })], [1, 1, 1, 1]);
  assert.deepEqual([...pointCoverage(twice, { size: 2, samples: 4, evenodd: true })], [0, 0, 0, 0]);
});

test('truth integrates a diagonal edge to its area', () => {
  // Triangle under the diagonal of one pixel: half its area, up to grid noise.
  const cov = pointCoverage(polygon([[0, 0], [1, 1], [0, 1]]), { size: 1, samples: 64 });
  assert.ok(Math.abs(cov[0] - 0.5) <= 1 / 64, `coverage ${cov[0]}`);
});

test('coverage stats are the unamplified mean and worst pixel', () => {
  assert.deepEqual(coverageStats(Float64Array.of(0, 0.5, 1, 0.25), Float64Array.of(0, 0.25, 1, 1)),
    { mean: 0.25, max: 0.75 });
});

test('error maps amplify, clip, and quantize before tinting', () => {
  const gray = grayRGBA(Float64Array.of(0, 0.5, 1.2));
  assert.deepEqual([...gray], [0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255]);
  const diff = diffRGBA(Float64Array.of(0.5, 0.5, 0.5, 0.2), Float64Array.of(0.5, 0.52, 0.45, 0.8), 15);
  // |d| * 15 = 0, 0.3, 0.75, 9 (clipped to 1)
  assert.deepEqual([...diff], [0, 0, 0, 255, 77, 77, 77, 255, 191, 191, 191, 255, 255, 255, 255, 255]);
  const tinted = diffRGBA(Float64Array.of(0), Float64Array.of(1), 1, [1, 0.5, 0]);
  assert.deepEqual([...tinted], [255, 128, 0, 255]);
});

test('the crop window is the paper detail: a source-sized window of the 2x upscale', () => {
  // 512 px at zoom 2: a 512 px window 40 upscaled px inside the top-right corner,
  // which shows source [236, 492) x [20, 276).
  assert.deepEqual(cropWindow(512, 512), {
    cx: 472, cy: 40, cw: 512, ch: 512, sx: 236, sy: 20, sw: 256, sh: 256,
  });
  assert.deepEqual(cropWindow(512, 512, { corner: 'bl' }), {
    cx: 40, cy: 472, cw: 512, ch: 512, sx: 20, sy: 236, sw: 256, sh: 256,
  });
  assert.throws(() => cropWindow(32, 32, { zoom: 2, inset: 40 }), /does not fit/);
  assert.throws(() => cropWindow(512, 512, { zoom: 1, inset: 1 }), /does not fit/);
});

test('crops copy whole source pixels, nearest neighbour', () => {
  // 4x4 source whose red channel is the pixel index.
  const w = 4, h = 4;
  const src = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) src.set([i, 0, 0, 255], i * 4);
  const red = (rgba) => [...rgba].filter((_, i) => i % 4 === 0);
  // zoom 2, inset 2: upscaled window (2, 2)..(6, 6) of 8x8 is source (1, 1)..(3, 3).
  assert.deepEqual(red(cropRGBA(src, w, h, { zoom: 2, inset: 2, corner: 'tl' })), [
    5, 5, 6, 6,
    5, 5, 6, 6,
    9, 9, 10, 10,
    9, 9, 10, 10,
  ]);
  // An odd upscaled offset still lands on whole source pixels.
  assert.deepEqual(red(upscaleCrop(src, w, h, 2, 1, 0, 4, 1)), [0, 1, 1, 2]);
});

test('Slug bands file every quad its y-hull touches and drop horizontal ones', () => {
  const quads = [];
  for (let i = 0; i < 12; i++) quads.push(...line(i, i, i + 1, i + 1)); // 12 diagonal steps over y in [0, 12]
  quads.push(...line(0, 6, 12, 6)); // horizontal: no crossings
  const curves = [], rows = [];
  const header = bandQuads(quads, 0, 12, curves, rows);
  assert.deepEqual(header, { rowBase: 0, bandCount: 3, invH: 0.25 });
  // 13 quads -> 3 bands of [start, count]. The steps ending on the band edges
  // y = 4 and y = 8 are filed into both bands; the horizontal quad into none.
  assert.deepEqual(rows, [0, 4, 4, 5, 9, 5]);
  assert.equal(curves.length / 6, 14);

  const scene = buildSlugScene(rect(1, 2, 5, 9));
  assert.equal(scene.instance.length, 20);
  assert.deepEqual([...scene.instance.slice(4, 8)], [1, 2, 5, 9]);
  // Horizontal bands cover y, vertical bands the rotated -x range.
  assert.equal(scene.instance[14], 2);
  assert.equal(scene.instance[18], -5);
  assert.equal(scene.curves.length / 6, 4);
});
