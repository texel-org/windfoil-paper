import assert from 'node:assert/strict';
import test from 'node:test';

import { l2Quality, targetFromSource, toGrayscale } from '../../demos/util/image.js';
import { LOSSES } from '../../demos/util/losses.js';
import { plotMarkersToSVG } from '../../demos/util/svg.js';
import { optSizeLabel, parseOptSizes, resolveOptSize } from '../../demos/util/opt-size.js';
import {
  assignCaseIds,
  caseLabel,
  parseSaveOptions,
  percentile,
  resolveSaveSize,
  saveFrameSettings,
  withBlurDefaults,
} from '../../demos/util/run.js';
import { formatLoss } from '../../demos/util/runtime.js';
import {
  WINDFOIL_ENVIRONMENTS,
  windfoilBackend,
  windfoilCommand,
} from '../windfoil-runtime.js';

test('shared L2 quality scores RGB', () => {
  const quality = l2Quality(Float32Array.of(0, 0, 0, 1), Float32Array.of(1, 1, 1, 1));
  assert.equal(quality.mseRgb, 1);
  assert.equal(quality.psnrDb, 0);
});

test('loss formatter stays readable as loss shrinks', () => {
  assert.equal(formatLoss(0.107382), '0.107382');
  assert.equal(formatLoss(0.1), '0.1');
  assert.equal(formatLoss(0.0012345678), '0.00123457');
  assert.equal(formatLoss(0.000001234567), '0.00000123457');
  assert.equal(formatLoss(1e-12), '0.000000000001');
  assert.equal(formatLoss(1e-20), '<0.000000000000001');
  assert.equal(formatLoss(0), '0');
});

test('plot targets collapse to their luma in place', () => {
  const target = {
    rgba: Float32Array.of(1, 0, 0, 1, 0, 1, 0, 1),
    u8: Uint8Array.of(255, 0, 0, 255, 0, 255, 0, 255),
    mean: [0.5, 0.5, 0],
  };
  toGrayscale(target);
  assert.ok(Math.abs(target.rgba[0] - 0.2126) < 1e-6);
  assert.equal(target.rgba[0], target.rgba[1]);
  assert.equal(target.rgba[0], target.rgba[2]);
  assert.ok(Math.abs(target.rgba[4] - 0.7152) < 1e-6);
  assert.equal(target.u8[0], Math.round(0.2126 * 255));
  assert.equal(target.u8[5], Math.round(0.7152 * 255));
  assert.equal(target.u8[3], 255);
  const mean = (0.2126 + 0.7152) / 2;
  assert.ok(target.mean.every((channel) => Math.abs(channel - mean) < 1e-6));
});

test('plot SVG emits line markers in creation order with per-mark color', () => {
  const svg = plotMarkersToSVG(
    [
      { mode: 'line', x0: 1, y0: 2, x1: 5, y1: 6.125, color: [0, 0, 0] },
      { mode: 'line', x0: 3, y0: 3, x1: 4, y1: 4, color: [1, 0, 1], opacity: 0.5 },
    ],
    { widthCm: 18, heightCm: 12, penWidthCm: 0.05 },
  );
  assert.match(svg, /viewBox="0 0 18 12" width="18cm" height="12cm"/);
  assert.match(svg, /<rect width="18" height="12" fill="#ffffff"\/>/);
  assert.match(svg, /<g fill="none" stroke-width="0.05" stroke-linecap="round">/);
  assert.match(svg, /<line x1="1" y1="2" x2="5" y2="6.125" stroke="#000000"\/>/);
  // Translucent marks carry their opacity; opaque ones stay bare.
  assert.match(svg, /<line x1="3" y1="3" x2="4" y2="4" stroke="#ff00ff" opacity="0.5"\/>/);
  // One group, in creation order — occlusion matches the render, not grouped by color.
  assert.equal((svg.match(/<g /g) || []).length, 1);
  assert.ok(svg.indexOf('#000000') < svg.indexOf('#ff00ff'));
});

test('plot SVG emits point markers as filled discs of the pen radius', () => {
  const svg = plotMarkersToSVG(
    [{ mode: 'point', x: 3, y: 4, color: [1, 0, 1] }],
    { widthCm: 18, heightCm: 12, penWidthCm: 0.05 },
  );
  assert.match(svg, /<circle cx="3" cy="4" r="0.025" fill="#ff00ff"\/>/);
  assert.doesNotMatch(svg, /stroke=/);
});

test('loss descriptors expose the subject and output contract', () => {
  assert.equal(LOSSES.l2.subjectKey, 'target');
  assert.equal(LOSSES.l2.loadsTarget, true);
  assert.equal(LOSSES.l2.splitSubjects, true);
  assert.deepEqual(LOSSES.l2.outputFields({ subject: 'photo.png' }), {
    target: 'photo.png',
  });

  assert.equal(LOSSES.clip.subjectKey, 'prompt');
  assert.equal(LOSSES.clip.loadsTarget, false);
  assert.equal(LOSSES.clip.splitSubjects, false);
  const openai = LOSSES.clip.outputFields({
    subject: 'a cat',
    augs: 4,
    lossUrl: 'ws://x',
    clipWeights: 'openai',
    clipag: false,
  });
  assert.equal(openai.prompt, 'a cat');
  assert.equal(openai.clipModel, 'ViT-B-32-quickgelu');
  // --clipag switches both the weights tag and the resolved model.
  const clipag = LOSSES.clip.parse({ clipag: true });
  assert.equal(clipag.clipag, true);
  assert.equal(clipag.clipWeights, 'clipag');
  assert.equal(
    LOSSES.clip.outputFields({ ...clipag, subject: 'x' }).clipModel,
    'ViT-B-32',
  );
});

test('percentiles use linear interpolation', () => {
  assert.equal(percentile([1, 2, 3, 4], 0.5), 2.5);
  assert.equal(percentile([1, 2, 3, 4], 0.95), 3.8499999999999996);
});

test('opt-size accepts numeric and max sweeps', () => {
  assert.deepEqual(parseOptSizes(['64', 'max', 128, 'MAX']), [64, 'max', 128, 'max']);
  assert.deepEqual(resolveOptSize(128, 'l2', { width: 768, height: 512 }), {
    width: 128,
    height: 85,
  });
  assert.deepEqual(resolveOptSize('max', 'l2', { width: 320, height: 480 }), {
    width: 320,
    height: 480,
  });
  assert.deepEqual(resolveOptSize('max', 'clip'), { width: 224, height: 224 });
  assert.deepEqual(resolveOptSize(192, 'clip'), { width: 192, height: 192 });
  assert.equal(optSizeLabel({ width: 128, height: 85 }), '128x85');
  assert.throws(() => parseOptSizes(['full']), /positive integers or max/);
});

test('L2 max keeps native dimensions without cropping', () => {
  const source = {
    width: 2,
    height: 1,
    channels: 3,
    depth: 8,
    data: Uint8Array.of(255, 0, 0, 0, 0, 255),
  };
  const dimensions = resolveOptSize('max', 'l2', source);
  const target = targetFromSource(source, dimensions.width, dimensions.height);
  assert.equal(target.sourceWidth, 2);
  assert.equal(target.sourceHeight, 1);
  assert.equal(target.rgba.length, 2 * 1 * 4);
  assert.deepEqual(Array.from(target.rgba), [1, 0, 0, 1, 0, 0, 1, 1]);
});

test('resolved rectangular size appears in case labels', () => {
  assert.equal(caseLabel({
    subject: 'photo.png',
    n: 64,
    width: 128,
    height: 85,
    budget: { mode: 'steps', value: 200 },
  }), 'photo-png-n64-o128x85-s200');
});

test('equivalent sweep cells receive unique output IDs', () => {
  const base = {
    subject: 'photo.png',
    n: 64,
    width: 128,
    height: 85,
    budget: { mode: 'steps', value: 200 },
  };
  const cases = assignCaseIds([{ ...base, optSize: 128 }, { ...base, optSize: 'max' }]);
  assert.deepEqual(cases.map(caseLabel), [
    'photo-png-n64-o128x85-s200',
    'photo-png-n64-o128x85-s200-c2',
  ]);
});

test('checkpoint options are explicit and strict', () => {
  assert.deepEqual(parseSaveOptions({}), { saveEvery: null, saveBlur: false });
  assert.deepEqual(parseSaveOptions({ 'save-every': '25', 'save-blur': 'true' }), {
    saveEvery: 25,
    saveBlur: true,
  });
  assert.equal(parseSaveOptions({ 'save-every': '25' }).saveBlur, true);
  assert.throws(() => parseSaveOptions({ 'save-every': true }), /requires a value/);
  assert.throws(() => parseSaveOptions({ 'save-blur': 'true' }), /requires --save-every/);
  assert.throws(() => parseSaveOptions({ 'save-every': '5', 'save-blur': 'yes' }), /true or false/);
});

test('save size never downsamples the optimization canvas', () => {
  const source = { width: 768, height: 512 };
  assert.deepEqual(resolveSaveSize(128, 'l2', source, { width: 256, height: 171 }), {
    width: 256,
    height: 171,
  });
  assert.deepEqual(resolveSaveSize('max', 'l2', source, { width: 256, height: 171 }), {
    width: 768,
    height: 512,
  });
  assert.deepEqual(resolveSaveSize(2048, 'l2', source, { width: 256, height: 171 }), {
    width: 2048,
    height: 1365,
  });
  assert.deepEqual(resolveSaveSize('max', 'clip', null, { width: 512, height: 512 }), {
    width: 512,
    height: 512,
  });
});

test('checkpoint filters can be crisp or retain optimization blur', () => {
  const crisp = saveFrameSettings({
    width: 128,
    height: 128,
    saveWidth: 256,
    saveHeight: 256,
    blur: false,
    current: { s: [4, 4] },
    background: [1, 1, 1],
  });
  assert.deepEqual(crisp, {
    scale: 0.5,
    origin: [0, 0],
    s: [0.5, 0.5],
    bg: [1, 1, 1],
  });
  assert.deepEqual(saveFrameSettings({
    width: 128,
    height: 128,
    saveWidth: 256,
    saveHeight: 256,
    blur: true,
    current: { s: [4, 4] },
    background: [1, 1, 1],
  }).s, [4, 4]);
});

test('Windfoil hosts share one runtime mapping', () => {
  assert.deepEqual(WINDFOIL_ENVIRONMENTS, ['node', 'deno', 'deno-dawn']);
  assert.deepEqual(windfoilCommand('node', ['demo.js']).args, ['demo.js']);
  assert.deepEqual(
    windfoilCommand('deno', ['demo.js']).args,
    ['run', '--unstable-webgpu', '-A', '--node-modules-dir=auto', 'demo.js'],
  );
  assert.deepEqual(
    windfoilCommand('deno-dawn', ['demo.js']).args,
    ['run', '-A', '--node-modules-dir=auto', 'demo.js'],
  );
  assert.equal(windfoilBackend('node'), 'dawn');
  assert.equal(windfoilBackend('deno'), 'wgpu');
  assert.throws(() => windfoilCommand('other', []), /unsupported Windfoil environment/);
});

test('blur defaults scale with the optimization canvas above 512px', () => {
  // At or below the 512px reference the historical defaults are untouched.
  assert.equal(withBlurDefaults({ blur: null, blurFloor: null }, 1, 128, 128).blur, 7);
  assert.equal(withBlurDefaults({ blur: null, blurFloor: null }, 1, 512, 288).blur, 7);
  assert.equal(withBlurDefaults({ blur: null, blurFloor: null }, 2, 512, 512).blurFloor, 2);
  // Above it, the anneal covers the same fraction of the canvas.
  const big = withBlurDefaults({ blur: null, blurFloor: null }, 1, 2048, 1152);
  assert.equal(big.blur, 28);
  assert.equal(big.blurFloor, 4);
  // Explicit flags stay absolute at any resolution.
  const explicit = withBlurDefaults({ blur: 3, blurFloor: 0.5 }, 1, 4096, 4096);
  assert.equal(explicit.blur, 3);
  assert.equal(explicit.blurFloor, 0.5);
});
