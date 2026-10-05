import assert from 'node:assert/strict';
import test from 'node:test';

import { anchorStyle, decodeAnchors } from '../../js/color-anchors.js';
import { learnedPaletteStyle } from '../../js/learned-palette.js';
import { withBackgroundColor } from '../../js/background-model.js';
import { GenericModel } from '../../js/generic/model.js';
import { roundCapsule } from '../../js/generic/primitives.js';
import { LoopModel } from '../../js/loop-model.js';
import { buildLineModel, lineCli } from '../../demos/lines/model.js';
import { buildPlotModel, plotCli } from '../../demos/plot/model.js';
import { buildRoundtripModel } from '../../demos/roundtrip/model.js';
import { annealedBlur, buildModel, createInit, shapeCli } from '../../demos/util/model.js';

test('LoopModel pullback matches finite differences', () => {
  const style = anchorStyle();
  const styleParams = style.init([[0.3, 0.5, 0.7], [0.7, 0.4, 0.2]], [0.8, 0.55]);
  const model = new LoopModel({
    ax: Float64Array.of(5, 24, 20, 34, 51, 46),
    ay: Float64Array.of(6, 8, 25, 9, 13, 28),
    cx: Float64Array.of(14, 28, 8, 43, 55, 35),
    cy: Float64Array.of(2, 18, 19, 5, 22, 19),
    n: 2,
    k: 3,
    style,
    styleParams,
  });
  checkModel(model);
});

test('anchor style decodes what it was seeded with', () => {
  const style = anchorStyle();
  const { colorAnchor } = style.init([[0.3, 0.5, 0.7]], [0.8]);
  const color = [0, 0, 0];
  const alpha = decodeAnchors(colorAnchor, 0, color);
  // The mobility floor pulls weights slightly toward uniform; the decode must
  // still land near the seed.
  for (let c = 0; c < 3; c++) assert.ok(Math.abs(color[c] - [0.3, 0.5, 0.7][c]) < 0.06, `c${c}=${color[c]}`);
  assert.ok(Math.abs(alpha - 0.8) < 0.1, `alpha=${alpha}`);
});

test('GenericModel pullback matches finite differences', () => {
  const style = anchorStyle();
  const params = {
    x0: Float64Array.of(5, 31),
    y0: Float64Array.of(9, 7),
    x1: Float64Array.of(25, 49),
    y1: Float64Array.of(18, 25),
    width: Float64Array.of(3, 4),
    ...style.init([[0.6, 0.4, 0.2]], [0.65]),
  };
  const model = new GenericModel({
    params,
    style,
    build: (tape, p) => Array.from({ length: 2 }, (_, i) => ({
      curves: roundCapsule(tape, p.x0[i], p.y0[i], p.x1[i], p.y1[i], p.width[i]),
      color: 0,
      alpha: 0,
    })),
  });
  checkModel(model);
});

for (const opaque of [false, true]) {
  test(`learned-palette pullback matches finite differences (${opaque ? 'opaque' : 'translucent'})`, () => {
    const style = learnedPaletteStyle({ count: 3, opaque, seed: 4 });
    const params = {
      x0: Float64Array.of(5, 31),
      y0: Float64Array.of(9, 7),
      x1: Float64Array.of(25, 49),
      y1: Float64Array.of(18, 25),
      width: Float64Array.of(3, 4),
      ...style.init([[0, 0, 0], [0, 0, 0]], [1, 1]),
    };
    const model = new GenericModel({
      params,
      style,
      build: (tape, p) => Array.from({ length: 2 }, (_, i) => ({
        curves: roundCapsule(tape, p.x0[i], p.y0[i], p.x1[i], p.y1[i], p.width[i]),
        color: i,
        alpha: i,
      })),
    });
    checkModel(model);
  });
}

test('learned palette hardens each shape to one discovered color', () => {
  const style = learnedPaletteStyle({ count: 4, opaque: true, seed: 1 });
  const styleParams = style.init([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [1, 1, 1]);
  // Push each shape's assignment to a distinct palette entry.
  styleParams.assign.fill(0);
  for (let i = 0; i < 3; i++) styleParams.assign[i * 4 + i] = 10;
  const params = {
    x0: Float64Array.of(1, 2, 3), y0: Float64Array.of(1, 2, 3),
    x1: Float64Array.of(4, 5, 6), y1: Float64Array.of(4, 5, 6),
    width: Float64Array.of(1, 1, 1), ...styleParams,
  };
  const model = new GenericModel({
    params, style,
    build: (tape, p) => Array.from({ length: 3 }, (_, i) => ({
      curves: roundCapsule(tape, p.x0[i], p.y0[i], p.x1[i], p.y1[i], p.width[i]),
      color: i, alpha: i,
    })),
  });
  const palette = style.paletteColors(params);
  model.harden();
  const { shapes } = model.decode();
  shapes.forEach((shape, i) => {
    assert.ok(shape.alpha > 0.99);
    for (let c = 0; c < 3; c++) assert.ok(Math.abs(shape.color[c] - palette[i][c]) < 1e-9);
  });
});

test('palette fidelity pulls learned colors toward the target colors', () => {
  const targets = [[0.9, 0.12, 0.1], [0.1, 0.12, 0.9]];
  const build = (fidelity) => {
    const style = learnedPaletteStyle({ count: 2, opaque: true, seed: 3, fidelity });
    const styleParams = style.init(targets, [1, 1]);
    styleParams.assign.fill(0);
    styleParams.assign[0] = 10; // shape 0 -> entry 0
    styleParams.assign[1 * 2 + 1] = 10; // shape 1 -> entry 1
    const params = {
      x0: Float64Array.of(1, 2), y0: Float64Array.of(1, 2),
      x1: Float64Array.of(4, 5), y1: Float64Array.of(4, 5),
      width: Float64Array.of(1, 1), ...styleParams,
    };
    const model = new GenericModel({
      params, style,
      build: (tape, p) => Array.from({ length: 2 }, (_, i) => ({
        curves: roundCapsule(tape, p.x0[i], p.y0[i], p.x1[i], p.y1[i], p.width[i]),
        color: i, alpha: i,
      })),
    });
    return { style, params, model };
  };
  // The regularizer scales to the fit gradient, so use a nonzero pixel cotangent
  // and isolate its effect as the difference from the fidelity-0 gradient.
  const base = build(0), fid = build(5);
  const curveGrads = Float32Array.from({ length: base.model.maxCurves * 6 }, (_, i) => Math.sin(i * 1.3) * 0.02);
  const shapeGrads = Float32Array.from({ length: base.model.n * 4 }, (_, i) => Math.cos(i * 1.1) * 0.03);
  const g0 = Float64Array.from(base.model.pullback({ curveGrads, shapeGrads }).palette);
  const g1 = Float64Array.from(fid.model.pullback({ curveGrads, shapeGrads }).palette);
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const before = fid.style.paletteColors(fid.params);
  // Apply only the regularizer's contribution (fidelity minus fit-only gradient).
  for (let i = 0; i < fid.params.palette.length; i++) fid.params.palette[i] -= 0.5 * (g1[i] - g0[i]);
  const after = fid.style.paletteColors(fid.params);
  assert.ok(g1.some((g, i) => g !== g0[i]), 'fidelity changed the palette gradient');
  for (let j = 0; j < 2; j++) {
    assert.ok(dist(after[j], targets[j]) < dist(before[j], targets[j]), `entry ${j} moved closer`);
  }
});

test('background-color decorator adds a finite-difference-correct bg gradient', () => {
  const style = anchorStyle();
  const params = {
    x0: Float64Array.of(5, 31), y0: Float64Array.of(9, 7),
    x1: Float64Array.of(25, 49), y1: Float64Array.of(18, 25),
    width: Float64Array.of(3, 4),
    ...style.init([[0.6, 0.4, 0.2]], [0.65]),
  };
  const inner = new GenericModel({
    params,
    style,
    build: (tape, p) => Array.from({ length: 2 }, (_, i) => ({
      curves: roundCapsule(tape, p.x0[i], p.y0[i], p.x1[i], p.y1[i], p.width[i]),
      color: 0,
      alpha: 0,
    })),
  });
  const { model, lrs } = withBackgroundColor(inner, { existing: { lr: 1 } }, {
    width: 64,
    height: 64,
    background: [0.3, 0.5, 0.7],
  });
  assert.ok(lrs.bg && lrs.existing, 'keeps existing lrs and adds bg');
  assert.equal(model.maxShapes, inner.maxShapes + 1);
  assert.equal(model.maxCurves, inner.maxCurves + 4);
  // The bg param decodes as the bottom (index 0) full-canvas opaque shape.
  const { shapes } = model.decode();
  assert.equal(shapes.length, 3);
  assert.ok(shapes[0].alpha > 0.99);
  for (let c = 0; c < 3; c++) {
    assert.ok(Math.abs(shapes[0].color[c] - [0.3, 0.5, 0.7][c]) < 1e-6);
  }
  checkModel(model);
});

test('background-color decorator offsets every gradient family past its shape', () => {
  let seen = null;
  const inner = {
    params: {}, n: 2, maxShapes: 2, maxCurves: 3,
    decode: () => ({ shapes: [] }),
    pullback: (result) => { seen = result; return {}; },
  };
  const { model } = withBackgroundColor(inner, {}, { width: 8, height: 8, background: [0, 0, 0] });
  const ramp = (length) => Float32Array.from({ length }, (_, i) => i);

  // 4 prepended curves (24 coordinates), one stride-4 shape slot, one blur pair.
  const grads = model.pullback({
    curveGrads: ramp(24 + 18), shapeGrads: ramp(4 + 8), blurGrads: ramp(2 + 4),
  });
  assert.deepEqual([seen.curveGrads[0], seen.curveGrads.length], [24, 18]);
  assert.deepEqual([seen.shapeGrads[0], seen.shapeGrads.length], [4, 8]);
  assert.deepEqual([seen.blurGrads[0], seen.blurGrads.length], [2, 4]);
  assert.deepEqual(Array.from(grads.bg), [0, 1, 2]);

  // A frozen group's array is absent; that absence passes through unindexed.
  const frozen = model.pullback({ shapeGrads: ramp(4 + 8) });
  assert.deepEqual(Object.keys(seen), ['shapeGrads']);
  assert.deepEqual(Array.from(frozen.bg), [0, 1, 2]);
  assert.deepEqual(Array.from(model.pullback({ curveGrads: ramp(24 + 18) }).bg), [0, 0, 0]);
});

test('demo models preserve rectangular dimensions', () => {
  const target = new Float32Array(12 * 6 * 4).fill(0.5);
  const initial = createInit({ n: 4, width: 12, height: 6, k: 4, target });
  assert.equal(initial.size, undefined);
  assert.equal(initial.width, 12);
  assert.equal(initial.height, 6);
  const loops = buildModel(initial);
  assert.equal(loops.lrs.ax.lr, 0.5 * 12 / 128);
  assert.equal(loops.lrs.ay.lr, 0.5 * 6 / 128);

  const lines = buildLineModel({ n: 2, width: 12, height: 6, target });
  assert.equal(lines.lrs.x0.lr, 0.5 * 12 / 128);
  assert.equal(lines.lrs.y0.lr, 0.5 * 6 / 128);
  assert.equal(lines.model.params.width.length, 2);
});

test('line model thickness stays positive when the raw width goes negative', () => {
  const target = new Float32Array(32 * 32 * 4).fill(0.5);
  // Signed area of the outline's anchor polygon: its sign tracks winding, so a
  // width that crossed zero would flip it.
  const area = (shape) => {
    let sum = 0;
    const c = shape.curves;
    for (let i = 0; i < c.length; i += 6) {
      sum += c[i] * c[i + 5] - c[i + 4] * c[i + 1];
    }
    return 0.5 * sum;
  };
  for (const primitive of ['capsule', 'line', 'point']) {
    const { model } = buildLineModel({ n: 1, width: 32, height: 32, target, primitive });
    const before = area(model.decode().shapes[0]);
    model.params.width[0] = -50;
    const after = area(model.decode().shapes[0]);
    assert.ok(Number.isFinite(after), `${primitive}: decoded outline must stay finite`);
    assert.ok(before * after > 0, `${primitive}: winding must not flip (${before} -> ${after})`);
    assert.ok(Math.abs(after) < Math.abs(before), `${primitive}: negative raw width must shrink the outline`);
  }
});

test('plot line model draws fixed-color pens with bounded lengths', () => {
  const target = new Float32Array(8 * 4 * 4).fill(0.25);
  const plot = buildPlotModel({
    n: 3,
    width: 8,
    height: 4,
    canvasCm: 18,
    penMm: 0.5,
    minLenCm: 0.2,
    maxLenCm: 1,
    colors: [[0, 0, 0], [1, 0, 1]],
    target,
  });
  assert.equal(plot.mode, 'line');
  assert.equal(plot.canvas.widthCm, 18);
  assert.equal(plot.canvas.heightCm, 9);
  assert.equal(plot.canvas.penWidthCm, 0.05);
  // A round capsule is a six-curve loop; color has no trainable anchors.
  assert.equal(plot.model.maxCurves, 3 * 6);
  assert.equal(plot.lrs.colorAnchor, undefined);
  assert.equal(plot.lrs.px.lr, 0.5 * 18 / 128);
  assert.equal(plot.lrs.py.lr, 0.5 * 9 / 128);
  assert.ok(plot.lrs.theta.lr > 0);
  assert.ok(plot.lrs.len.lr > 0);

  const markers = plot.markers();
  assert.equal(markers.length, 3);
  markers.forEach((m, i) => {
    assert.equal(m.mode, 'line');
    // Endpoints stay symmetric around the trained center.
    assert.ok(Math.abs(0.5 * (m.x0 + m.x1) - plot.model.params.px[i]) < 1e-6);
    assert.ok(Math.abs(0.5 * (m.y0 + m.y1) - plot.model.params.py[i]) < 1e-6);
    // Length lands inside the requested [min, max] band.
    const length = Math.hypot(m.x1 - m.x0, m.y1 - m.y0);
    assert.ok(length >= 0.2 - 1e-6 && length <= 1 + 1e-6, `length=${length}`);
    // Each marker keeps one of the two palette colors.
    assert.ok(m.color === plot.palette[0] || m.color === plot.palette[1]);
  });

  const { shapes } = plot.model.decode();
  shapes.forEach((shape) => {
    assert.ok(shape.alpha > 0.99);
    const isBlack = shape.color.every((channel) => channel < 0.01);
    const isMagenta =
      shape.color[0] > 0.99 && shape.color[1] < 0.01 && shape.color[2] > 0.99;
    assert.ok(isBlack || isMagenta);
  });
  checkModel(plot.model);

  // Even a runaway length parameter stays within the length band.
  plot.model.params.len[0] = 1e6;
  const long = plot.markers()[0];
  assert.ok(Math.hypot(long.x1 - long.x0, long.y1 - long.y0) <= 1 + 1e-6);
  plot.model.params.len[0] = -1e6;
  const short = plot.markers()[0];
  assert.ok(Math.hypot(short.x1 - short.x0, short.y1 - short.y0) >= 0.2 - 1e-6);
});

test('plot point model stipples fixed-radius discs in one default pen', () => {
  const target = new Float32Array(8 * 4 * 4).fill(0.25);
  const plot = buildPlotModel({
    n: 4,
    width: 8,
    height: 4,
    mode: 'point',
    canvasCm: 18,
    penMm: 0.5,
    target,
  });
  assert.equal(plot.mode, 'point');
  assert.deepEqual(plot.palette, [[0, 0, 0]]);
  // Point markers train only in position — no angle or length parameters.
  assert.equal(plot.model.params.theta, undefined);
  assert.equal(plot.model.params.len, undefined);
  assert.equal(plot.lrs.theta, undefined);

  const markers = plot.markers();
  assert.equal(markers.length, 4);
  markers.forEach((m, i) => {
    assert.equal(m.mode, 'point');
    assert.equal(m.x, plot.model.params.px[i]);
    assert.equal(m.y, plot.model.params.py[i]);
    assert.deepEqual(m.color, [0, 0, 0]);
  });

  const { shapes } = plot.model.decode();
  shapes.forEach((shape) => {
    assert.ok(shape.color.every((channel) => channel < 0.01));
    assert.ok(shape.alpha > 0.99);
  });
  checkModel(plot.model);
});

test('model descriptors expose a consistent CLI contract', () => {
  assert.deepEqual(shapeCli.defaults, { n: 512, steps: 500, size: 512 });
  assert.equal(shapeCli.primitive(null), 'quadratic-loop');
  assert.equal(shapeCli.supportsInit, true);
  assert.deepEqual(shapeCli.parse({ k: '5' }), { k: 5, colorCount: 1, opaque: false, fidelity: 0, learnBlur: false });
  assert.deepEqual(
    shapeCli.parse({ 'color-count': '4', opaque: true, 'palette-fidelity': '2' }),
    { k: 8, colorCount: 4, opaque: true, fidelity: 2, learnBlur: false },
  );

  assert.equal(lineCli.primitive(null), 'round-capsule');
  assert.equal(lineCli.supportsInit, false);
  assert.deepEqual(lineCli.parse({}), {
    k: 6, colorCount: 1, opaque: false, fidelity: 0, learnBlur: false,
  });
  assert.equal(lineCli.parse({ 'learn-blur': true }).learnBlur, true);

  assert.deepEqual(plotCli.defaults, { n: 4000, steps: 800, size: 512 });
  assert.equal(plotCli.bandLimited, true);
  assert.equal(plotCli.supportsInit, false);
  // Plot flag defaults: line mode, single black pen (no --colors).
  const line = plotCli.parse({});
  assert.equal(line.mode, 'line');
  assert.equal(line.colors, null);
  // --primitive=point + an explicit palette parse into rgb triples.
  const point = plotCli.parse({ primitive: 'point', colors: '#ff00ff,#00ff00' });
  assert.equal(point.mode, 'point');
  assert.deepEqual(point.colors, [[1, 0, 1], [0, 1, 0]]);
  assert.throws(() => plotCli.parse({ primitive: 'squiggle' }), /line.*point/);
  assert.equal(plotCli.primitive({ mode: 'point' }), 'stipple-disc');
  assert.equal(plotCli.primitive({ mode: 'line' }), 'round-capsule');
  // --color-count enables discovery; it and a fixed --colors are exclusive.
  assert.equal(line.colorCount, 1);
  assert.equal(plotCli.parse({ 'color-count': '5' }).colorCount, 5);
  assert.throws(() => plotCli.parse({ 'color-count': '1' }), />= 2/);
  assert.throws(
    () => plotCli.parse({ colors: '#000', 'color-count': '3' }),
    /only one/,
  );
});

test('plot prepareTarget honours the colors / grayscale rule', () => {
  const make = () => ({
    rgba: Float32Array.of(1, 0, 0, 1, 0, 1, 0, 1),
    u8: null,
    mean: [0.5, 0.5, 0],
  });
  // No pens -> target is made b&w.
  const gray = make();
  plotCli.prepareTarget(gray, { colors: null, grayscale: false });
  assert.equal(gray.rgba[0], gray.rgba[1]);
  // Pens, no --grayscale -> target stays in color.
  const color = make();
  plotCli.prepareTarget(color, { colors: [[1, 0, 1]], grayscale: false });
  assert.notEqual(color.rgba[0], color.rgba[1]);
  // Pens + --grayscale -> target is made b&w.
  const both = make();
  plotCli.prepareTarget(both, { colors: [[1, 0, 1]], grayscale: true });
  assert.equal(both.rgba[0], both.rgba[1]);
});

test('anneal schedule can settle on a blur floor', () => {
  assert.equal(annealedBlur(0, 100, 7, 2), 7);
  assert.equal(annealedBlur(55, 100, 7, 2), 2);
  assert.equal(annealedBlur(100, 100, 7, 2), 2);
  assert.equal(annealedBlur(55, 100, 7, 1), 1);
  assert.equal(annealedBlur(0, 100, 1, 1), 1);
});

test('bounded learned blur stays within its range and has the correct gradient', () => {
  const options = {
    n: 1, width: 64, height: 32, seed: 7,
    learnBlur: true, blurInit: 7, blurFloor: 1, blurCeiling: 32,
  };
  const models = [
    buildModel({ ...createInit(options), ...options }).model,
    buildLineModel({ ...options, primitive: 'capsule' }).model,
  ];
  for (const model of models) {
    const widthAt = (parameter) => {
      model.params.blur[0] = parameter;
      return model.decode().shapes[0].s;
    };
    for (const parameter of [-1000, -1, 0, 1, 1000]) {
      const width = widthAt(parameter);
      assert.ok(width >= 1 && width <= 32);
      // The x and y filter gradients both flow into the one shared width.
      const analytic = model.pullback({ blurGrads: Float32Array.of(1, 2) }).blur[0];
      const h = 0.001;
      const numeric = (widthAt(parameter + h) - widthAt(parameter - h)) / (2 * h);
      assert.ok(Math.abs(analytic - 3 * numeric) < 0.002);
    }
  }
});

test('roundtrip model preserves ragged topology, styles, rules, and shared joins', () => {
  const triangle = {
    curves: Float64Array.of(
      1, 1, 3, 0, 5, 1,
      5, 1, 6, 4, 3, 6,
      3, 6, 0, 4, 1, 1,
    ),
    color: [0.2, 0.4, 0.8],
    alpha: 0.75,
    fillRule: 'evenodd',
    contours: [3],
  };
  const loop = {
    curves: Float64Array.of(
      8, 2, 10, 1, 12, 2,
      12, 2, 10, 5, 8, 2,
    ),
    color: [0.8, 0.3, 0.1],
    alpha: 1,
  };
  const { model, lrs } = buildRoundtripModel([triangle, loop], {
    translation: [2, -1],
    width: 20,
    height: 10,
  });
  const decoded = model.decode();
  assert.equal(model.maxCurves, 5);
  assert.equal(decoded.shapes[0].fillRule, 'evenodd');
  assert.equal(decoded.shapes[1].fillRule, 'nonzero');
  assert.deepEqual(decoded.shapes[0].color, triangle.color);
  assert.equal(decoded.shapes[0].alpha, triangle.alpha);
  assert.deepEqual([...decoded.shapes[0].curves.slice(0, 2)], [3, 0]);
  assert.deepEqual([...decoded.shapes[0].curves.slice(-2)], [3, 0]);
  assert.equal(lrs.ax.lr, 0.5 * 20 / 128);
  assert.equal(lrs.ay.lr, 0.5 * 10 / 128);
  checkModel(model);
});

test('roundtrip model retains multiple contours in one compound path', () => {
  const shape = {
    curves: Float64Array.of(
      0, 0, 1, 0, 2, 0,
      2, 0, 1, 2, 0, 0,
      4, 0, 5, 0, 6, 0,
      6, 0, 5, 2, 4, 0,
    ),
    contours: [2, 2],
    color: [0.1, 0.2, 0.3],
    alpha: 1,
    fillRule: 'evenodd',
  };
  const { model } = buildRoundtripModel([shape], {
    translation: [1, 2], width: 8, height: 4,
  });
  const decoded = model.decode();
  assert.deepEqual(decoded.shapes[0].contours, [2, 2]);
  assert.deepEqual([...decoded.shapes[0].curves.slice(0, 2)], [1, 2]);
  assert.deepEqual([...decoded.shapes[0].curves.slice(12, 14)], [5, 2]);
  checkModel(model);
});

function checkModel(model) {
  const decoded = model.decode();
  const curveCount = decoded.shapes.reduce((sum, shape) => sum + shape.curves.length / 6, 0);
  const curveGrads = Float32Array.from({ length: curveCount * 6 }, (_, i) => Math.sin(i * 1.7) * 0.03);
  const shapeGrads = Float32Array.from(
    { length: model.n * 4 },
    (_, i) => Math.cos(i * 1.3) * 0.05,
  );
  assert.throws(() => model.pullback({ curveGrads: curveGrads.subarray(1), shapeGrads }));
  assert.throws(() => model.pullback({ curveGrads, shapeGrads: shapeGrads.subarray(1) }));
  const analytic = model.pullback({ curveGrads, shapeGrads });
  const epsilon = 1e-5;
  for (const [group, values] of Object.entries(model.params)) {
    for (let i = 0; i < values.length; i++) {
      const original = values[i];
      values[i] = original + epsilon;
      const highX = values[i]; // effective step after the array's rounding
      const high = objective(model, curveGrads, shapeGrads);
      values[i] = original - epsilon;
      const lowX = values[i];
      const low = objective(model, curveGrads, shapeGrads);
      values[i] = original;
      const numeric = (high - low) / (highX - lowX);
      const scale = Math.max(1e-6, Math.abs(numeric), Math.abs(analytic[group][i]));
      assert.ok(
        Math.abs(numeric - analytic[group][i]) / scale < 3e-4,
        `${group}[${i}]: analytic=${analytic[group][i]} numeric=${numeric}`,
      );
    }
  }
}

function objective(model, curveWeights, shapeWeights) {
  const { shapes } = model.decode();
  let value = 0;
  let curve = 0;
  for (let index = 0; index < shapes.length; index++) {
    const shape = shapes[index];
    for (const coordinate of shape.curves) value += coordinate * curveWeights[curve++];
    for (let c = 0; c < 3; c++) value += shape.color[c] * shapeWeights[4 * index + c];
    value += shape.alpha * shapeWeights[4 * index + 3];
  }
  return value;
}
