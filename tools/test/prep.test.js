import assert from 'node:assert/strict';
import test from 'node:test';

import { packScene } from '../../js/prep.js';

function rootControls(t) {
  return [0, -t, 1 - 2 * t];
}

function curveWithRoots(tx, ty) {
  const x = rootControls(tx);
  const y = rootControls(ty);
  return [x[0], y[0], x[1], y[1], x[2], y[2]];
}

const roots = [
  [0, 0], [1, 1], [0, 0.4], [0.4, 1],
  [0, 1], [0.4, 0.4], [0.25, 0.75],
];
const curves = roots.flatMap(([tx, ty]) => curveWithRoots(tx, ty));
curves.push(3, 3, 3, 3, 3, 3);
const shapes = [
  { curves: Float64Array.from(curves.slice(0, 24)), color: [0.2, 0.4, 0.6], alpha: 0.75 },
  {
    curves: Float64Array.from(curves.slice(24)),
    color: [0.6, 0.4, 0.2],
    alpha: 0.5,
    fillRule: 'evenodd',
  },
];
// Shared across the tests below: the reuse test depends on these first views.
const scratch = {};
const scene = packScene(shapes, scratch);

test('packScene emits the compact scene fields and counts', () => {
  assert.deepEqual(
    Object.keys(scene).sort(),
    ['curveCount', 'curveMetaData', 'pieceCount', 'pieceData', 'shapeData'],
  );
  assert.equal(scene.curveCount, 8);
  assert.equal(scene.pieceCount, 12);
  assert.ok(scene.pieceData.length >= scene.pieceCount * 6, 'piece scratch is smaller than the live count');
  assert.ok(
    scratch.weights === undefined && scratch.splitT === undefined,
    'CPU pullback storage was allocated',
  );
});

test('curve metadata records split parameters, piece starts, and monotone masks', () => {
  const expectedMasks = [0b100, 0b001, 0b110, 0b011, 0b010, 0b101, 0b111, 0];
  const expectedStarts = [0, 1, 2, 4, 6, 7, 9, 12];
  const metaU32 = new Uint32Array(scene.curveMetaData.buffer);
  for (let i = 0; i < expectedMasks.length; i++) {
    assert.equal(metaU32[4 * i + 2], expectedStarts[i], `curve ${i} piece start`);
    assert.equal(metaU32[4 * i + 3], expectedMasks[i], `curve ${i} mask`);
    const expectedT = i < roots.length ? roots[i].slice().sort((a, b) => a - b) : [0, 0];
    assert.equal(scene.curveMetaData[4 * i], Math.fround(expectedT[0]), `curve ${i} t1`);
    assert.equal(scene.curveMetaData[4 * i + 1], Math.fround(expectedT[1]), `curve ${i} t2`);
  }
});

test('shape piece ranges and fill rules are packed', () => {
  const shapeMeta = new Uint32Array(scene.shapeData.buffer);
  assert.deepEqual([shapeMeta[8], shapeMeta[9]], [0, 6], 'first shape compact range');
  assert.deepEqual([shapeMeta[24], shapeMeta[25]], [6, 6], 'second shape compact range');
  assert.equal(shapeMeta[10], 0, 'omitted fill rule did not default to nonzero');
  assert.equal(shapeMeta[26], 1, 'evenodd fill rule was not packed');
});

test('no degenerate point pieces are emitted', () => {
  const pieceWords = new Uint32Array(
    scene.pieceData.buffer,
    scene.pieceData.byteOffset,
    scene.pieceCount * 6,
  );
  for (let piece = 0; piece < scene.pieceCount; piece++) {
    const o = piece * 6;
    const isPoint = pieceWords[o + 2] === pieceWords[o] && pieceWords[o + 3] === pieceWords[o + 1] &&
      pieceWords[o + 4] === pieceWords[o] && pieceWords[o + 5] === pieceWords[o + 1];
    assert.ok(!isPoint, `emitted point piece ${piece}`);
  }
});

test('scratch buffers are reused and a stale fill-rule word is reset', () => {
  const again = packScene(shapes, scratch);
  assert.strictEqual(again.pieceData, scene.pieceData, 'stable compact view was reallocated');
  assert.strictEqual(again.shapeData, scene.shapeData, 'shape scratch was reallocated');
  assert.strictEqual(again.curveMetaData, scene.curveMetaData, 'curve metadata was reallocated');

  delete shapes[1].fillRule;
  try {
    const defaulted = packScene(shapes, scratch);
    assert.equal(new Uint32Array(defaulted.shapeData.buffer)[26], 0, 'reused fill rule word was not reset');
  } finally {
    shapes[1].fillRule = 'evenodd';
  }
});

test('curves that collapse to a point in f32 are dropped', () => {
  const roundedPoint = [{
    curves: Float64Array.of(1e8, 3, 1e8 + 1, 3, 1e8 + 2, 3),
    color: [0, 0, 0],
    alpha: 1,
  }];
  assert.equal(packScene(roundedPoint).pieceCount, 0, 'f32-identical point curve was retained');
  roundedPoint[0].curves = Float64Array.of(1e8, 3, 1e8 + 8, 3, 1e8 + 16, 3);
  assert.equal(packScene(roundedPoint).pieceCount, 1, 'f32-distinct curve was removed');
});

test('malformed curve data and unknown fill rules are rejected', () => {
  assert.throws(() => packScene([{ curves: [0, 1], color: [0, 0, 0] }]));
  assert.throws(
    () => packScene([{ curves: curves.slice(0, 6), color: [0, 0, 0], fillRule: 'winding' }]),
    /fillRule/,
  );
});
