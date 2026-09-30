import assert from 'node:assert/strict';
import test from 'node:test';

import { maxRenderSide, planChunks } from '../../demos/render/tiled.js';

// WebGPU defaults: 128 MiB storage bindings, 65535 workgroups per dimension.
const DEFAULT_LIMITS = {
  maxBufferSize: 268435456,
  maxStorageBufferBindingSize: 134217728,
  maxComputeWorkgroupsPerDimension: 65535,
};

// A 4 GiB-binding adapter: the bin_sort dispatch axis becomes the bound.
const LARGE_LIMITS = {
  maxBufferSize: 4294967292,
  maxStorageBufferBindingSize: 4294967292,
  maxComputeWorkgroupsPerDimension: 65535,
};

test('max render side honors binding bytes and the bin dispatch axis', () => {
  // sqrt(128 MiB / 16 B) = 2896.3; bins would allow 16 * 255 = 4080.
  assert.equal(maxRenderSide(DEFAULT_LIMITS), 2896);
  assert.equal(maxRenderSide(LARGE_LIMITS), 4080);
  assert.equal(maxRenderSide({ ...LARGE_LIMITS, maxBufferSize: 16 * 16 * 16 }), 16);
  assert.throws(() => maxRenderSide({ ...LARGE_LIMITS, maxBufferSize: 256 }), /cannot fit/);
});

test('images within one pass stay a single exact-size chunk', () => {
  const plan = planChunks({ width: 1024, height: 768, limits: DEFAULT_LIMITS });
  assert.equal(plan.tileWidth, 1024);
  assert.equal(plan.tileHeight, 768);
  assert.deepEqual([plan.columns, plan.rows], [1, 1]);
  assert.deepEqual(plan.chunks, [{ x: 0, y: 0, width: 1024, height: 768 }]);
});

test('oversized images split at the device limit with edge remainders', () => {
  const plan = planChunks({ width: 6000, height: 500, limits: DEFAULT_LIMITS });
  assert.equal(plan.tileWidth, 2896);
  assert.equal(plan.tileHeight, 500);
  assert.deepEqual([plan.columns, plan.rows], [3, 1]);
  assert.deepEqual(plan.chunks.map((c) => c.width), [2896, 2896, 208]);
});

test('an 18k render on a large adapter uses the default 4080px grid', () => {
  const plan = planChunks({ width: 18000, height: 18000, limits: LARGE_LIMITS });
  assert.equal(plan.tileWidth, 4080);
  assert.deepEqual([plan.columns, plan.rows], [5, 5]);
  assert.equal(plan.chunks.length, 25);
  assert.deepEqual(plan.chunks.at(-1), { x: 16320, y: 16320, width: 1680, height: 1680 });
});

test('chunks partition the raster exactly once', () => {
  const width = 33;
  const height = 17;
  const plan = planChunks({ width, height, limits: DEFAULT_LIMITS, chunk: 16 });
  const covered = new Uint8Array(width * height);
  for (const { x, y, width: w, height: h } of plan.chunks) {
    for (let row = y; row < y + h; row++) {
      for (let col = x; col < x + w; col++) covered[row * width + col]++;
    }
  }
  assert.ok(covered.every((count) => count === 1));
});

test('requested chunk sizes are clamped to the device and validated', () => {
  const clamped = planChunks({ width: 8000, height: 8000, limits: DEFAULT_LIMITS, chunk: 100000 });
  assert.equal(clamped.tileWidth, 2896);
  const small = planChunks({ width: 64, height: 64, limits: DEFAULT_LIMITS, chunk: 24 });
  assert.deepEqual([small.columns, small.rows], [3, 3]);
  assert.throws(() => planChunks({ width: 64, height: 64, limits: DEFAULT_LIMITS, chunk: 0 }), /positive integer/);
  assert.throws(() => planChunks({ width: 0, height: 64, limits: DEFAULT_LIMITS }), /positive integers/);
});
