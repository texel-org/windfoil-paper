import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import {
  BLEND_MODES,
  Renderer,
  countTileEntries,
  growTileEntryCapacity,
  sortCapacity,
  sortCapacityFor,
} from '../../js/renderer.js';
import {
  CUBIC_CENTER,
  CUBIC_INNER,
  CUBIC_MID,
  CUBIC_OUTER,
  GL3_CENTER,
  GL3_EDGE,
  GL3_NODE,
  KERNELS,
  resolveKernel,
} from '../../js/filter-kernels.js';

function shapeData(shapes) {
  const out = new Float32Array(shapes.length * 16);
  for (let i = 0; i < shapes.length; i++) {
    const { bbox, s = null } = shapes[i];
    out.set(bbox, 16 * i);
    if (s) out.set(s, 16 * i + 12);
  }
  return out;
}

const settings = { width: 32, height: 32, s: [1, 1], scale: 1, origin: [0, 0] };

test('compact tile count handles empty, offscreen, and rectangular scenes', () => {
  assert.equal(countTileEntries(new Float32Array(), settings), 0);
  assert.equal(countTileEntries(shapeData([{ bbox: [8, 8, 8, 8] }]), settings), 1);
  assert.equal(countTileEntries(shapeData([{ bbox: [15.5, 8, 15.5, 8] }]), settings), 2);
  assert.equal(countTileEntries(shapeData([{ bbox: [0, 0, 31, 31] }]), settings), 4);
  assert.equal(countTileEntries(shapeData([{ bbox: [40, 8, 50, 10] }]), settings), 0);
  assert.equal(countTileEntries(
    shapeData([{ bbox: [14, 8, 14, 8] }]),
    { width: 64, height: 16, s: [1, 1], scale: 2, origin: [-16, 0] },
  ), 1);
});

test('per-shape filters override the global filter', () => {
  const shapes = shapeData([
    { bbox: [8, 8, 8, 8] },
    { bbox: [8, 8, 8, 8], s: [40, 40] },
  ]);
  assert.equal(countTileEntries(shapes, settings), 1 + 4);
});

test('the kernel radius expands the binned footprint', () => {
  const wide = { width: 64, height: 64, s: [28, 28], scale: 1, origin: [0, 0] };
  const shapes = shapeData([{ bbox: [8, 8, 8, 8] }]);
  assert.equal(countTileEntries(shapes, wide), 4);
  assert.equal(countTileEntries(shapes, { ...wide, kernel: 'box' }), 4);
  // Conservatively bin to the tent target profile's 2s support.
  assert.equal(countTileEntries(shapes, { ...wide, kernel: 'tent' }), 9);
  // Conservatively bin to the cubic target profile's 4s support.
  assert.equal(countTileEntries(shapes, { ...wide, kernel: 'cubic' }), 16);
  assert.throws(
    () => countTileEntries(shapes, { ...wide, kernel: 'gauss' }),
    /unknown filter kernel/,
  );
});

test('the WGSL kernel constants match the JS registry', async () => {
  const source = await readFile(new URL('../../js/wgsl/coverage.wgsl', import.meta.url), 'utf8');
  assert.ok(source.includes(`const GL3_NODE : f32 = ${GL3_NODE};`));
  assert.ok(source.includes(`const GL3_EDGE : f32 = ${GL3_EDGE};`));
  assert.ok(source.includes(`const GL3_CENTER : f32 = ${GL3_CENTER};`));
  assert.ok(source.includes(`const CUBIC_OUTER : f32 = ${CUBIC_OUTER};`));
  assert.ok(source.includes(`const CUBIC_MID : f32 = ${CUBIC_MID};`));
  assert.ok(source.includes(`const CUBIC_INNER : f32 = ${CUBIC_INNER};`));
  assert.ok(source.includes(`const CUBIC_CENTER : f32 = ${CUBIC_CENTER};`));
  assert.equal(KERNELS.box.code, 0);
  assert.equal(KERNELS.tent.code, 1);
  assert.equal(KERNELS.cubic.code, 2);
  assert.match(source, /fn kernel_radius\(\)[\s\S]*KERNEL_CUBIC\) \{ return 2\.0; \}/);
  assert.equal(resolveKernel('box').radius, 0.5);
  assert.equal(resolveKernel('tent').radius, 1);
  assert.equal(resolveKernel('cubic').radius, 2);
  assert.equal(resolveKernel('cubic').boxPasses, 4);
  assert.equal(resolveKernel('cubic').taps.length, 7);
  assert.equal(resolveKernel().code, 0);
  assert.throws(() => resolveKernel('toString'), /unknown filter kernel/);
  assert.throws(() => resolveKernel('__proto__'), /unknown filter kernel/);
  assert.match(source, /KERNEL_CUBIC\) \{ return 7u; \}/);
  for (const line of [
    'i == 0u) { return vec2<f32>(-3.0 * GL3_NODE, CUBIC_OUTER);',
    'i == 1u) { return vec2<f32>(-2.0 * GL3_NODE, CUBIC_MID);',
    'i == 2u) { return vec2<f32>(-GL3_NODE, CUBIC_INNER);',
    'i == 3u) { return vec2<f32>(0.0, CUBIC_CENTER);',
    'i == 4u) { return vec2<f32>(GL3_NODE, CUBIC_INNER);',
    'i == 5u) { return vec2<f32>(2.0 * GL3_NODE, CUBIC_MID);',
    'return vec2<f32>(3.0 * GL3_NODE, CUBIC_OUTER);',
  ]) assert.ok(source.includes(line), line);
  // Tap weights integrate the residual profile exactly (they sum to one).
  for (const { taps } of Object.values(KERNELS)) {
    const total = taps.reduce((sum, [, weight]) => sum + weight, 0);
    assert.ok(Math.abs(total - 1) < 1e-15);
    assert.ok(Object.isFrozen(taps));
    assert.ok(taps.every(Object.isFrozen));
  }
});

test('box keeps its direct backward and multi-tap pieces sum taps before atomics', async () => {
  const source = await readFile(new URL('../../js/wgsl/render.wgsl', import.meta.url), 'utf8');
  const start = source.indexOf('if (FILTER_KERNEL == KERNEL_BOX) {', source.indexOf('fn backward('));
  const end = source.indexOf('P *= (1.0 - a);', start);
  const backward = source.slice(start, end);
  const boxPieceLoop = backward.indexOf('for (var p = 0u; p < count; p++)');
  const boxAtomic = backward.indexOf('add_piece_grad(o, g.dq1.x);', boxPieceLoop);
  const multi = backward.indexOf('} else {');
  const pieceLoop = backward.indexOf('for (var p = 0u; p < count; p++)', multi);
  const rowLoop = backward.indexOf('for (var j = 0u; j < taps; j++)', pieceLoop);
  const tapLoop = backward.indexOf('for (var i = 0u; i < taps; i++)', rowLoop);
  const accumulation = backward.indexOf('dq1 += g.dq1;', tapLoop);
  const atomic = backward.indexOf('add_piece_grad(o, dq1.x);', accumulation);
  assert.ok(start >= 0 && end > start);
  assert.ok(boxPieceLoop >= 0 && boxAtomic > boxPieceLoop && boxAtomic < multi);
  assert.ok(pieceLoop > multi && rowLoop > pieceLoop && tapLoop > rowLoop);
  assert.ok(accumulation > tapLoop && atomic > accumulation);
  // No atomic inside the tap loops: one fixed-point add per piece coordinate.
  assert.doesNotMatch(backward.slice(rowLoop, accumulation), /add_piece_grad/);
});

test('tile count uses the same f32 settings as the GPU', () => {
  // A bound exactly on pixel 15 stays outward-conservative across tile 16.
  assert.equal(countTileEntries(
    shapeData([{ bbox: [15, 0, 15, 0] }]),
    { width: 32, height: 1, s: [1, 1], scale: 1, origin: [0, 0] },
  ), 2);
  assert.equal(countTileEntries(
    shapeData([{ bbox: [9_999_980, 0, 9_999_980, 0] }]),
    { width: 32, height: 1, s: [2, 2], scale: 2, origin: [9_999_980.25, 0] },
  ), 1);
  const scale = Math.fround(0.01);
  const filter = Math.fround(0.1);
  assert.equal(countTileEntries(
    shapeData([{ bbox: [999_900.5, 0, 999_900.5, 0] }]),
    { width: 32, height: 1, s: [filter, filter], scale, origin: [999_900.125, 0] },
  ), 1);
  assert.throws(() => countTileEntries(
    shapeData([{ bbox: [0, 0, 0, 0] }]),
    { ...settings, scale: Number.MIN_VALUE },
  ), /positive and finite after f32 conversion/);
});

test('tile counting rejects malformed scenes and u32 prefix overflow', () => {
  assert.throws(() => countTileEntries(new Float32Array(15), settings), /16 values/);
  assert.throws(() => countTileEntries(
    shapeData([{ bbox: [2, 0, 1, 1] }]), settings,
  ), /inverted bbox/);
  assert.throws(() => countTileEntries(
    shapeData([{ bbox: [0, 0, 1, 1], s: [1, 0] }]), settings,
  ), /filter must be positive/);

  const dimension = 1_048_592; // 65,537 tiles per axis
  assert.throws(() => countTileEntries(
    shapeData([{ bbox: [0, 0, dimension - 1, dimension - 1] }]),
    { width: dimension, height: dimension, s: [1, 1] },
  ), /u32 prefix offsets would overflow/);
});

test('tile capacity grows geometrically within the device binding limit', () => {
  assert.equal(growTileEntryCapacity(1, 0, 4), 1);
  assert.equal(growTileEntryCapacity(1, 1025, 1 << 20), 2048);
  assert.equal(growTileEntryCapacity(4096, 4000, 1 << 20), 4096);
  assert.equal(growTileEntryCapacity(2048, 3000, 12_000), 3000);
  assert.throws(() => growTileEntryCapacity(1, 4, 12), /scene needs 16 compact tile bytes/);
  assert.throws(() => growTileEntryCapacity(1, 0, 3), /too small/);
});

test('sort capacity follows the device and the scene', () => {
  // Above its capacity a tile falls back to a serial comb sort, so capacity
  // decides whether a dense tile costs microseconds or tens of milliseconds.
  assert.equal(sortCapacity({ maxComputeWorkgroupStorageSize: 16384 }, null), 2048);
  assert.equal(sortCapacity({ maxComputeWorkgroupStorageSize: 49152 }, null), 8192);
  assert.equal(sortCapacity({}, null), 2048); // WebGPU's guaranteed minimum
  assert.equal(sortCapacity({ maxComputeWorkgroupStorageSize: 1 << 20 }, null), 16384);
  assert.equal(sortCapacity({ maxComputeWorkgroupStorageSize: 16384 }, '4096'), 4096);
  assert.throws(() => sortCapacity({}, '3000'), /power of two/);

  // A sparse scene keeps the small specialization: large workgroup storage
  // costs occupancy, which is measurable when tiles are many and short.
  assert.equal(sortCapacityFor(18, 8192), 2048);
  assert.equal(sortCapacityFor(900, 8192), 4096);
  assert.equal(sortCapacityFor(6256, 8192), 8192);
  assert.equal(sortCapacityFor(6256, 2048), 2048); // never exceeds the device
});

test('renderer validates pixel arrays', async () => {
  const renderer = new Renderer(
    { limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20 } },
    { width: 2, height: 3, maxPieces: 0, maxShapes: 0, maxCurves: 0 },
  );
  assert.throws(() => renderer.uploadTarget(new Float32Array(23)), /length 24/);
  assert.throws(() => renderer.uploadTarget(new Float64Array(24)), /Float32Array/);
  await assert.rejects(renderer.backward(new Float32Array(23)), /length 24/);
  await assert.rejects(renderer.backward(new Float64Array(24)), /Float32Array/);
});

test('renderer exposes compact and former dense culling sizes', () => {
  const renderer = new Renderer(
    { limits: { maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 29 } },
    { width: 4096, height: 2304, maxPieces: 0, maxShapes: 50_000, maxCurves: 0 },
  );
  assert.deepEqual(renderer.getCullingInfo(), {
    tileSize: 16,
    tiles: [256, 144],
    entries: 0,
    peakEntries: 0,
    capacity: 1,
    bytes: 0,
    peakBytes: 0,
    capacityBytes: 4,
    denseEntries: 1_843_200_000,
    denseBytes: 7_372_800_000,
  });
});

test('renderer grows compact storage and refreshes every dependent bind group', async () => {
  const previousUsage = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = {
    STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8,
  };
  const buffers = [];
  const bindGroups = [];
  const device = {
    limits: { maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 29 },
    queue: { writeBuffer() {} },
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createComputePipeline: ({ compute }) => ({
      entryPoint: compute.entryPoint,
      getBindGroupLayout: () => ({}),
    }),
    createBuffer: ({ size, usage }) => {
      const buffer = { size, usage, destroyed: false, destroy() { this.destroyed = true; } };
      buffers.push(buffer);
      return buffer;
    },
    createBindGroup: ({ entries }) => {
      const group = { entries };
      bindGroups.push(group);
      return group;
    },
  };

  let renderer;
  try {
    renderer = await Renderer.create(device, {
      width: 512, height: 512, maxPieces: 0, maxShapes: 2, maxCurves: 0,
    });
    const scene = (count) => ({
      pieceData: new Float32Array(),
      pieceCount: 0,
      shapeData: shapeData(Array.from({ length: count }, () => ({ bbox: [0, 0, 511, 511] }))),
      curveMetaData: new Float32Array(),
      curveCount: 0,
    });
    renderer.uploadScene(scene(1), settings);
    assert.equal(renderer.getCullingInfo().capacity, 1024);
    renderer.uploadScene(scene(2), settings);
    assert.equal(renderer.getCullingInfo().capacity, 2048);
    const latest = renderer.tileShapesBuf;
    // Forward, backward and bin-fill are rebound eagerly. The sort bind group is
    // specialized per capacity and built on demand, so growth invalidates its
    // cache instead -- either way nothing may keep the retired buffer.
    const refreshed = bindGroups.slice(-3).filter(({ entries }) =>
      entries.some(({ binding }) => binding === 11));
    assert.equal(refreshed.length, 3);
    assert(refreshed.every(({ entries }) =>
      entries.find(({ binding }) => binding === 11).resource.buffer === latest));
    assert.equal(renderer._sortBinds.size, 0, 'sort bind cache must be invalidated');
  } finally {
    renderer?.destroy();
    if (previousUsage === undefined) delete globalThis.GPUBufferUsage;
    else globalThis.GPUBufferUsage = previousUsage;
  }
  assert(buffers.every(({ destroyed }) => destroyed));
});

test('kernels specialize the filter pipelines once each and rebind on switch', async () => {
  const previousUsage = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = {
    STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8,
  };
  const pipelines = [];
  const bindGroups = [];
  const device = {
    limits: { maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 29 },
    queue: { writeBuffer() {} },
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createComputePipeline: ({ compute }) => {
      const layout = {};
      const pipe = {
        entryPoint: compute.entryPoint,
        constants: compute.constants ?? {},
        layout,
        getBindGroupLayout: () => layout,
      };
      pipelines.push(pipe);
      return pipe;
    },
    createBuffer: ({ size }) => ({ size, destroy() {} }),
    createBindGroup: (descriptor) => {
      bindGroups.push(descriptor);
      return descriptor;
    },
  };
  const scene = {
    pieceData: new Float32Array(),
    pieceCount: 0,
    shapeData: shapeData([{ bbox: [0, 0, 7, 7] }]),
    curveMetaData: new Float32Array(),
    curveCount: 0,
  };
  const filtered = () => pipelines.filter(({ constants }) => 'FILTER_KERNEL' in constants);
  const FILTERED = ['forward', 'backward', 'bin_count', 'bin_fill'];
  try {
    const renderer = await Renderer.create(device, {
      width: 8, height: 8, maxPieces: 0, maxShapes: 1, maxCurves: 0,
    });
    assert.deepEqual(filtered().map(({ entryPoint }) => entryPoint).sort(), [...FILTERED].sort());
    assert(filtered().every(({ constants }) => constants.FILTER_KERNEL === KERNELS.box.code));
    assert.equal(renderer.fwdPipe.constants.BLEND_MODE, 0);

    renderer.uploadScene(scene, { ...settings, width: 8, height: 8, kernel: 'tent' });
    assert.equal(filtered().length, 8);
    assert(filtered().slice(-4).every(({ constants }) => constants.FILTER_KERNEL === KERNELS.tent.code));
    assert.equal(renderer.fwdPipe.constants.FILTER_KERNEL, KERNELS.tent.code);
    assert.equal(renderer.bwdPipe.constants.TRAIN_GEOMETRY, 1);
    for (const [pipe, bind] of [
      ['fwdPipe', 'fwdBind'], ['bwdPipe', 'bwdBind'],
      ['binCountPipe', 'binCountBind'], ['binFillPipe', 'binFillBind'],
    ]) assert.equal(renderer[bind].layout, renderer[pipe].layout, bind);

    renderer.uploadTarget(new Float32Array(8 * 8 * 4));
    const fused = () => pipelines.filter(({ entryPoint }) => entryPoint === 'forward_l2');
    assert.equal(fused().length, 1);
    assert.equal(renderer.fwdL2Pipe.constants.FILTER_KERNEL, KERNELS.tent.code);
    assert.equal(renderer.fwdL2Bind.layout, renderer.fwdL2Pipe.layout);

    renderer.uploadScene(scene, { ...settings, width: 8, height: 8 });
    assert.equal(filtered().length, 10);
    assert.equal(renderer.fwdPipe.constants.FILTER_KERNEL, KERNELS.box.code);
    assert.equal(renderer.fwdL2Pipe.constants.FILTER_KERNEL, KERNELS.box.code);
    assert.equal(renderer.fwdL2Bind.layout, renderer.fwdL2Pipe.layout);
    renderer.uploadScene(scene, { ...settings, width: 8, height: 8, kernel: 'tent' });
    assert.equal(filtered().length, 10);
    assert.throws(
      () => renderer.uploadScene(scene, { ...settings, kernel: 'gauss' }),
      /unknown filter kernel/,
    );
    renderer.destroy();
  } finally {
    if (previousUsage === undefined) delete globalThis.GPUBufferUsage;
    else globalThis.GPUBufferUsage = previousUsage;
  }
});

test('blend modes validate, specialize pipelines, and skip the painter sort', async () => {
  assert.throws(() => new Renderer(
    { limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20 } },
    { width: 2, height: 2, maxPieces: 0, maxShapes: 0, maxCurves: 0, blend: 'overlay' },
  ), /blend must be one of src-over, add, multiply, screen/);
  assert.throws(() => new Renderer(
    { limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20 } },
    {
      width: 2,
      height: 2,
      maxPieces: 0,
      maxShapes: 0,
      maxCurves: 0,
      blend: 'multiply',
      outputAlpha: true,
    },
  ), /outputAlpha is only supported with src-over/);

  const previousUsage = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8 };
  const run = async (blend) => {
    const pipelines = [];
    const dispatched = [];
    const device = {
      limits: { maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 29 },
      queue: { writeBuffer() {}, submit() {} },
      createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
      createComputePipeline: ({ compute }) => {
        const pipe = {
          entryPoint: compute.entryPoint,
          constants: compute.constants ?? null,
          getBindGroupLayout: () => ({}),
        };
        pipelines.push(pipe);
        return pipe;
      },
      createBuffer: ({ size, usage }) => ({ size, usage, destroy() {} }),
      createBindGroup: ({ entries }) => ({ entries }),
      createCommandEncoder: () => ({
        clearBuffer() {},
        beginComputePass: () => ({
          setPipeline: (pipe) => dispatched.push(pipe.entryPoint),
          setBindGroup() {},
          dispatchWorkgroups() {},
          end() {},
        }),
        finish: () => ({}),
      }),
    };
    const renderer = await Renderer.create(device, {
      width: 64, height: 64, maxPieces: 0, maxShapes: 1, maxCurves: 0, blend,
    });
    renderer.uploadScene({
      pieceData: new Float32Array(),
      pieceCount: 0,
      shapeData: shapeData([{ bbox: [0, 0, 63, 63] }]),
      curveMetaData: new Float32Array(),
      curveCount: 0,
    }, settings);
    renderer.forwardNoRead();
    renderer.destroy();
    return { pipelines, dispatched };
  };

  try {
    for (const [blend, { code, sorted }] of Object.entries(BLEND_MODES)) {
      const { pipelines, dispatched } = await run(blend === 'src-over' ? undefined : blend);
      // Forward, backward, and the fused L2 forward all carry the blend; the
      // backward specialization keeps its train flags alongside it.
      const forward = pipelines.find((pipe) => pipe.entryPoint === 'forward');
      const backward = pipelines.find((pipe) => pipe.entryPoint === 'backward');
      assert.equal(forward.constants.BLEND_MODE, code, blend);
      assert.equal(forward.constants.OUTPUT_ALPHA, 0, blend);
      assert.equal(backward.constants.BLEND_MODE, code, blend);
      assert.equal(backward.constants.TRAIN_GEOMETRY, 1, blend);
      // Order-independent blends never build or dispatch the painter sort;
      // src-over keeps its exact ascending sort.
      const sortPipes = pipelines.filter((pipe) => pipe.entryPoint === 'bin_sort');
      assert.equal(sortPipes.length > 0, sorted, blend);
      assert.equal(dispatched.includes('bin_sort'), sorted, blend);
      assert(dispatched.includes('bin_fill'), blend);
      assert.equal(dispatched.at(-1), 'forward', blend);
    }
  } finally {
    if (previousUsage === undefined) delete globalThis.GPUBufferUsage;
    else globalThis.GPUBufferUsage = previousUsage;
  }
});

test('blur training widens the gradient stride only when enabled', async () => {
  const previousUsage = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8 };
  const run = async (train) => {
    const pipelines = [];
    const buffers = [];
    const device = {
      limits: { maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 29 },
      queue: { writeBuffer() {} },
      createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
      createComputePipeline: ({ compute }) => {
        const pipe = {
          entryPoint: compute.entryPoint,
          constants: compute.constants ?? null,
          getBindGroupLayout: () => ({}),
        };
        pipelines.push(pipe);
        return pipe;
      },
      createBuffer: ({ size, usage }) => {
        const buffer = { size, usage, destroy() {} };
        buffers.push(buffer);
        return buffer;
      },
      createBindGroup: ({ entries }) => ({ entries }),
    };
    const renderer = await Renderer.create(device, {
      width: 32, height: 32, maxPieces: 4, maxShapes: 10, maxCurves: 4, train,
    });
    const result = {
      stride: renderer.gradStride,
      frozen: renderer.frozen,
      backward: pipelines.find((pipe) => pipe.entryPoint === 'backward').constants,
      reduce: pipelines.find((pipe) => pipe.entryPoint === 'reduce_slots').constants,
      shapeGradsBytes: renderer.shapeGradsBuf.size,
    };
    renderer.destroy();
    return result;
  };

  try {
    const plain = await run(undefined);
    assert.equal(plain.stride, 4);
    assert.deepEqual(plain.frozen, []); // blur off is the baseline, not a freeze
    assert.equal(plain.backward.TRAIN_BLUR, 0);
    assert.equal(plain.backward.SHAPE_GRAD_STRIDE, 4);
    assert.equal(plain.reduce.SHAPE_GRAD_STRIDE, 4);
    // Two words (hi, lo) per accumulator.
    assert.equal(plain.shapeGradsBytes, 2 * 16 * 10 * 4 * 4);

    const blurred = await run({ blur: true });
    assert.equal(blurred.stride, 6);
    assert.equal(blurred.backward.TRAIN_BLUR, 1);
    assert.equal(blurred.backward.SHAPE_GRAD_STRIDE, 6);
    assert.equal(blurred.reduce.SHAPE_GRAD_STRIDE, 6);
    assert.equal(blurred.shapeGradsBytes, 2 * 16 * 10 * 6 * 4);
  } finally {
    if (previousUsage === undefined) delete globalThis.GPUBufferUsage;
    else globalThis.GPUBufferUsage = previousUsage;
  }
});

test('color range and tonemap domain are validated at construction', () => {
  const device = { limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20 } };
  const base = { width: 2, height: 2, maxPieces: 0, maxShapes: 0, maxCurves: 0 };
  // multiply/screen composite factors in [0,1]; only unit-range codecs fit.
  assert.throws(() => new Renderer(device, { ...base, blend: 'multiply', colorRange: 'nonneg' }),
    /colorRange must be 'unit'/);
  assert.throws(() => new Renderer(device, { ...base, blend: 'screen', colorRange: 'signed' }),
    /colorRange must be 'unit'/);
  // The Reinhard pair has a pole at x = -1/k; signed light needs 'smooth'.
  assert.throws(() => new Renderer(device, { ...base, tonemap: 'reinhard', colorRange: 'signed' }),
    /pole at negative light/);
  assert.throws(() =>
    new Renderer(device, { ...base, tonemap: 'reinhard-white', colorRange: 'signed' }),
  /pole at negative light/);
  assert.throws(() => new Renderer(device, { ...base, colorRange: 'hdr' }),
    /colorRange must be/);
  // The valid pairings construct (destroyed without ever touching the device).
  new Renderer(device, { ...base, blend: 'add', tonemap: 'smooth', colorRange: 'signed' });
  new Renderer(device, { ...base, blend: 'add', tonemap: 'reinhard-white', colorRange: 'nonneg' });
  new Renderer(device, { ...base, blend: 'multiply', colorRange: 'unit' });
});

test('tonemap specializes only the loss kernels and keeps the image linear', async () => {
  assert.throws(() => new Renderer(
    { limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20 } },
    { width: 2, height: 2, maxPieces: 0, maxShapes: 0, maxCurves: 0, tonemap: 'aces' },
  ), /tonemap must be one of none, reinhard/);

  const previousUsage = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8 };
  const run = async (tonemap) => {
    const pipelines = [];
    let uniformBytes = 0;
    const device = {
      limits: { maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 29 },
      queue: { writeBuffer() {} },
      createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
      createComputePipeline: ({ compute }) => {
        const pipe = {
          entryPoint: compute.entryPoint,
          constants: compute.constants ?? null,
          getBindGroupLayout: () => ({}),
        };
        pipelines.push(pipe);
        return pipe;
      },
      createBuffer: ({ size, usage }) => {
        if (usage & 8) uniformBytes = size;
        return { size, usage, destroy() {} };
      },
      createBindGroup: ({ entries }) => ({ entries }),
    };
    const renderer = await Renderer.create(device, {
      width: 8, height: 8, maxPieces: 0, maxShapes: 1, maxCurves: 0, tonemap,
    });
    renderer.uploadTarget(new Float32Array(8 * 8 * 4)); // builds the L2 pipelines
    const constants = (entryPoint) =>
      pipelines.find((pipe) => pipe.entryPoint === entryPoint).constants;
    const result = {
      uniformBytes,
      forward: constants('forward'),
      backward: constants('backward'),
      l2grad: constants('l2grad'),
      fused: constants('forward_l2'),
    };
    renderer.destroy();
    return result;
  };

  try {
    const plain = await run(undefined);
    assert.equal(plain.uniformBytes, 80);
    assert.equal(plain.l2grad.TONEMAP, 0);
    assert.equal(plain.fused.TONEMAP, 0);

    const mapped = await run('reinhard');
    assert.equal(mapped.l2grad.TONEMAP, 1);
    assert.equal(mapped.fused.TONEMAP, 1);
    // The composite stays linear: forward and backward carry no TONEMAP.
    assert.equal('TONEMAP' in mapped.forward, false);
    assert.equal('TONEMAP' in mapped.backward, false);

    // The white-point operators are further codes on the same loss kernels;
    // the composite pipelines still never see them.
    const white = await run('reinhard-white');
    assert.equal(white.l2grad.TONEMAP, 2);
    assert.equal(white.fused.TONEMAP, 2);
    assert.equal('TONEMAP' in white.forward, false);
    const smooth = await run('smooth');
    assert.equal(smooth.l2grad.TONEMAP, 3);
    assert.equal(smooth.fused.TONEMAP, 3);
    assert.equal('TONEMAP' in smooth.backward, false);
  } finally {
    if (previousUsage === undefined) delete globalThis.GPUBufferUsage;
    else globalThis.GPUBufferUsage = previousUsage;
  }
});

test('order-independent blend adjoints stay free of the forward image', async () => {
  const dir = new URL('../../js/wgsl/', import.meta.url);
  const source = await readFile(new URL('render.wgsl', dir), 'utf8');
  assert.match(source, /override BLEND_MODE : u32 = 0u;/);
  // src-over is the only mode that replays against the composited image...
  assert.match(source, /if \(BLEND_MODE == BLEND_SRC_OVER\) \{\s*outC = vec3<f32>\(image\[idx\]/);
  // ...and the only mode allowed a data-dependent early exit: unsorted tile
  // lists make any transmittance cutoff order-dependent.
  assert.match(source, /BLEND_MODE == BLEND_SRC_OVER && P < 1e-5/);
  // multiply and screen reconstruct each factor's sibling product with a
  // guarded divide; screen's runs in complement space.
  assert.match(source, /U\.bg\.rgb \* M \/ max\(m, vec3<f32>\(1e-3\)\)/);
  assert.match(source, /\(1\.0 - U\.bg\.rgb\) \* M \/ max\(m, vec3<f32>\(1e-3\)\)/);
  // Blur adjoints are the Euler-identity contraction of the piece gradients
  // with their own coordinates -- no separate filter adjoint code path.
  assert.match(source, /override TRAIN_BLUR : bool = false;/);
  assert.match(source, /sxAdj \+= q1\.x \* g\.dq1\.x \+ q2\.x \* g\.dq2\.x \+ q3\.x \* g\.dq3\.x;/);
  assert.match(source, /add_shape_grad\(so \+ 4u, -sxAdj \/ sf\.x\);/);
  // The tonemap lives at the loss boundary only: l2_pixel maps to display
  // space and accumulates dL/dk; composite() and backward() never reference
  // TONEMAP, so the image buffer stays linear.
  assert.match(source, /override TONEMAP : u32 = 0u;/);
  assert.match(source, /dLdImage\[idx \+ c\] = cot \* U\.tonemapK \/ \(denom \* denom\);/);
  assert.match(source, /dk \+= cot \* x \/ \(denom \* denom\);/);
  assert.doesNotMatch(source.slice(source.indexOf('fn composite'), source.indexOf('fn store_image')), /TONEMAP/);
});

test('GPU binning retains its exact ascending painter-order sort', async () => {
  const dir = new URL('../../js/wgsl/', import.meta.url);
  const source = (await Promise.all(['scene', 'coverage', 'binning', 'render']
    .map((name) => readFile(new URL(`${name}.wgsl`, dir), 'utf8')))).join('\n');
  // The tile-offset scan is parallel (block scan, block-sum scan, add), but must
  // still leave tileOffset as the exclusive prefix sum with the total at [nt],
  // and must still reset tileCount for bin_fill's cursors.
  assert.match(source, /fn bin_scan_block[\s\S]*atomicExchange\(&tileCount\[t\], 0u\)/);
  assert.match(source, /fn bin_scan_block[\s\S]*tileOffset\[t\] = scanScratch\[lidx\] - v;/);
  assert.match(source, /fn bin_scan_blocks[\s\S]*tileOffset\[nt\] = carry;/);
  assert.match(source, /fn bin_scan_add[\s\S]*tileOffset\[gid\.x\] \+= tileBlockSum\[wgid\.x\];/);
  assert.match(source, /fn bin_fill[\s\S]*tileShapes\[pos\] = si;/);
  assert.match(source, /fn bin_sort[\s\S]*if \(a > b\)/);
  assert.match(source, /fn forward[\s\S]*let si = tileShapes\[k - 1u\]/);
  assert.match(source, /fn backward[\s\S]*let si = tileShapes\[k - 1u\]/);
});
