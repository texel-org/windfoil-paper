// Focused, deterministic renderer benchmark; no image downloads or encoding.
// Usage: node tools/perf.js --out=output/perf/run.json
// --source=output/perf/baseline selects a saved source tree for A/B runs.
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { cpus, release } from 'node:os';

const args = Object.fromEntries(process.argv.slice(2).map((arg) => {
  const [key, ...value] = arg.replace(/^--/, '').split('=');
  return [key, value.join('=') || '1'];
}));
const options = new Set(['source', 'out', 'gpu', 'cases', 'repeats', 'steps', 'style', 'k', 'blur', 'warmup', 'optimize', 'blend']);
for (const key of Object.keys(args)) if (!options.has(key)) throw new Error(`unknown option --${key}`);
const integer = (name, fallback, minimum = 1) => {
  const value = Number(args[name] ?? fallback);
  if (!Number.isInteger(value) || value < minimum) throw new Error(`${name} must be an integer >= ${minimum}`);
  return value;
};
const steps = integer('steps', 30), repeats = integer('repeats', 2);
const warmup = integer('warmup', 5, 0);
const k = integer('k', 4, 3), blur = Number(args.blur ?? 1);
if (!(blur > 0) || !Number.isFinite(blur)) throw new Error('blur must be positive and finite');
if (args.style && !['anchor', 'raw'].includes(args.style)) throw new Error('style must be anchor or raw');
const cases = (args.cases ?? '128:512:0.04,512:512:0.04,512:4096:0.04,512:512:0.3')
  .split(',').map((cell) => cell.split(':').map(Number));
if (cases.some(([size, n, radius], i) => cases[i].length !== 3 || !Number.isInteger(size) || size < 1 ||
    !Number.isInteger(n) || n < 1 || !Number.isFinite(radius) || radius <= 0)) {
  throw new Error('cases must contain positive size:n:radius triples');
}
const source = resolve(args.source ?? '.');
const load = (name) => import(pathToFileURL(`${source}/js/${name}.js`));
const started = performance.now();
const { Renderer, requestDevice, getWebGPUHostInfo } = await load('renderer');
const { LoopModel } = await load('loop-model');
const { rawStyle } = await load('raw-style');
const { anchorStyle } = await load('color-anchors');
const { Adam } = await load('adam');
const imported = performance.now();

const measure = async (fn) => {
  const t = performance.now();
  const value = await fn();
  return { ms: performance.now() - t, value };
};
const summary = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  return { median: sorted[Math.floor(sorted.length / 2)],
    p90: sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.9))],
    min: sorted[0], max: sorted.at(-1) };
};

function modelFor(size, n, radius) {
  let seed = 7;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
  const ax = new Float32Array(n * k), ay = new Float32Array(n * k);
  const cx = new Float32Array(n * k), cy = new Float32Array(n * k);
  const color = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const x = random() * size, y = random() * size;
    const r = size * radius * (0.5 + random());
    for (let j = 0; j < k; j++) {
      const a = j * 2 * Math.PI / k;
      ax[i * k + j] = x + Math.cos(a) * r;
      ay[i * k + j] = y + Math.sin(a) * r;
      cx[i * k + j] = x + Math.cos(a + Math.PI / k) * r * 1.4;
      cy[i * k + j] = y + Math.sin(a + Math.PI / k) * r * 1.4;
    }
    for (let c = 0; c < 3; c++) color[i * 3 + c] = (random() - 0.5) * 3;
  }
  const style = args.style === 'anchor' ? anchorStyle()
    : rawStyle({ channels: 'rgb', transfer: 'sigmoid', alpha: 0.9 });
  const styleParams = args.style === 'anchor'
    ? style.init(Array.from({ length: n }, (_, i) => Array.from(color.subarray(3 * i, 3 * i + 3),
      (v) => 1 / (1 + Math.exp(-v)))), new Float32Array(n).fill(0.9))
    : { rawColor: color };
  return new LoopModel({ ax, ay, cx, cy, n, k, style, styleParams });
}

// Capture host compilation costs without changing the dispatch path.
function instrument(device, records, capture) {
  return new Proxy(device, { get(target, name) {
    if (name === 'createComputePipeline' || name === 'createShaderModule') {
      return (descriptor) => {
        const t = performance.now();
        const result = target[name](descriptor);
        records.push({ operation: name, entry: descriptor.compute?.entryPoint,
          constants: descriptor.compute?.constants, ms: performance.now() - t });
        if (descriptor.compute) capture.names.set(result, descriptor.compute.entryPoint);
        return result;
      };
    }
    if (name === 'createCommandEncoder' && capture.commands) {
      return (...args) => {
        const encoder = target.createCommandEncoder(...args);
        return new Proxy(encoder, { get(enc, method) {
          if (method === 'clearBuffer') return (...values) => {
            capture.commands.push({ clear: values }); return enc.clearBuffer(...values);
          };
          if (method === 'beginComputePass') return (...values) => {
            const pass = enc.beginComputePass(...values);
            let pipeline, bind;
            return new Proxy(pass, { get(p, action) {
              if (action === 'setPipeline') return (value) => { pipeline = value; p.setPipeline(value); };
              if (action === 'setBindGroup') return (index, value) => { bind = value; p.setBindGroup(index, value); };
              if (action === 'dispatchWorkgroups') return (...grid) => {
                capture.commands.push({ pipeline, bind, grid }); p.dispatchWorkgroups(...grid);
              };
              const value = Reflect.get(p, action, p);
              return typeof value === 'function' ? value.bind(p) : value;
            } });
          };
          const value = Reflect.get(enc, method, enc);
          return typeof value === 'function' ? value.bind(enc) : value;
        } });
      };
    }
    const value = Reflect.get(target, name, target);
    return typeof value === 'function' ? value.bind(target) : value;
  } });
}

// Diagnostic replay brackets each dispatch with GPU timestamps. Splitting the
// compute passes changes scheduling, so these explain bottlenecks; the normal
// step's wall time above remains the performance comparison.
async function profile(device, renderer, capture) {
  if (!device.features.has('timestamp-query')) return null;
  capture.commands = [];
  await renderer.stepGpuLoss();
  const commands = capture.commands;
  capture.commands = null;
  const dispatches = commands.filter((c) => c.pipeline);
  const count = dispatches.length * 2;
  const queries = device.createQuerySet({ type: 'timestamp', count });
  const resolved = device.createBuffer({ size: count * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const staging = device.createBuffer({ size: count * 8,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    const samples = [];
    for (let repeat = 0; repeat < 5; repeat++) {
      const enc = device.createCommandEncoder();
      let at = 0;
      for (const command of commands) {
        if (command.clear) { enc.clearBuffer(...command.clear); continue; }
        const pass = enc.beginComputePass({ timestampWrites: { querySet: queries,
          beginningOfPassWriteIndex: at++, endOfPassWriteIndex: at++ } });
        pass.setPipeline(command.pipeline); pass.setBindGroup(0, command.bind);
        pass.dispatchWorkgroups(...command.grid); pass.end();
      }
      enc.resolveQuerySet(queries, 0, count, resolved, 0);
      enc.copyBufferToBuffer(resolved, 0, staging, 0, count * 8);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const times = new BigUint64Array(staging.getMappedRange());
      samples.push(dispatches.map((_, i) => Number(times[2 * i + 1] - times[2 * i]) / 1e6));
      staging.unmap();
    }
    return dispatches.map((c, i) => ({ entry: capture.names.get(c.pipeline),
      ms: summary(samples.map((s) => s[i])) }));
  } finally { queries.destroy(); resolved.destroy(); staging.destroy(); }
}

async function run() {
  const requested = await measure(() => requestDevice({ timestampQuery: args.gpu === '1' }));
  const nativeDevice = requested.value;
  nativeDevice.pushErrorScope('validation');
  const records = [];
  const capture = { names: new Map(), commands: null };
  const device = instrument(nativeDevice, records, capture);
  const result = {
    date: new Date().toISOString(), source, node: process.version,
    os: `${process.platform} ${release()}`, cpu: cpus()[0]?.model,
    host: getWebGPUHostInfo(), options: args, importMs: imported - started,
    deviceMs: requested.ms, beforeDeviceMs: process.uptime() * 1000 - requested.ms,
    cases: [],
  };
  try {
    for (const [size, n, radius] of cases) {
      for (let repeat = 0; repeat < repeats; repeat++) {
        const model = modelFor(size, n, radius);
        const lrs = Object.fromEntries(Object.keys(model.params).map((key) => [key, { lr: 0.01 }]));
        const adam = new Adam(model.params, lrs);
        const target = new Float32Array(size * size * 4).fill(0.3);
        const settings = { s: [blur, blur], bg: [1, 1, 1] };
        const compilationStart = records.length;
        const init = await measure(() => Renderer.create(device, {
          width: size, height: size, maxShapes: n, maxPieces: model.maxPieces,
          maxCurves: model.maxCurves, train: { alpha: model.style.trainsAlpha }, blend: args.blend,
        }));
        const r = init.value;
        try {
          const scene = model.decode().scene;
          const upload = await measure(() => { r.uploadScene(scene, settings); r.uploadTarget(target); });
          const first = await measure(() => r.stepGpuLoss());
          for (let i = 0; i < warmup; i++) await r.stepGpuLoss();
          const samples = { decode: [], upload: [], gpuAndReadback: [], pullback: [], adam: [], total: [] };
          let loss;
          for (let i = 0; i < steps; i++) {
            const t0 = performance.now();
            const decoded = model.decode();
            const t1 = performance.now();
            r.uploadScene(decoded.scene, settings);
            const t2 = performance.now();
            const grad = await r.stepGpuLoss();
            const t3 = performance.now();
            const grads = model.pullback(grad);
            const t4 = performance.now();
            if (args.optimize === '1') adam.step(model.params, grads);
            const t5 = performance.now();
            samples.decode.push(t1 - t0); samples.upload.push(t2 - t1);
            samples.gpuAndReadback.push(t3 - t2); samples.pullback.push(t4 - t3);
            samples.adam.push(t5 - t4); samples.total.push(t5 - t0);
            loss = grad.loss;
          }
          const cell = { size, n, radius, k: model.k, repeat, initMs: init.ms,
            uploadMs: upload.ms, firstStepMs: first.ms,
            readyMs: init.ms + upload.ms + first.ms,
            timings: Object.fromEntries(Object.entries(samples).map(([key, values]) => [key, summary(values)])),
            culling: r.getCullingInfo(), loss,
            compilation: records.slice(compilationStart) };
          if (args.gpu === '1') cell.gpuStages = await profile(nativeDevice, r, capture);
          result.cases.push(cell);
          console.log(JSON.stringify({ size, n, radius, repeat, readyMs: cell.readyMs,
            totalMs: cell.timings.total.median, gpuMs: cell.timings.gpuAndReadback.median }));
        } finally { r.destroy(); }
      }
    }
  } finally {
    const error = await nativeDevice.popErrorScope();
    nativeDevice.destroy();
    if (error) throw new Error(`GPU validation failed: ${error.message}`);
  }
  const file = resolve(args.out ?? 'output/perf/latest.json');
  await mkdir(dirname(file), { recursive: true });
  await writeFile(file, JSON.stringify(result, null, 2) + '\n');
  console.log(`wrote ${file}`);
}

try { await run(); process.exit(0); }
catch (error) { console.error(error); process.exit(1); }
