// Shared WebGPU host: tiled forward render, analytic VJP, and fused L2.

import { TONEMAP_DOMAINS, TONEMAP_HAS_WHITE, TONEMAP_MODES } from './tonemap.js';

const WGSL_SOURCES = ['scene', 'coverage', 'binning', 'render']
  .map((name) => new URL(`./wgsl/${name}.wgsl`, import.meta.url));

let dawnHost = null;
let dawnLoad = null;
let hostInfo = {
  environment: 'uninitialized',
  backend: 'uninitialized',
  adapterInfo: null,
  limits: null,
};

const RELEVANT_LIMITS = [
  'maxTextureDimension2D',
  'maxBufferSize',
  'maxStorageBufferBindingSize',
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupsPerDimension',
];

function readLimits(limits) {
  return Object.fromEntries(RELEVANT_LIMITS.flatMap((name) =>
    limits?.[name] === undefined ? [] : [[name, Number(limits[name])]]));
}

const isRealDeno = () => typeof Deno !== 'undefined' && !!Deno.version?.deno;
const isNode = () => typeof process !== 'undefined' && !!process.versions?.node && !isRealDeno();

async function dawnSpecifier() {
  if (isRealDeno()) return ['npm', 'webgpu@0.4.0'].join(':');
  if (process.platform === 'darwin') {
    const osModule = ['node', 'os'].join(':');
    const { release } = await import(/* @vite-ignore */ osModule);
    if (Number.parseInt(release(), 10) < 24) return ['webgpu', 'legacy'].join('-');
  }
  return ['web', 'gpu'].join('');
}

async function loadDawn() {
  dawnLoad ??= (async () => {
    // Opaque to Vite: browsers never load the native Dawn addon.
    const specifier = await dawnSpecifier();
    const { create, globals } = await import(/* @vite-ignore */ specifier);
    Object.assign(globalThis, globals);
    dawnHost ??= create([]);
    return dawnHost;
  })();
  return dawnLoad;
}

function denoBackendMode() {
  if (!isRealDeno()) return null;
  let mode = 'auto';
  try {
    mode = Deno.env.get('WF_WEBGPU_BACKEND') ?? 'auto';
  } catch (error) {
    // Keep built-in WebGPU usable without --allow-env.
    if (error?.name !== 'NotCapable' && error?.name !== 'PermissionDenied') throw error;
  }
  if (mode !== 'auto' && mode !== 'dawn' && mode !== 'wgpu') {
    throw new Error(`invalid WF_WEBGPU_BACKEND=${JSON.stringify(mode)} (expected auto, dawn, or wgpu)`);
  }
  return mode;
}

async function requestAdapter() {
  const mode = denoBackendMode();
  const opts = { powerPreference: 'high-performance' };

  if (isNode()) {
    const gpu = await loadDawn();
    return { adapter: await gpu.requestAdapter(opts), environment: 'node', backend: 'dawn' };
  }

  // Dawn avoids built-in wgpu's high mapAsync latency; built-in remains the fallback.
  if (mode && mode !== 'wgpu') {
    try {
      const gpu = await loadDawn();
      const adapter = await gpu.requestAdapter(opts);
      if (!adapter) throw new Error('Dawn returned no WebGPU adapter');
      return { adapter, environment: 'deno-dawn', backend: 'dawn' };
    } catch (error) {
      if (mode === 'dawn') {
        const detail = String(error?.message ?? error);
        if (
          Deno.build.os === 'darwin' &&
          /MTLLogStateDescriptor|built for macOS 15/i.test(detail)
        ) {
          throw new Error(
            'The published Deno+Dawn addon requires macOS 15 or newer. ' +
              'Use WF_RUNTIME=deno or the default Node runtime on this Mac.',
            { cause: error },
          );
        }
        throw new Error(
          'WF_WEBGPU_BACKEND=dawn requested, but the webgpu native addon could not initialize ' +
            '(install gpu dependencies and grant --allow-ffi, or use -A)',
          { cause: error },
        );
      }
      console.warn(`Deno+Dawn unavailable (${error?.message ?? error}); falling back to built-in wgpu`);
    }
  }

  const gpu = globalThis.navigator?.gpu;
  const adapter = await gpu?.requestAdapter(opts);
  return mode === 'wgpu' || mode === 'auto'
    ? { adapter, environment: 'deno', backend: 'wgpu' }
    : { adapter, environment: 'web', backend: 'webgpu' };
}

/** Last successfully selected host, for benchmark/status output. */
export function getWebGPUHostInfo() {
  return hostInfo;
}

export async function requestDevice() {
  const { adapter, environment, backend } = await requestAdapter();
  if (!adapter) {
    throw new Error('No WebGPU adapter (Deno: run with --unstable-webgpu on a GPU host).');
  }
  // Large scenes need the adapter's available buffer limits.
  const requiredLimits = {};
  for (const name of ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxComputeWorkgroupStorageSize']) {
    if (adapter.limits?.[name] !== undefined) requiredLimits[name] = adapter.limits[name];
  }
  const device = await adapter.requestDevice({ requiredLimits });
  device.addEventListener?.('uncapturederror', (e) => {
    console.error('WebGPU error:', e.error?.message);
  });
  hostInfo = {
    environment,
    backend,
    adapterInfo: adapter.info ?? null,
    limits: readLimits(device.limits ?? adapter.limits),
  };
  return device;
}

async function readSource(url) {
  // Host-agnostic WGSL read: Deno, Node (fs — fetch() can't do file://), else browser fetch.
  if (typeof Deno !== 'undefined') return Deno.readTextFile(url);
  if (typeof process !== 'undefined' && process.versions?.node) {
    const fsModule = ['node:fs', 'promises'].join('/');
    const urlModule = ['node', 'url'].join(':');
    const [{ readFile }, { fileURLToPath }] = await Promise.all([
      import(/* @vite-ignore */ fsModule),
      import(/* @vite-ignore */ urlModule),
    ]);
    return readFile(fileURLToPath(url), 'utf8');
  }
  return fetch(url).then((r) => r.text());
}

/** Concatenate the WGSL modules into one shader and fail on compile errors. */
async function createShaderModule(device, urls) {
  const parts = await Promise.all(urls.map((url) => readSource(url)));
  const module = device.createShaderModule({ code: parts.join('\n') });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) {
    throw new Error('WGSL compile errors:\n' + errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
  }
  return module;
}

// Host-agnostic env read: Deno needs permission, browsers have no process.
function readEnv(name) {
  try {
    if (typeof Deno !== 'undefined') return Deno.env.get(name) ?? null;
    if (typeof process !== 'undefined') return process.env?.[name] ?? null;
  } catch {
    // Missing --allow-env is not an error; fall through to the default.
  }
  return null;
}

const SLOTS = 16; // strided copies for shape-gradient accumulation (reduced on GPU by reduce_slots)
const TILE = 16;  // pixels per tile side; MUST match TILE in scene.wgsl
const SCAN_WG = 256; // tile-offset scan workgroup; MUST match SCAN_WG in binning.wgsl
const SORT_CAPACITY_MIN = 2048;   // fits WebGPU's guaranteed 16 KiB of workgroup storage
const SORT_CAPACITY_MAX = 16384;  // beyond this the sort is no longer the bottleneck

// Entries bin_sort can hold in workgroup memory. Above it a tile falls back to a
// serial comb sort, which is ~100x slower, so use whatever the device allows.
// Power of two: the bitonic network sorts a power-of-two span.
// Capacity the workload wants: enough headroom over the mean tile list that
// realistic tiles stay off the serial fallback, without allocating workgroup
// storage a sparse scene will never use -- large scratch costs occupancy, which
// is measurable when tiles are small and numerous.
export function sortCapacityFor(meanEntriesPerTile, deviceCapacity) {
  let wanted = SORT_CAPACITY_MIN;
  const target = 4 * Math.max(0, meanEntriesPerTile);
  while (wanted < target && wanted < deviceCapacity) wanted *= 2;
  return Math.min(wanted, deviceCapacity);
}

export function sortCapacity(limits, override = readEnv('WF_SORT_CAPACITY')) {
  // The override exists to exercise the serial fallback, which is otherwise
  // unreachable on a device with generous workgroup storage.
  if (override) {
    const wanted = Number(override);
    if (!Number.isInteger(wanted) || wanted < 2 || (wanted & (wanted - 1)) !== 0) {
      throw new Error('WF_SORT_CAPACITY must be a power of two >= 2');
    }
    return wanted;
  }
  const bytes = Number(limits?.maxComputeWorkgroupStorageSize ?? 16384);
  const usable = Math.max(0, bytes - 64); // sortInfo and alignment slack
  let capacity = SORT_CAPACITY_MIN;
  while (capacity * 2 * 4 <= usable && capacity * 2 <= SORT_CAPACITY_MAX) capacity *= 2;
  return capacity;
}
const UINT32_MAX = 0xffffffff;

// Scene-wide blend modes. code MUST match the BLEND_* constants in
// render.wgsl. sorted marks compositing as painter-order dependent;
// order-independent modes skip bin_sort entirely (dense tiles never touch the
// serial fallback) and their backward never reads the forward image. Future
// modes (weighted OIT, log-sum-exp, ...) are one row here plus their
// branches in composite() and backward().
export const BLEND_MODES = Object.freeze({
  'src-over': Object.freeze({ code: 0, sorted: true }),
  'add': Object.freeze({ code: 1, sorted: false }),
  'multiply': Object.freeze({ code: 2, sorted: false }),
  'screen': Object.freeze({ code: 3, sorted: false }),
});

function positiveF32(value, name) {
  const rounded = Math.fround(value);
  if (!(rounded > 0) || !Number.isFinite(rounded)) {
    throw new Error(`${name} must be positive and finite after f32 conversion`);
  }
  return rounded;
}

function finiteF32(value, name) {
  const rounded = Math.fround(value);
  if (!Number.isFinite(rounded)) throw new Error(`${name} must be finite after f32 conversion`);
  return rounded;
}

// Conservative scalar mirror of axis_tiles in binning.wgsl. The outward
// margin covers device-level f32 differences at pixel and tile boundaries.
function tileAxisSpan(lo, hi, origin, scale, dim, margin) {
  const plo = Math.fround(Math.fround(lo - origin) / scale) - 0.5;
  const phi = Math.fround(Math.fround(hi - origin) / scale) - 0.5;
  const last = Math.fround(dim - 1);
  if (phi < -margin || plo > last + margin) return 0;
  const i0 = Math.min(dim - 1, Math.max(0, Math.floor(plo - margin)));
  const i1 = Math.min(dim - 1, Math.max(0, Math.ceil(phi + margin)));
  return Math.floor(i1 / TILE) - Math.floor(i0 / TILE) + 1;
}

/** Count the compact shape-index entries produced by bin_count. */
export function countTileEntries(shapeData, {
  width,
  height,
  s,
  scale = 1,
  origin = [0, 0],
}) {
  if (!(shapeData instanceof Float32Array) || shapeData.length % 16 !== 0) {
    throw new Error('shapeData must contain 16 values per shape');
  }
  if (!Number.isInteger(width) || width < 1 || width > UINT32_MAX ||
      !Number.isInteger(height) || height < 1 || height > UINT32_MAX) {
    throw new Error('width and height must be positive u32 integers');
  }
  if (!s || s.length !== 2) throw new Error('s must contain two filter sizes');
  if (!origin || origin.length !== 2) throw new Error('origin must contain two coordinates');
  // Count from the exact values uploaded to WGSL; otherwise f32 conversion of
  // a large origin can move a bbox on-screen after a smaller CPU count.
  const globalX = positiveF32(s[0], 's[0]');
  const globalY = positiveF32(s[1], 's[1]');
  const pixelScale = positiveF32(scale, 'scale');
  const ox = finiteF32(origin[0], 'origin[0]');
  const oy = finiteF32(origin[1], 'origin[1]');
  // Relevant projected bounds lie near the image extent, so one axis-level
  // tolerance covers several f32 ULPs without per-shape magnitude work.
  const xMargin = 1e-6 * (width + 1);
  const yMargin = 1e-6 * (height + 1);

  let entries = 0;
  for (let o = 0; o < shapeData.length; o += 16) {
    const bx0 = shapeData[o];
    const by0 = shapeData[o + 1];
    const bx1 = shapeData[o + 2];
    const by1 = shapeData[o + 3];
    if (!(bx0 <= bx1) || !(by0 <= by1)) throw new Error(`shape ${o / 16} has an inverted bbox`);
    if (!Number.isFinite(bx0 + by0 + bx1 + by1)) throw new Error(`shape ${o / 16} bbox must be finite`);
    const sx = shapeData[o + 12] > 0 ? shapeData[o + 12] : globalX;
    const sy = shapeData[o + 12] > 0 ? shapeData[o + 13] : globalY;
    if (!(sx > 0) || !(sy > 0) || !Number.isFinite(sx + sy)) {
      throw new Error(`shape ${o / 16} filter must be positive and finite`);
    }
    const hx = 0.5 * sx;
    const hy = 0.5 * sy;
    const columns = tileAxisSpan(
      Math.fround(bx0 - hx), Math.fround(bx1 + hx), ox, pixelScale, width, xMargin,
    );
    if (!columns) continue;
    const rows = tileAxisSpan(
      Math.fround(by0 - hy), Math.fround(by1 + hy), oy, pixelScale, height, yMargin,
    );
    entries += columns * rows;
  }
  if (entries > UINT32_MAX) {
    throw new Error(`scene has more than ${UINT32_MAX} tile entries; u32 prefix offsets would overflow`);
  }
  return entries;
}

/** Geometric tile-list growth bounded by WebGPU's storage-buffer limits. */
export function growTileEntryCapacity(current, required, byteLimit) {
  if (!Number.isSafeInteger(current) || current < 1 ||
      !Number.isSafeInteger(required) || required < 0) {
    throw new Error('tile capacities must be non-negative safe integers');
  }
  const limit = Math.min(UINT32_MAX, Math.floor(byteLimit / 4));
  if (!Number.isSafeInteger(limit) || limit < 1) throw new Error(`tile buffer byte limit ${byteLimit} is too small`);
  if (required > limit) {
    throw new Error(`scene needs ${required * 4} compact tile bytes; device limit is ${limit * 4}`);
  }
  if (required <= current) return current;
  let capacity = Math.min(limit, Math.max(1024, current));
  while (capacity < required) capacity = Math.min(limit, capacity * 2);
  return capacity;
}

export class Renderer {
  /** @param {GPUDevice} device @param {{width:number,height:number,maxPieces:number,maxShapes:number,maxCurves:number,train?:object,blend?:string,tonemap?:string,colorRange?:string,outputAlpha?:boolean}} o */
  constructor(device, {
    width,
    height,
    maxPieces,
    maxShapes,
    maxCurves,
    train,
    blend,
    tonemap,
    colorRange,
    outputAlpha = false,
  }) {
    // The whole scene composites with one blend mode, fixed per renderer and
    // specialized into the forward/backward pipelines like the train flags.
    this.blend = blend ?? 'src-over';
    if (!BLEND_MODES[this.blend]) {
      throw new Error(`blend must be one of ${Object.keys(BLEND_MODES).join(', ')}`);
    }
    this.outputAlpha = !!outputAlpha;
    if (this.outputAlpha && this.blend !== 'src-over') {
      throw new Error('outputAlpha is only supported with src-over blending');
    }
    // The tonemap sits at the loss boundary only: the composite (and its
    // backward) stay linear, imageBuf holds linear light, and the fused L2
    // compares in display space with the chain rule folded into dLdImage.
    // Meant for add-mode HDR scenes; 'none' compiles the loss exactly as
    // before. setExposure(k, w) sets the learnable scalars; stepGpuLoss
    // returns their gradients as kGrad (and wGrad for white-point operators).
    this.tonemap = tonemap ?? 'none';
    if (!(this.tonemap in TONEMAP_MODES)) {
      throw new Error(`tonemap must be one of ${Object.keys(TONEMAP_MODES).join(', ')}`);
    }
    this.tonemapCode = TONEMAP_MODES[this.tonemap];
    this.hasWhite = !!TONEMAP_HAS_WHITE[this.tonemap];
    // What the scene's colors may contain, declared by the color codec:
    // 'unit' ([0,1], the default), 'nonneg' ([0,inf)), or 'signed' (all of R).
    // Invalid combinations are rejected here, at construction, rather than
    // guarded at run time: multiply/screen multiply factors that must stay in
    // [0,1], and the Reinhard operators have a pole at x = -1/k.
    this.colorRange = colorRange ?? 'unit';
    if (!['unit', 'nonneg', 'signed'].includes(this.colorRange)) {
      throw new Error("colorRange must be 'unit', 'nonneg', or 'signed'");
    }
    if (this.colorRange !== 'unit' && (this.blend === 'multiply' || this.blend === 'screen')) {
      throw new Error(`blend '${this.blend}' composites factors in [0,1]; colorRange must be 'unit'`);
    }
    if (this.colorRange === 'signed' && TONEMAP_DOMAINS[this.tonemap] === 'nonneg') {
      throw new Error(
        `tonemap '${this.tonemap}' has a pole at negative light; use 'smooth' (or 'none') for signed colors`);
    }
    // Parameter groups to train. All true reproduces the previous behaviour.
    // Frozen groups are specialized out of the backward pipeline, and their
    // buffers, uploads, reductions and readbacks are skipped entirely.
    // blur (per-shape filter size) is the one group that defaults OFF: it
    // widens the shape-gradient stride from 4 to 6, so leaving it out keeps
    // the default pipelines byte-for-byte identical to before.
    const t = train ?? {};
    const flag = (value) => value === undefined ? true : !!value;
    this.train = Object.freeze({
      geometry: flag(t.geometry),
      colour: flag(t.colour ?? t.color),
      alpha: flag(t.alpha),
      blur: !!t.blur,
    });
    // frozen reports groups explicitly excluded from the default-on set;
    // blur is opt-in, so its absence is the baseline, not a freeze.
    this.frozen = Object.freeze(Object.entries(this.train)
      .filter(([name, on]) => !on && name !== 'blur').map(([name]) => name));
    if (!this.train.geometry && !this.train.colour && !this.train.alpha && !this.train.blur) {
      throw new Error('train: at least one parameter group must be enabled');
    }
    this.gradStride = this.train.blur ? 6 : 4;
    if (!Number.isInteger(width) || width < 1 || !Number.isInteger(height) || height < 1) {
      throw new Error('width and height must be positive integers');
    }
    if (!Number.isInteger(maxPieces) || maxPieces < 0 ||
        !Number.isInteger(maxShapes) || maxShapes < 0) {
      throw new Error('maxPieces and maxShapes must be non-negative integers');
    }
    if (!Number.isInteger(maxCurves) || maxCurves < 0) {
      throw new Error('maxCurves must be a non-negative integer');
    }
    this.device = device;
    this.width = width;
    this.height = height;
    this.maxPieces = maxPieces;
    this.maxShapes = maxShapes;
    this.maxCurves = maxCurves;
    this.pixels = width * height;
    this.gridX = Math.ceil(width / 8);
    this.gridY = Math.ceil(height / 8);
    this.tilesX = Math.ceil(width / TILE);
    this.tilesY = Math.ceil(height / TILE);
    this.nTiles = this.tilesX * this.tilesY;
    this._uniformData = new ArrayBuffer(80);
    this._uniformU32 = new Uint32Array(this._uniformData);
    this._uniformF32 = new Float32Array(this._uniformData);
    this._exposure = 1;
    this._white = 1;
    this.tileEntryCount = 0;
    this.peakTileEntryCount = 0;
    this.tileEntryCapacity = 1;
    this.tileBufferByteLimit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
    growTileEntryCapacity(1, 0, this.tileBufferByteLimit);
    this._retiredTileShapeBuffers = [];
  }

  static async create(device, opts) {
    const r = new Renderer(device, opts);
    try {
      await r.#init();
      return r;
    } catch (error) {
      r.destroy();
      throw error;
    }
  }

  async #init() {
    const d = this.device;
    const module = await createShaderModule(d, WGSL_SOURCES);
    this.module = module; // retained for lazily specialized pipelines

    const mk = (size, usage) => d.createBuffer({ size, usage });
    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.uniforms = mk(80, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.shapesBuf = mk(Math.max(this.maxShapes * 64, 64), STORAGE);
    this.piecesBuf = mk(Math.max(this.maxPieces * 24, 4), STORAGE);
    this.imageBuf = mk(this.pixels * 16, STORAGE);
    this.dLdImageBuf = mk(this.pixels * 16, STORAGE);
    // Geometry buffers are the large ones (pieces * 24 and curves * 24), and
    // curveMeta is uploaded every step purely for reduce_curve_grads, so a
    // frozen-geometry renderer allocates and uploads none of them.
    // `layout: 'auto'` derives bindings from *static* references in the module,
    // which override specialization does not remove -- so a frozen group's
    // binding must still exist. Bind a 4-byte placeholder instead of the real
    // buffer: the allocation is what costs (pieces * 24 can be tens of MB), not
    // the binding. curveGrads and curveMeta are referenced only by
    // reduce_curve_grads, a pipeline we simply do not create, so those really
    // are never allocated.
    const wantShape = this.train.colour || this.train.alpha || this.train.blur;
    this.wantShape = wantShape;
    this.pieceGradsBuf = mk(this.train.geometry ? Math.max(this.maxPieces * 48, 8) : 4, STORAGE);
    this.shapeGradsBuf = mk(
      wantShape ? Math.max(SLOTS * this.maxShapes * this.gradStride * 8, 8) : 4, STORAGE);
    this.curveMetaBuf = this.train.geometry ? mk(Math.max(this.maxCurves * 16, 16), STORAGE) : null;
    this.curveGradsBuf = this.train.geometry ? mk(Math.max(this.maxCurves * 24, 4), STORAGE) : null;
    this.tileCountBuf = mk(this.nTiles * 4, STORAGE);
    this.tileOffsetBuf = mk((this.nTiles + 1) * 4, STORAGE);
    this.scanBlocks = Math.ceil(this.nTiles / SCAN_WG);
    this.tileBlockSumBuf = mk(Math.max(this.scanBlocks * 4, 4), STORAGE);
    this.tileShapesBuf = mk(4, STORAGE);
    this.slots = SLOTS;
    // Fusing L2 into the composite kernel saves a dispatch and an image
    // round-trip, but adds live state to the heavy loop. WF_FUSE_L2=0 selects
    // the generic two-pass form for A/B measurement.
    this.fuseL2 = readEnv('WF_FUSE_L2') !== '0';
    // WF_SCAN=serial restores the original single-lane tile-offset scan, for
    // measuring what the parallel scan is worth on a given canvas.
    this.serialScan = readEnv('WF_SCAN') === 'serial';

    const pipe = (entryPoint) => d.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
    this.blendCode = BLEND_MODES[this.blend].code;
    this.sorted = BLEND_MODES[this.blend].sorted;
    this.fwdPipe = d.createComputePipeline({
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'forward',
        constants: { BLEND_MODE: this.blendCode, OUTPUT_ALPHA: this.outputAlpha ? 1 : 0 },
      },
    });
    this.bwdPipe = d.createComputePipeline({
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'backward',
        constants: {
          BLEND_MODE: this.blendCode,
          TRAIN_GEOMETRY: this.train.geometry ? 1 : 0,
          TRAIN_COLOR: this.train.colour ? 1 : 0,
          TRAIN_ALPHA: this.train.alpha ? 1 : 0,
          TRAIN_BLUR: this.train.blur ? 1 : 0,
          SHAPE_GRAD_STRIDE: this.gradStride,
        },
      },
    });
    this.binCountPipe = pipe('bin_count');
    this.binScanPipe = pipe('bin_scan');
    this.binScanBlockPipe = pipe('bin_scan_block');
    this.binScanBlocksPipe = pipe('bin_scan_blocks');
    this.binScanAddPipe = pipe('bin_scan_add');
    this.binFillPipe = pipe('bin_fill');
    // Pipelines are specialized per capacity and cached; the scene picks one.
    // Order-independent blends never sort, so they never build a sort pipeline.
    this.deviceSortCapacity = sortCapacity(d.limits);
    this.sortCapacity = SORT_CAPACITY_MIN;
    this._sortPipes = new Map();
    this._sortBinds = new Map();
    this.binSortPipe = this.sorted ? this.#sortPipeline(this.sortCapacity) : null;
    this.reducePipe = wantShape
      ? d.createComputePipeline({
          layout: 'auto',
          compute: {
            module,
            entryPoint: 'reduce_slots',
            constants: { SHAPE_GRAD_STRIDE: this.gradStride },
          },
        })
      : null;
    this.curveReducePipe = this.train.geometry ? pipe('reduce_curve_grads') : null;

    const entry = (binding, buffer) => ({ binding, resource: { buffer } });
    const bind = (pipe, entries) => d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    this.binCountBind = bind(this.binCountPipe, [
      entry(0, this.uniforms),
      entry(1, this.shapesBuf),
      entry(9, this.tileCountBuf),
    ]);
    this.binScanBind = bind(this.binScanPipe, [
      entry(0, this.uniforms),
      entry(9, this.tileCountBuf),
      entry(10, this.tileOffsetBuf),
    ]);
    this.binScanBlockBind = bind(this.binScanBlockPipe, [
      entry(0, this.uniforms),
      entry(9, this.tileCountBuf),
      entry(10, this.tileOffsetBuf),
      entry(14, this.tileBlockSumBuf),
    ]);
    this.binScanBlocksBind = bind(this.binScanBlocksPipe, [
      entry(0, this.uniforms),
      entry(10, this.tileOffsetBuf),
      entry(14, this.tileBlockSumBuf),
    ]);
    this.binScanAddBind = bind(this.binScanAddPipe, [
      entry(0, this.uniforms),
      entry(10, this.tileOffsetBuf),
      entry(14, this.tileBlockSumBuf),
    ]);
    this.reduceBind = this.reducePipe ? bind(this.reducePipe, [
      entry(0, this.uniforms),
      entry(6, this.shapeGradsBuf),
    ]) : null;
    this.curveReduceBind = this.curveReducePipe ? bind(this.curveReducePipe, [
      entry(0, this.uniforms),
      entry(5, this.pieceGradsBuf),
      entry(12, this.curveMetaBuf),
      entry(13, this.curveGradsBuf),
    ]) : null;
    this.#refreshTileShapeBindings();
  }

  #sortPipeline(capacity) {
    let pipe = this._sortPipes.get(capacity);
    if (!pipe) {
      pipe = this.device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: this.module,
          entryPoint: 'bin_sort',
          constants: { SORT_CAPACITY: capacity },
        },
      });
      this._sortPipes.set(capacity, pipe);
    }
    return pipe;
  }

  #sortBindGroup(capacity) {
    let bind = this._sortBinds.get(capacity);
    if (!bind) {
      bind = this.device.createBindGroup({
        layout: this.#sortPipeline(capacity).getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniforms } },
          { binding: 10, resource: { buffer: this.tileOffsetBuf } },
          { binding: 11, resource: { buffer: this.tileShapesBuf } },
        ],
      });
      this._sortBinds.set(capacity, bind);
    }
    return bind;
  }

  #refreshTileShapeBindings() {
    // Sort bind groups reference the tile buffers, so they are rebuilt too.
    this._sortBinds?.clear();
    const d = this.device;
    const entry = (binding, buffer) => ({ binding, resource: { buffer } });
    const bind = (pipe, entries) => d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    this.fwdBind = bind(this.fwdPipe, [
      entry(0, this.uniforms),
      entry(1, this.shapesBuf),
      entry(2, this.piecesBuf),
      entry(3, this.imageBuf),
      entry(10, this.tileOffsetBuf),
      entry(11, this.tileShapesBuf),
    ]);
    this.bwdBind = bind(this.bwdPipe, [
      entry(0, this.uniforms),
      entry(1, this.shapesBuf),
      entry(2, this.piecesBuf),
      entry(3, this.imageBuf),
      entry(4, this.dLdImageBuf),
      entry(5, this.pieceGradsBuf),
      entry(6, this.shapeGradsBuf),
      entry(10, this.tileOffsetBuf),
      entry(11, this.tileShapesBuf),
    ]);
    this.binFillBind = bind(this.binFillPipe, [
      entry(0, this.uniforms),
      entry(1, this.shapesBuf),
      entry(9, this.tileCountBuf),
      entry(10, this.tileOffsetBuf),
      entry(11, this.tileShapesBuf),
    ]);
    // The fused L2 pipeline reads the tile lists too, so it must be rebound
    // whenever the tile-shape buffer is reallocated.
    if (this.fwdL2Pipe) {
      this.fwdL2Bind = bind(this.fwdL2Pipe, [
        entry(0, this.uniforms),
        entry(1, this.shapesBuf),
        entry(2, this.piecesBuf),
        entry(3, this.imageBuf),
        entry(4, this.dLdImageBuf),
        entry(7, this.targetBuf),
        entry(8, this.lossBuf),
        entry(10, this.tileOffsetBuf),
        entry(11, this.tileShapesBuf),
      ]);
    }
  }

  #ensureTileEntryCapacity(required) {
    this.tileEntryCount = required;
    this.peakTileEntryCount = Math.max(this.peakTileEntryCount, required);
    const capacity = growTileEntryCapacity(
      this.tileEntryCapacity,
      required,
      this.tileBufferByteLimit,
    );
    if (capacity === this.tileEntryCapacity) return;
    const previous = this.tileShapesBuf;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.tileShapesBuf = this.device.createBuffer({ size: capacity * 4, usage });
    this.tileEntryCapacity = capacity;
    this.#refreshTileShapeBindings();
    this._retiredTileShapeBuffers.push(previous);
  }

  getCullingInfo() {
    const denseEntries = this.nTiles * this.maxShapes;
    return {
      tileSize: TILE,
      tiles: [this.tilesX, this.tilesY],
      entries: this.tileEntryCount,
      peakEntries: this.peakTileEntryCount,
      capacity: this.tileEntryCapacity,
      bytes: this.tileEntryCount * 4,
      peakBytes: this.peakTileEntryCount * 4,
      capacityBytes: this.tileEntryCapacity * 4,
      denseEntries,
      denseBytes: denseEntries * 4,
    };
  }

  // Allocate the target-sized L2 buffers only when used.
  #ensureL2() {
    if (this.l2gradPipe) return;
    const d = this.device;
    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.targetBuf = d.createBuffer({ size: this.pixels * 16, usage: STORAGE });
    this.lossBuf = d.createBuffer({ size: 16, usage: STORAGE });
    this.l2gradPipe = d.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.module,
        entryPoint: 'l2grad',
        constants: { TONEMAP: this.tonemapCode },
      },
    });
    this.l2gradBind = d.createBindGroup({
      layout: this.l2gradPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniforms } },
        { binding: 3, resource: { buffer: this.imageBuf } },
        { binding: 4, resource: { buffer: this.dLdImageBuf } },
        { binding: 7, resource: { buffer: this.targetBuf } },
        { binding: 8, resource: { buffer: this.lossBuf } },
      ],
    });
    // Fused forward+L2: same outputs as forward then l2grad, one dispatch.
    this.fwdL2Pipe = d.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.module,
        entryPoint: 'forward_l2',
        constants: {
          BLEND_MODE: this.blendCode,
          OUTPUT_ALPHA: this.outputAlpha ? 1 : 0,
          TONEMAP: this.tonemapCode,
        },
      },
    });
    this.fwdL2Bind = d.createBindGroup({
      layout: this.fwdL2Pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniforms } },
        { binding: 1, resource: { buffer: this.shapesBuf } },
        { binding: 2, resource: { buffer: this.piecesBuf } },
        { binding: 3, resource: { buffer: this.imageBuf } },
        { binding: 4, resource: { buffer: this.dLdImageBuf } },
        { binding: 7, resource: { buffer: this.targetBuf } },
        { binding: 8, resource: { buffer: this.lossBuf } },
        { binding: 10, resource: { buffer: this.tileOffsetBuf } },
        { binding: 11, resource: { buffer: this.tileShapesBuf } },
      ],
    });
  }

  // scene: output of prep.packScene; settings: {s:[sx,sy], scale, origin:[x,y], bg:[r,g,b]}
  uploadScene(scene, { s, scale = 1, origin = [0, 0], bg = [1, 1, 1] }) {
    const d = this.device;
    // The Reinhard operators keep their pole unreachable only if the whole
    // composite stays nonnegative; the codec-validated color range covers the
    // shapes, and the background is checked here where it arrives.
    if (TONEMAP_DOMAINS[this.tonemap] === 'nonneg' &&
        (bg[0] < 0 || bg[1] < 0 || bg[2] < 0)) {
      throw new Error(`tonemap '${this.tonemap}' needs a nonnegative background`);
    }
    if (!(scene.pieceData instanceof Float32Array) || scene.pieceData.length < scene.pieceCount * 6) {
      throw new Error('pieceData must contain 6 values per compact piece');
    }
    if (scene.pieceCount > this.maxPieces) throw new Error(`pieceCount ${scene.pieceCount} > maxPieces ${this.maxPieces}`);
    if (!(scene.shapeData instanceof Float32Array) || scene.shapeData.length % 16 !== 0) {
      throw new Error('shapeData must contain 16 values per shape');
    }
    const nShapes = scene.shapeData.length / 16;
    if (nShapes > this.maxShapes) throw new Error(`nShapes ${nShapes} > maxShapes ${this.maxShapes}`);
    const curveMeta = scene.curveMetaData;
    if (!(curveMeta instanceof Float32Array) || curveMeta.length !== scene.curveCount * 4) {
      throw new Error('curveMetaData must contain 4 words per curve');
    }
    if (scene.curveCount > this.maxCurves) {
      throw new Error(`curveCount ${scene.curveCount} > maxCurves ${this.maxCurves}`);
    }
    const previousCapacity = this.sortCapacity;
    const tileEntries = countTileEntries(scene.shapeData, {
      width: this.width,
      height: this.height,
      s,
      scale,
      origin,
    });
    this.#ensureTileEntryCapacity(tileEntries);
    // Pick the sort specialization this scene needs. Guessing low only costs
    // speed -- the serial fallback is exact -- so a mean-based estimate is safe.
    if (this.sorted) {
      this.sortCapacity = sortCapacityFor(tileEntries / this.nTiles, this.deviceSortCapacity);
      if (this.sortCapacity !== previousCapacity) {
        this.binSortPipe = this.#sortPipeline(this.sortCapacity);
      }
    }
    const u32 = this._uniformU32;
    const f32 = this._uniformF32;
    u32[0] = this.width;
    u32[1] = this.height;
    u32[2] = nShapes;
    u32[3] = this.slots;
    f32[4] = origin[0];
    f32[5] = origin[1];
    f32[6] = scale;
    f32[7] = this._fixedScale ?? 0;
    f32[8] = s[0];
    f32[9] = s[1];
    u32[10] = this.tilesX;
    u32[11] = this.tilesY;
    f32[12] = bg[0];
    f32[13] = bg[1];
    f32[14] = bg[2];
    u32[15] = scene.curveCount;
    f32[16] = this._exposure;
    f32[17] = this._white;
    d.queue.writeBuffer(this.uniforms, 0, this._uniformData);
    d.queue.writeBuffer(this.shapesBuf, 0, scene.shapeData);
    if (scene.pieceCount) d.queue.writeBuffer(this.piecesBuf, 0, scene.pieceData, 0, scene.pieceCount * 6);
    // curveMeta feeds reduce_curve_grads only; frozen geometry never reads it.
    if (this.curveMetaBuf && curveMeta.byteLength) {
      d.queue.writeBuffer(this.curveMetaBuf, 0, curveMeta);
    }
    this.nShapes = nShapes;
    this.nPieces = scene.pieceCount;
    this.nCurves = scene.curveCount;
  }

  // WebGPU orders dispatches within a pass, so a whole phase is one pass.
  #dispatch(pass, pipe, bind, x, y = 1) {
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(x, y);
  }

  // Build sorted per-tile shape lists; forward and backward share them.
  // Callers clear tileCountBuf first (bin_count accumulates).
  #binDispatches(pass) {
    const shapeWG = Math.ceil(this.nShapes / 64);
    if (shapeWG) this.#dispatch(pass, this.binCountPipe, this.binCountBind, shapeWG);
    if (this.serialScan) {
      this.#dispatch(pass, this.binScanPipe, this.binScanBind, 1);
    } else {
      this.#dispatch(pass, this.binScanBlockPipe, this.binScanBlockBind, this.scanBlocks);
      this.#dispatch(pass, this.binScanBlocksPipe, this.binScanBlocksBind, 1);
      this.#dispatch(pass, this.binScanAddPipe, this.binScanAddBind, this.scanBlocks);
    }
    if (shapeWG) {
      this.#dispatch(pass, this.binFillPipe, this.binFillBind, shapeWG);
      // Order-independent blends composite bin_fill's arbitrary order exactly.
      if (this.sorted) {
        this.#dispatch(pass, this.binSortPipe, this.#sortBindGroup(this.sortCapacity), this.nTiles);
      }
    }
  }

  #reduceDispatches(pass) {
    if (this.reducePipe) {
      this.#dispatch(pass, this.reducePipe, this.reduceBind,
        Math.ceil((this.nShapes * this.gradStride) / 64));
    }
    if (this.curveReducePipe && this.nCurves) {
      this.#dispatch(pass, this.curveReducePipe, this.curveReduceBind, Math.ceil(this.nCurves / 64));
    }
  }

  // Clear gradient accumulators without a host allocation or upload.
  #clearGrads(enc) {
    // Frozen groups accumulate nothing, so their placeholder is never cleared.
    // hi words, then the lo words in the buffer's second half.
    if (this.train.geometry) {
      enc.clearBuffer(this.pieceGradsBuf, 0, this.nPieces * 24);
      enc.clearBuffer(this.pieceGradsBuf, this.pieceGradsBuf.size / 2, this.nPieces * 24);
    }
    if (this.wantShape) {
      const bytes = this.slots * this.nShapes * this.gradStride * 4;
      enc.clearBuffer(this.shapeGradsBuf, 0, bytes);
      enc.clearBuffer(this.shapeGradsBuf, this.shapeGradsBuf.size / 2, bytes);
    }
  }

  // Reuse staging buffers across steps.
  #staging(key, bytes) {
    this._staging ??= new Map();
    let s = this._staging.get(key);
    if (!s || s.size < bytes) {
      s?.destroy();
      s = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      this._staging.set(key, s);
    }
    return s;
  }

  async #readback(buffer, bytes) {
    return (await this.#readbackMany([{ key: 'r', buffer, bytes }]))[0];
  }

  // Pack results into one staging buffer and one mapAsync.
  async #mapMany(specs, encodeBeforeCopy = null) {
    const d = this.device;
    const enc = d.createCommandEncoder();
    encodeBeforeCopy?.(enc);
    const offsets = [];
    let bytes = 0;
    for (const s of specs) {
      bytes = Math.ceil(bytes / 8) * 8; // getMappedRange offsets require 8-byte alignment
      offsets.push(bytes);
      bytes += s.bytes;
    }
    bytes = Math.max(4, Math.ceil(bytes / 4) * 4);
    const key = `many:${specs.map((s) => s.key).join('+')}`;
    const stg = this.#staging(key, bytes);
    for (let i = 0; i < specs.length; i++) {
      const s = specs[i];
      if (s.bytes) enc.copyBufferToBuffer(s.buffer, 0, stg, offsets[i], s.bytes);
    }
    d.queue.submit([enc.finish()]);
    await stg.mapAsync(GPUMapMode.READ, 0, bytes);
    return { buffer: stg, offsets };
  }

  // Read several GPU buffers (as fresh Float32Arrays) in a single sync point.
  async #readbackMany(specs) {
    const mapped = await this.#mapMany(specs);
    const out = specs.map((s, i) => new Float32Array(
      mapped.buffer.getMappedRange(mapped.offsets[i], s.bytes).slice(0),
    ));
    mapped.buffer.unmap();
    return out;
  }

  // The current exposure/white point, for hosts that display alongside training.
  get exposure() {
    return this._exposure;
  }

  get white() {
    return this._white;
  }

  // The learnable exposure (and white point, for the white-normalized
  // operators) of a tonemapped loss; plain uniform writes, so they can change
  // every step without touching any pipeline.
  setExposure(k, w = undefined) {
    const effectiveK = positiveF32(k, 'exposure');
    const effectiveW = w === undefined ? this._white : positiveF32(w, 'white point');
    if (this._exposure === effectiveK && this._white === effectiveW) return;
    this._uniformF32[16] = effectiveK;
    this._uniformF32[17] = effectiveW;
    this.device.queue.writeBuffer(this.uniforms, 64, this._uniformData, 64, 8);
    this._exposure = effectiveK;
    this._white = effectiveW;
  }

  // Scale fixed-point accumulation to stay within i32.
  #setFixedScale(scale) {
    const effective = positiveF32(scale, 'fixed scale');
    if (this._fixedScale === effective) return;
    this._uniformF32[7] = effective;
    this.device.queue.writeBuffer(this.uniforms, 28, this._uniformData, 28, 4);
    this._fixedScale = effective;
  }

  // Copy mapped f32 results into a reusable host array.
  #f32Copy(stg, n, key, offset = 0) {
    this._f32Out ??= new Map();
    let out = this._f32Out.get(key);
    if (!out || out.length !== n) {
      out = new Float32Array(n);
      this._f32Out.set(key, out);
    }
    if (n === 0) return out;
    out.set(new Float32Array(stg.getMappedRange(offset, n * 4)));
    return out;
  }

  // Frozen groups are omitted from the readback entirely -- their arrays come
  // back undefined rather than zero-filled, which is the point: a zero fill
  // would keep the transfer we are trying to avoid.
  #gradSpecs() {
    const specs = [];
    if (this.train.geometry) specs.push(this.#curveGradSpec());
    if (this.wantShape) {
      specs.push({
        key: 'shape',
        buffer: this.shapeGradsBuf,
        bytes: this.nShapes * this.gradStride * 4,
      });
    }
    return specs;
  }

  // Blur training interleaves (r, g, b, alpha, sx, sy) per shape; consumers
  // keep the stride-4 shapeGrads layout, so split the wide decode into the
  // familiar array plus a separate (sx, sy) pair per shape.
  #splitShapeGrads(stg, offset) {
    const n = this.nShapes;
    const wide = this.#f32Copy(stg, n * 6, 'shapeWide', offset);
    this._blurSplit ??= { shape: new Float32Array(0), blur: new Float32Array(0) };
    const split = this._blurSplit;
    if (split.shape.length !== n * 4) {
      split.shape = new Float32Array(n * 4);
      split.blur = new Float32Array(n * 2);
    }
    for (let i = 0; i < n; i++) {
      split.shape[4 * i] = wide[6 * i];
      split.shape[4 * i + 1] = wide[6 * i + 1];
      split.shape[4 * i + 2] = wide[6 * i + 2];
      split.shape[4 * i + 3] = wide[6 * i + 3];
      split.blur[2 * i] = wide[6 * i + 4];
      split.blur[2 * i + 1] = wide[6 * i + 5];
    }
    return split;
  }

  #gradResult(stg, offsets) {
    const out = {};
    let i = 0;
    if (this.train.geometry) out.curveGrads = this.#curveGradResult(stg, offsets[i++]);
    if (this.wantShape) {
      if (this.train.blur) {
        const split = this.#splitShapeGrads(stg, offsets[i++]);
        out.shapeGrads = split.shape;
        out.blurGrads = split.blur;
      } else {
        out.shapeGrads = this.#f32Copy(stg, this.nShapes * 4, 'shape', offsets[i++]);
      }
    }
    return { out, used: i };
  }

  #curveGradSpec() {
    return { key: 'curve', buffer: this.curveGradsBuf, bytes: this.nCurves * 24 };
  }

  #curveGradResult(stg, offset = 0) {
    if (this.nCurves === 0) return (this._emptyCurveGrads ??= new Float32Array(0));
    return this.#f32Copy(stg, this.nCurves * 6, 'curve', offset);
  }

  // Forward render. Returns Float32Array(H*W*4) rgba.
  async forward() {
    this.forwardNoRead();
    return this.#readback(this.imageBuf, this.pixels * 16);
  }

  // Forward without host readback.
  forwardNoRead() {
    const enc = this.device.createCommandEncoder();
    enc.clearBuffer(this.tileCountBuf, 0, this.nTiles * 4);
    const pass = enc.beginComputePass();
    this.#binDispatches(pass);
    this.#dispatch(pass, this.fwdPipe, this.fwdBind, this.gridX, this.gridY);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  // Read the current H*W*4 image.
  async readImage() {
    return this.#readback(this.imageBuf, this.pixels * 16);
  }

  // Upload the (fixed) target image once; call before stepping.
  uploadTarget(target) {
    if (!(target instanceof Float32Array) || target.length !== this.pixels * 4) {
      throw new Error(`target must be a Float32Array of length ${this.pixels * 4}`);
    }
    this.#ensureL2();
    this.device.queue.writeBuffer(this.targetBuf, 0, target);
  }

  // Whole L2 step at one sync point: the forward submit renders while the
  // host encodes backward, reductions, and the packed readback.
  async stepGpuLoss() {
    this.#ensureL2();
    // L2's bounded cotangent gives a constant safe fixed-point scale. The
    // bound assumes residuals of order 1: src-over and multiply keep the image
    // inside [0, 1], and add-mode scenes keep the ~20x i32 headroom unless the
    // render overshoots the target by more than that factor.
    this.#setFixedScale(1e8 / (this.pixels * (2 / (3 * this.pixels))));
    {
      const enc = this.device.createCommandEncoder();
      enc.clearBuffer(this.lossBuf, 0, 12); // word 0: loss, word 1: dL/dk, word 2: dL/dW
      enc.clearBuffer(this.tileCountBuf, 0, this.nTiles * 4);
      const pass = enc.beginComputePass();
      this.#binDispatches(pass);
      if (this.fuseL2) {
        this.#dispatch(pass, this.fwdL2Pipe, this.fwdL2Bind, this.gridX, this.gridY);
      } else {
        this.#dispatch(pass, this.fwdPipe, this.fwdBind, this.gridX, this.gridY);
        this.#dispatch(pass, this.l2gradPipe, this.l2gradBind, this.gridX, this.gridY);
      }
      pass.end();
      this.device.queue.submit([enc.finish()]);
    }
    const mapped = await this.#mapMany([
      ...this.#gradSpecs(),
      { key: 'loss', buffer: this.lossBuf, bytes: 12 },
    ], (enc) => {
      this.#clearGrads(enc);
      const pass = enc.beginComputePass();
      this.#dispatch(pass, this.bwdPipe, this.bwdBind, this.gridX, this.gridY);
      this.#reduceDispatches(pass);
      pass.end();
    });
    const stg = mapped.buffer;
    const { out, used } = this.#gradResult(stg, mapped.offsets);
    const lossWords = new Int32Array(stg.getMappedRange(mapped.offsets[used], 12));
    // 2^30, matches LOSS_SCALE
    out.loss = lossWords[0] / 1073741824;
    // 2^24, matches KGRAD_SCALE; only a tonemapped loss accumulates it, and
    // only the white-point operators accumulate dL/dW.
    if (this.tonemapCode) out.kGrad = lossWords[1] / 16777216;
    if (this.hasWhite) out.wGrad = lossWords[2] / 16777216;
    stg.unmap();
    return out;
  }

  // Backward VJP. Returned arrays are reused on the next call.
  async backward(dLdImage) {
    const d = this.device;
    if (!(dLdImage instanceof Float32Array) || dLdImage.length !== this.pixels * 4) {
      throw new Error(`dLdImage must be a Float32Array of length ${this.pixels * 4}`);
    }
    d.queue.writeBuffer(this.dLdImageBuf, 0, dLdImage);
    // Scale from the largest cotangent.
    let maxAbs = 0;
    for (let i = 0; i < dLdImage.length; i++) { const a = Math.abs(dLdImage[i]); if (a > maxAbs) maxAbs = a; }
    this.#setFixedScale(maxAbs > 0 ? 1e8 / (this.pixels * maxAbs) : 1);
    // Read both accumulators at one sync point.
    const mapped = await this.#mapMany(this.#gradSpecs(), (enc) => {
      this.#clearGrads(enc);
      const pass = enc.beginComputePass();
      this.#dispatch(pass, this.bwdPipe, this.bwdBind, this.gridX, this.gridY);
      this.#reduceDispatches(pass);
      pass.end();
    });
    const stg = mapped.buffer;
    const { out } = this.#gradResult(stg, mapped.offsets);
    stg.unmap();
    return out;
  }

  destroy() {
    const names = [
      'uniforms', 'shapesBuf', 'piecesBuf', 'imageBuf', 'dLdImageBuf',
      'pieceGradsBuf', 'shapeGradsBuf', 'curveMetaBuf', 'curveGradsBuf',
      'tileCountBuf', 'tileOffsetBuf', 'tileShapesBuf', 'targetBuf', 'lossBuf',
      'tileBlockSumBuf',
    ];
    for (const name of names) this[name]?.destroy();
    for (const buffer of this._retiredTileShapeBuffers) buffer.destroy();
    this._retiredTileShapeBuffers.length = 0;
    for (const buffer of this._staging?.values() ?? []) buffer.destroy();
    this._staging?.clear();
  }
}
