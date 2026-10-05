// Self-contained host for the Slug comparison renderer (slug.wgsl).

import { readText } from '../util/runtime.js';

const WGSL_URL = new URL('./slug.wgsl', import.meta.url);

// ~6 curves per band, at most 64 bands.
const TARGET_PER_BAND = 6;
const MAX_BANDS = 64;
// Must equal SORT_MIN in slug.wgsl.
export const BAND_SORT_MIN = 4;
export const FLOATS_PER_INSTANCE = 20;

const bandIndex = (y, y0, invH, R) => (invH <= 0 ? 0 : Math.min(Math.max(Math.floor((y - y0) * invH), 0), R - 1));

/** File quads into row bands over [y0, y1]; horizontal quads are dropped. */
export function bandQuads(quads, y0, y1, curveOut, rowOut) {
  const n = quads.length / 6;
  const R = n <= TARGET_PER_BAND ? 1 : Math.min(Math.ceil(n / TARGET_PER_BAND), MAX_BANDS);
  const invH = R > 1 && y1 > y0 ? R / (y1 - y0) : 0;
  const buckets = Array.from({ length: R }, () => []);
  for (let k = 0; k < n; k++) {
    const yLo = Math.min(quads[k * 6 + 1], quads[k * 6 + 3], quads[k * 6 + 5]);
    const yHi = Math.max(quads[k * 6 + 1], quads[k * 6 + 3], quads[k * 6 + 5]);
    const lo = bandIndex(yLo, y0, invH, R), hi = bandIndex(yHi, y0, invH, R);
    for (let b = lo; b <= hi; b++) buckets[b].push(k);
  }
  const rowBase = rowOut.length / 2;
  const xMax = (k) => Math.max(quads[k * 6], quads[k * 6 + 2], quads[k * 6 + 4]);
  const hasYSpan = (k) => {
    const y = Math.fround(quads[k * 6 + 1]);
    return y !== Math.fround(quads[k * 6 + 3]) || y !== Math.fround(quads[k * 6 + 5]);
  };
  for (const bucket of buckets) {
    if (bucket.length > BAND_SORT_MIN) bucket.sort((a, c) => xMax(c) - xMax(a));
    const gather = bucket.filter(hasYSpan);
    rowOut.push(curveOut.length / 6, gather.length);
    for (const k of gather) {
      for (let j = 0; j < 6; j++) curveOut.push(quads[k * 6 + j]);
    }
  }
  return { rowBase, bandCount: R, invH };
}

// Rotate by -90 degrees into (y, -x) for the vertical bands.
function rotateQuads(quads) {
  const out = new Array(quads.length);
  for (let i = 0; i < quads.length; i += 2) {
    out[i] = quads[i + 1];
    out[i + 1] = -quads[i];
  }
  return out;
}

/** Hull bbox [x0, y0, x1, y1] of interleaved x, y coordinates. */
export function hullBox(flat) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < flat.length; i += 2) {
    if (flat[i] < x0) x0 = flat[i];
    if (flat[i] > x1) x1 = flat[i];
    if (flat[i + 1] < y0) y0 = flat[i + 1];
    if (flat[i + 1] > y1) y1 = flat[i + 1];
  }
  return [x0, y0, x1, y1];
}

/** One shape as Slug's dual band atlas plus its 20-float instance (white, pixel units). */
export function buildSlugScene(quads, evenodd = false) {
  const bbox = hullBox(quads);
  const [x0, y0, x1, y1] = bbox;
  const curves = [], rows = [];
  const h = bandQuads(quads, y0, y1, curves, rows);
  // Vertical bands file the rotated quads by their rotated y (= -x) over [-x1, -x0].
  const v = bandQuads(rotateQuads(quads), -x1, -x0, curves, rows);
  const instance = new Float32Array([
    0, 0, 1, evenodd ? 1 : 0,
    ...bbox,
    1, 1, 1, 1,
    h.rowBase, h.bandCount, y0, h.invH,
    v.rowBase, v.bandCount, -x1, v.invH,
  ]);
  return { curves: new Float32Array(curves), rows: new Uint32Array(rows), instance };
}

function storage(device, data) {
  const buffer = device.createBuffer({
    size: Math.max(16, data.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  if (data.byteLength) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

/** Render flat quads white over black with Slug; returns coverage in 0..1. */
export async function slugCoverage(device, quads, { size, evenodd = false }) {
  const { curves, rows, instance } = buildSlugScene(quads, evenodd);
  const module = device.createShaderModule({ code: await readText(WGSL_URL) });
  const format = 'rgba8unorm';
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: {
      module,
      entryPoint: 'fs',
      targets: [{
        format,
        // premultiplied over: out = src + dst * (1 - src.a)
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'triangle-strip' },
  });
  // res, style (unused), camera (identity)
  const uniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uniform, 0, new Float32Array([size, size, 1, 1, 1, 1, 0, 0]));
  const buffers = [uniform, storage(device, instance), storage(device, curves), storage(device, rows)];
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });

  const target = device.createTexture({
    size: [size, size],
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const bytesPerRow = Math.ceil((size * 4) / 256) * 256;
  const readback = device.createBuffer({
    size: bytesPerRow * size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: target.createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      loadOp: 'clear',
      storeOp: 'store',
    }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(4, 1);
  pass.end();
  encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow }, [size, size]);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(readback.getMappedRange());
  const cov = new Float64Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) cov[y * size + x] = padded[y * bytesPerRow + x * 4] / 255;
  }
  readback.unmap();
  for (const buffer of [readback, ...buffers]) buffer.destroy();
  target.destroy();
  return cov;
}
