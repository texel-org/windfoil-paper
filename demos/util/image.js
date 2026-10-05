import { unzlibSync, zlibSync } from 'fflate';
import {
  ChunkType,
  ColorType,
  decode as decodePng,
  encode as encodePng,
  encode_pHYs_PPI,
} from 'png-tools';
import { readBytes } from './runtime.js';

const clampSample = (x, max) => Math.max(0, Math.min(max, Math.round(x * max)));
const clamp8 = (x) => clampSample(x, 255);

export async function loadTarget(path, width, height = width) {
  return targetFromSource(await loadImageSource(path), width, height);
}

export async function loadImageSource(path) {
  const bytes = await readBytes(path);
  let source;
  // png-tools expands every PNG color type to 8- or 16-bit RGBA.
  if (isPng(bytes)) source = decodePng(bytes, unzlibSync);
  else if (isJpeg(bytes)) {
    const jpeg = await import('jpeg-js');
    const decoded = (jpeg.decode ?? jpeg.default.decode)(bytes, { useTArray: true, formatAsRGBA: true });
    source = { ...decoded, channels: 4, depth: 8 };
  } else throw new Error('unsupported image format: expected PNG or JPEG');
  if (source.depth !== 8 && source.depth !== 16) throw new Error(`unsupported image depth: ${source.depth}`);
  return source;
}

const isPng = (bytes) => bytes.length >= 8 &&
  [137, 80, 78, 71, 13, 10, 26, 10].every((value, index) => bytes[index] === value);
const isJpeg = (bytes) => bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;

export function targetFromSource(source, width, height = width) {
  const rgba = resizeImage(source, width, height);
  const u8 = new Uint8Array(rgba.length);
  const mean = [0, 0, 0];
  const pixels = width * height;
  for (let i = 0; i < pixels; i++) {
    for (let c = 0; c < 3; c++) {
      mean[c] += rgba[4 * i + c];
      u8[4 * i + c] = clamp8(rgba[4 * i + c]);
    }
    u8[4 * i + 3] = 255;
  }
  for (let c = 0; c < 3; c++) mean[c] /= pixels;
  return { rgba, u8, mean, sourceWidth: source.width, sourceHeight: source.height };
}

export function resizeImage(source, width, height) {
  const { width: sw, height: sh, channels, data, depth } = source;
  const max = depth === 16 ? 65535 : 255;
  const out = new Float32Array(width * height * 4);

  const sample = (x, y, c) => {
    const index = (Math.max(0, Math.min(sh - 1, y)) * sw + Math.max(0, Math.min(sw - 1, x))) * channels;
    if (channels === 1 || channels === 2) return data[index] / max;
    return data[index + c] / max;
  };
  const alpha = (x, y) => {
    if (channels !== 2 && channels !== 4) return 1;
    const index = (Math.max(0, Math.min(sh - 1, y)) * sw + Math.max(0, Math.min(sw - 1, x))) * channels;
    return data[index + channels - 1] / max;
  };

  for (let y = 0; y < height; y++) {
    const sy = (y + 0.5) * sh / height - 0.5;
    const y0 = Math.floor(sy), fy = sy - y0;
    for (let x = 0; x < width; x++) {
      const sx = (x + 0.5) * sw / width - 0.5;
      const x0 = Math.floor(sx), fx = sx - x0;
      const weights = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy];
      const points = [[x0, y0], [x0 + 1, y0], [x0, y0 + 1], [x0 + 1, y0 + 1]];
      const o = 4 * (y * width + x);
      let a = 0;
      for (let k = 0; k < 4; k++) a += weights[k] * alpha(points[k][0], points[k][1]);
      for (let c = 0; c < 3; c++) {
        let value = 0;
        for (let k = 0; k < 4; k++) {
          value += weights[k] * sample(points[k][0], points[k][1], c) *
            alpha(points[k][0], points[k][1]);
        }
        out[o + c] = value + (1 - a);
      }
      out[o + 3] = 1;
    }
  }
  return out;
}

// Separable box filter matching the renderer's pixel filter of size s: each
// output pixel averages the input over a box s pixels wide, clamped at edges.
export function boxBlurImage(rgba, width, height, size) {
  const kernel = boxKernel(size);
  const pass = (input, w, h, stride, line) => {
    const out = new Float32Array(input.length);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let r = 0, g = 0, b = 0;
        for (let k = 0; k < kernel.weights.length; k++) {
          const tap = Math.max(0, Math.min(w - 1, x + k - kernel.reach));
          const at = line * y + stride * tap;
          const weight = kernel.weights[k];
          r += weight * input[at];
          g += weight * input[at + 1];
          b += weight * input[at + 2];
        }
        const at = line * y + stride * x;
        out[at] = r;
        out[at + 1] = g;
        out[at + 2] = b;
        out[at + 3] = 1;
      }
    }
    return out;
  };
  const horizontal = pass(rgba, width, height, 4, width * 4);
  return pass(horizontal, height, width, width * 4, 4);
}

function boxKernel(size) {
  const half = Math.max(size, 1e-6) / 2;
  const reach = Math.max(0, Math.ceil(half - 0.5));
  const weights = [];
  let total = 0;
  for (let r = -reach; r <= reach; r++) {
    const overlap = Math.max(0, Math.min(half, r + 0.5) - Math.max(-half, r - 0.5));
    weights.push(overlap);
    total += overlap;
  }
  return { reach, weights: weights.map((weight) => weight / total) };
}

// Lazily cache box-blurred copies of a target, bucketing the filter size so
// an annealing schedule reuses a handful of blurred targets.
export function blurredTargetProvider(rgba, width, height, quantum = 0.5) {
  const cache = new Map();
  return (size) => {
    const key = Math.round(size / quantum) * quantum;
    if (key <= 1) return rgba;
    if (!cache.has(key)) cache.set(key, boxBlurImage(rgba, width, height, key));
    return cache.get(key);
  };
}

// Extend an RGBA image outward by `pad` pixels on every side, replicating the
// border pixels (edge clamp) — the same boundary the target blur already uses.
// Optimizing on this enlarged canvas moves the frame edge out of view so edge
// marks are fit like interior ones instead of piling against the boundary.
export function padImage(rgba, width, height, pad) {
  if (pad <= 0) return rgba;
  const pw = width + 2 * pad;
  const ph = height + 2 * pad;
  const out = new Float32Array(pw * ph * 4);
  for (let y = 0; y < ph; y++) {
    const sy = Math.max(0, Math.min(height - 1, y - pad));
    for (let x = 0; x < pw; x++) {
      const sx = Math.max(0, Math.min(width - 1, x - pad));
      const s = 4 * (sy * width + sx);
      const o = 4 * (y * pw + x);
      out[o] = rgba[s];
      out[o + 1] = rgba[s + 1];
      out[o + 2] = rgba[s + 2];
      out[o + 3] = rgba[s + 3];
    }
  }
  return out;
}

// Collapse a loaded target to its Rec. 709 luma in place.
export function toGrayscale(target) {
  const { rgba, u8, mean } = target;
  const pixels = rgba.length / 4;
  let sum = 0;
  for (let i = 0; i < pixels; i++) {
    const at = 4 * i;
    const luma = 0.2126 * rgba[at] + 0.7152 * rgba[at + 1] + 0.0722 * rgba[at + 2];
    rgba[at] = rgba[at + 1] = rgba[at + 2] = luma;
    if (u8) u8[at] = u8[at + 1] = u8[at + 2] = clamp8(luma);
    sum += luma;
  }
  if (mean) mean[0] = mean[1] = mean[2] = sum / pixels;
  return target;
}

// Quantize an f32 RGBA image into a region of an integer RGBA canvas.
// Transparent Windfoil output is premultiplied, as compositing math should be,
// so convert it to the straight-alpha representation expected by PNG while
// copying.
function blitRgbaSamples(
  max,
  canvas,
  canvasWidth,
  x0,
  y0,
  image,
  imageWidth,
  width,
  height,
  { transparent = false } = {},
) {
  for (let row = 0; row < height; row++) {
    let src = 4 * row * imageWidth;
    let at = 4 * ((y0 + row) * canvasWidth + x0);
    for (let col = 0; col < width; col++, src += 4, at += 4) {
      const alpha = transparent ? Math.max(0, Math.min(1, image[src + 3])) : 1;
      const alphaSample = clampSample(alpha, max);
      const straight = transparent && alphaSample > 0 ? 1 / alpha : 1;
      canvas[at] = alphaSample > 0 ? clampSample(image[src] * straight, max) : 0;
      canvas[at + 1] = alphaSample > 0 ? clampSample(image[src + 1] * straight, max) : 0;
      canvas[at + 2] = alphaSample > 0 ? clampSample(image[src + 2] * straight, max) : 0;
      canvas[at + 3] = alphaSample;
    }
  }
}

export function blitRgba8(...args) {
  return blitRgbaSamples(255, ...args);
}

export function blitRgba16(...args) {
  return blitRgbaSamples(65535, ...args);
}

// Paint everything outside a pixel-space rect of a straight-alpha RGBA canvas
// with an opaque color. Pixels the rect only partly covers blend by their exact
// box coverage, matching the renderer's analytic antialiasing.
export function letterboxRgba(canvas, width, height, rect, color, { depth = 8 } = {}) {
  if (depth !== 8 && depth !== 16) throw new Error('PNG depth must be 8 or 16');
  const max = depth === 16 ? 65535 : 255;
  const columns = spanCoverage(width, rect.x, rect.x + rect.width);
  const rows = spanCoverage(height, rect.y, rect.y + rect.height);
  const allColumns = Array.from({ length: width }, (_, col) => col);
  const edgeColumns = allColumns.filter((col) => columns[col] < 1);
  for (let row = 0; row < height; row++) {
    for (const col of rows[row] < 1 ? allColumns : edgeColumns) {
      const coverage = rows[row] * columns[col];
      const at = 4 * (row * width + col);
      const inside = coverage * canvas[at + 3] / max;
      const alpha = inside + 1 - coverage;
      for (let channel = 0; channel < 3; channel++) {
        const blended = canvas[at + channel] / max * inside + color[channel] * (1 - coverage);
        canvas[at + channel] = clampSample(blended / alpha, max);
      }
      canvas[at + 3] = clampSample(alpha, max);
    }
  }
}

function spanCoverage(count, start, end) {
  const coverage = new Float64Array(count);
  for (let i = 0; i < count; i++) coverage[i] = Math.max(0, Math.min(i + 1, end) - Math.max(i, start));
  return coverage;
}

// Encode 8- or 16-bit RGBA samples, optionally tagged with a print resolution.
export function rgbaToPng(data, width, height, { depth = 8, ppi = null } = {}) {
  if (depth !== 8 && depth !== 16) throw new Error('PNG depth must be 8 or 16');
  const ancillary = [];
  if (ppi !== null) {
    if (!(ppi > 0) || !Number.isFinite(ppi)) throw new Error('PNG PPI must be positive and finite');
    ancillary.push({ type: ChunkType.pHYs, data: encode_pHYs_PPI(ppi) });
  }
  return encodePng({ width, height, data, depth, colorType: ColorType.RGBA, ancillary }, zlibSync);
}

export function imageToPng(image, width, height) {
  const data = new Uint8Array(width * height * 4);
  blitRgba8(data, width, 0, 0, image, width, width, height);
  return rgbaToPng(data, width, height);
}

export function l2Quality(image, target) {
  if (image.length !== target.length || image.length % 4) throw new Error('L2 image size mismatch');
  let error = 0;
  const pixels = image.length / 4;
  for (let i = 0; i < pixels; i++) {
    for (let channel = 0; channel < 3; channel++) {
      const delta = image[4 * i + channel] - target[4 * i + channel];
      error += delta * delta;
    }
  }
  const mseRgb = error / (pixels * 3);
  return { mseRgb, psnrDb: 10 * Math.log10(1 / Math.max(mseRgb, 1e-12)) };
}
