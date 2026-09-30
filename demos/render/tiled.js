// Chunked forward rendering: images larger than one GPU pass are rendered
// region by region with a shifted origin and composited on the host.

import { Renderer } from '../../js/renderer.js';
import { blitRgba8, blitRgba16 } from '../util/image.js';

// The renderer's image and cotangent storage buffers hold one f32 RGBA texel
// per pixel; each must fit in a single storage binding.
const IMAGE_BYTES_PER_PIXEL = 16;
// Must match TILE in js/wgsl/scene.wgsl; bin_sort runs one workgroup per bin
// along a single dispatch axis.
const BIN = 16;
const DEFAULT_CHUNK = 4096;

/** Largest square region the device can render in one pass, aligned to BIN. */
export function maxRenderSide(limits) {
  const byteLimit = Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize);
  const byBytes = Math.floor(Math.sqrt(byteLimit / IMAGE_BYTES_PER_PIXEL));
  const byBins = BIN * Math.floor(Math.sqrt(limits.maxComputeWorkgroupsPerDimension));
  const side = Math.floor(Math.min(byBytes, byBins) / BIN) * BIN;
  if (side < BIN) throw new Error(`device limits cannot fit a ${BIN}x${BIN} render`);
  return side;
}

/** Split a raster into renderable chunks of at most `chunk` pixels per side. */
export function planChunks({ width, height, limits, chunk = null }) {
  if (!Number.isInteger(width) || width < 1 || !Number.isInteger(height) || height < 1) {
    throw new Error('width and height must be positive integers');
  }
  const side = maxRenderSide(limits);
  if (chunk !== null && (!Number.isInteger(chunk) || chunk < 1)) {
    throw new Error('chunk size must be a positive integer');
  }
  const cap = Math.min(chunk ?? DEFAULT_CHUNK, side);
  const tileWidth = Math.min(width, cap);
  const tileHeight = Math.min(height, cap);
  const columns = Math.ceil(width / tileWidth);
  const rows = Math.ceil(height / tileHeight);
  const chunks = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const x = column * tileWidth;
      const y = row * tileHeight;
      chunks.push({
        x,
        y,
        width: Math.min(tileWidth, width - x),
        height: Math.min(tileHeight, height - y),
      });
    }
  }
  return { tileWidth, tileHeight, columns, rows, chunks };
}

// Render the scene chunk by chunk into an 8- or 16-bit RGBA buffer. One
// renderer sized to the chunk is reused with a per-chunk origin; edge chunks
// render the full extent and composite only the in-bounds region. Coverage is
// analytic per pixel, so chunk seams need no overlap.
export async function renderChunked(device, scene, raster, {
  s,
  bg,
  transparent = false,
  depth = 8,
  chunk = null,
  onChunk = null,
}) {
  if (depth !== 8 && depth !== 16) throw new Error('PNG depth must be 8 or 16');
  const plan = planChunks({
    width: raster.width,
    height: raster.height,
    limits: device.limits,
    chunk,
  });
  const rgba = depth === 16
    ? new Uint16Array(raster.width * raster.height * 4)
    : new Uint8Array(raster.width * raster.height * 4);
  const blit = depth === 16 ? blitRgba16 : blitRgba8;
  const compositeBackground = transparent ? [0, 0, 0] : bg;
  const renderer = await Renderer.create(device, {
    width: plan.tileWidth,
    height: plan.tileHeight,
    maxShapes: scene.shapeData.length / 16,
    maxPieces: scene.pieceCount,
    maxCurves: scene.curveCount,
    outputAlpha: transparent,
    // This utility only runs the forward pass. Keep the tiny shape-alpha
    // allocation and omit the much larger geometry-gradient buffers.
    train: { geometry: false, colour: false, alpha: true },
  });
  try {
    for (let index = 0; index < plan.chunks.length; index++) {
      const region = plan.chunks[index];
      const started = performance.now();
      renderer.uploadScene(scene, {
        s,
        scale: raster.scale,
        origin: [
          raster.origin[0] + region.x * raster.scale,
          raster.origin[1] + region.y * raster.scale,
        ],
        bg: compositeBackground,
      });
      const image = await renderer.forward();
      blit(
        rgba,
        raster.width,
        region.x,
        region.y,
        image,
        plan.tileWidth,
        region.width,
        region.height,
        { transparent },
      );
      onChunk?.({ ...region, index, total: plan.chunks.length, ms: performance.now() - started });
    }
  } finally {
    renderer.destroy();
  }
  return { rgba, plan };
}
