// The rasterisers under comparison; each returns coverage in 0..1.

import { createCanvas } from '@napi-rs/canvas';

import { packScene } from '../../js/prep.js';
import { Renderer } from '../../js/renderer.js';

const to8 = (v) => Math.round(Math.max(0, Math.min(1, v)) * 255) / 255;

/** Windfoil, this repository's renderer: the analytic box filter at s = 1 px, white over black. */
export async function windfoilCoverage(device, quads, { size, evenodd = false }) {
  const scene = packScene([{
    curves: quads,
    color: [1, 1, 1],
    alpha: 1,
    fillRule: evenodd ? 'evenodd' : 'nonzero',
  }]);
  const renderer = await Renderer.create(device, {
    width: size,
    height: size,
    maxShapes: 1,
    maxPieces: scene.pieceCount,
    maxCurves: scene.curveCount,
    // Forward only: skip the geometry-gradient buffers.
    train: { geometry: false, colour: false, alpha: true },
  });
  try {
    renderer.uploadScene(scene, { s: [1, 1], scale: 1, origin: [0, 0], bg: [0, 0, 0] });
    const image = await renderer.forward();
    const cov = new Float64Array(size * size);
    for (let i = 0; i < cov.length; i++) cov[i] = to8(image[i * 4]);
    return cov;
  } finally {
    renderer.destroy();
  }
}

/** Skia through @napi-rs/canvas, the rasteriser Chrome ships: one path, filled white on black. */
export function skiaCoverage(quads, { size, evenodd = false }) {
  const ctx = createCanvas(size, size).getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  // A new subpath wherever an edge does not continue the previous one.
  let px = null, py = null;
  for (let i = 0; i < quads.length; i += 6) {
    const x0 = quads[i], y0 = quads[i + 1];
    if (px === null || Math.abs(x0 - px) > 1e-4 || Math.abs(y0 - py) > 1e-4) {
      if (px !== null) ctx.closePath();
      ctx.moveTo(x0, y0);
    }
    ctx.quadraticCurveTo(quads[i + 2], quads[i + 3], quads[i + 4], quads[i + 5]);
    px = quads[i + 4];
    py = quads[i + 5];
  }
  ctx.closePath();
  ctx.fill(evenodd ? 'evenodd' : 'nonzero');
  const rgba = ctx.getImageData(0, 0, size, size).data;
  const cov = new Float64Array(size * size);
  for (let i = 0; i < cov.length; i++) cov[i] = rgba[i * 4] / 255;
  return cov;
}
