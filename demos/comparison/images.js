// Grayscale renders, amplified error maps, and detail crops as RGBA8.

/** Default error-map gain: a coverage difference of 1/AMP already reads full-bright. */
export const AMP = 15;
export const WHITE = [1, 1, 1];

export const CORNERS = ['tr', 'tl', 'br', 'bl'];

const mapRGBA = (n, rgbAt) => {
  const d = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = rgbAt(i);
    const o = i * 4;
    d[o] = r;
    d[o + 1] = g;
    d[o + 2] = b;
    d[o + 3] = 255;
  }
  return d;
};

/** Coverage as opaque grayscale, white = covered. */
export const grayRGBA = (cov) =>
  mapRGBA(cov.length, (i) => {
    const v = Math.round(Math.max(0, Math.min(1, cov[i])) * 255);
    return [v, v, v];
  });

/** |a - b| x gain, clipped at 1. */
export const diffRGBA = (a, b, gain = AMP, tint = WHITE) =>
  mapRGBA(a.length, (i) => {
    const v = Math.round(Math.min(Math.abs(a[i] - b[i]) * gain, 1) * 255);
    return [Math.round(v * tint[0]), Math.round(v * tint[1]), Math.round(v * tint[2])];
  });

/** A cw x ch window of the nearest-neighbour z-times upscale. */
export function upscaleCrop(rgba, w, h, z, cx, cy, cw, ch) {
  const d = new Uint8Array(cw * ch * 4);
  for (let y = 0; y < ch; y++) {
    const sy = Math.min(h - 1, Math.max(0, ((cy + y) / z) | 0));
    for (let x = 0; x < cw; x++) {
      const sx = Math.min(w - 1, Math.max(0, ((cx + x) / z) | 0));
      const o = (y * cw + x) * 4, s = (sy * w + sx) * 4;
      d[o] = rgba[s];
      d[o + 1] = rgba[s + 1];
      d[o + 2] = rgba[s + 2];
      d[o + 3] = 255;
    }
  }
  return d;
}

/** The detail window, `inset` upscaled pixels inside the chosen corner. */
export function cropWindow(w, h, { zoom = 2, inset = 40, corner = 'tr' } = {}) {
  const W = w * zoom, H = h * zoom;
  const cw = w, ch = h;
  if (cw + inset > W || ch + inset > H) {
    throw new Error(`a ${cw}x${ch} crop inset ${inset}px does not fit a x${zoom} upscale of ${w}x${h}`);
  }
  const cx = corner === 'tr' || corner === 'br' ? W - cw - inset : inset;
  const cy = corner === 'tr' || corner === 'tl' ? inset : H - ch - inset;
  return { cx, cy, cw, ch, sx: cx / zoom, sy: cy / zoom, sw: cw / zoom, sh: ch / zoom };
}

/** Crop an RGBA8 image to its detail window; output keeps the source size. */
export function cropRGBA(rgba, w, h, opts = {}) {
  const win = cropWindow(w, h, opts);
  return upscaleCrop(rgba, w, h, opts.zoom ?? 2, win.cx, win.cy, win.cw, win.ch);
}
