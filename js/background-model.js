import { packScene } from './prep.js';

// Train the background color alongside the scene, with no renderer or shader
// change. The renderer composites `img = sum(shapes) + bg * P`, where P is the
// final transmittance, so dL/d(bg) = sum over pixels of dL/dimage * P. That is
// exactly the per-shape color gradient the renderer already computes for a
// full-canvas opaque shape drawn underneath everything (it sees transmittance P
// at every pixel). So this decorator prepends such a shape — geometry fixed,
// color trainable — and reads its existing color gradient as the bg gradient.
//
// The prepended shape is index 0 (drawn first / bottom), so it owns the first
// 4 curves and the first shape slot; the inner model's gradients are the rest.

const EDGE = 2; // extend past the canvas so anti-aliased borders still fill

function canvasQuad(width, height) {
  const pts = [
    [-EDGE, -EDGE],
    [width + EDGE, -EDGE],
    [width + EDGE, height + EDGE],
    [-EDGE, height + EDGE],
  ];
  const curves = new Float64Array(24); // 4 straight edges as degenerate quads
  for (let i = 0; i < 4; i++) {
    const [x0, y0] = pts[i];
    const [x1, y1] = pts[(i + 1) % 4];
    const o = 6 * i;
    curves[o] = x0;
    curves[o + 1] = y0;
    curves[o + 2] = 0.5 * (x0 + x1);
    curves[o + 3] = 0.5 * (y0 + y1);
    curves[o + 4] = x1;
    curves[o + 5] = y1;
  }
  return curves;
}

const clamp01 = (v) => Math.max(0, Math.min(1, v));

export function withBackgroundColor(inner, lrs, { width, height, background, lr = 0.01 }) {
  const bg = Float32Array.from([
    background[0] ?? 0,
    background[1] ?? 0,
    background[2] ?? 0,
  ]);
  const bgShape = { curves: canvasQuad(width, height), color: [0, 0, 0], alpha: 1 };
  const scratch = {};
  const model = {
    inner,
    // The full-canvas background shape is prepended at index 0; consumers that
    // render the background by other means (e.g. the SVG's <rect>) can skip it.
    prependedShapes: 1,
    params: { ...inner.params, bg },
    n: inner.n + 1,
    maxShapes: inner.maxShapes + 1,
    maxCurves: inner.maxCurves + 4,
    maxPieces: (inner.maxCurves + 4) * 3,

    decode() {
      const decoded = inner.decode();
      // Keep the trained background in sRGB gamut. Shape colors are in-gamut by
      // construction (convex combinations of in-gamut anchors), but the raw bg
      // parameter has no such guard: the renderer will happily composite an
      // out-of-gamut color (e.g. a negative channel), and the raster fit can
      // exploit that as a free global tint — yet the SVG/PNG export must clamp
      // it, so the exported vector diverges from the optimized raster. Project
      // the parameter back into [0,1] each step so what we fit is what we emit.
      bg[0] = clamp01(bg[0]);
      bg[1] = clamp01(bg[1]);
      bg[2] = clamp01(bg[2]);
      bgShape.color[0] = bg[0];
      bgShape.color[1] = bg[1];
      bgShape.color[2] = bg[2];
      const shapes = [bgShape, ...decoded.shapes];
      return { shapes, scene: packScene(shapes, scratch) };
    },

    pullback(result) {
      const { curveGrads, shapeGrads, blurGrads } = result;
      // Every gradient family is offset by the prepended shape, each by its own
      // stride: 4 curves (24 coordinates), one stride-4 shape slot, and one
      // (dsx, dsy) blur pair. A renderer with a frozen group omits its array
      // entirely, so forward the same absence inward rather than indexing it.
      const innerGrads = inner.pullback({
        ...(curveGrads === undefined ? {} : { curveGrads: curveGrads.subarray(24) }),
        ...(shapeGrads === undefined ? {} : { shapeGrads: shapeGrads.subarray(4) }),
        ...(blurGrads === undefined ? {} : { blurGrads: blurGrads.subarray(2) }),
      });
      // The bg shape's color gradient is dL/d(bg); its alpha gradient is unused.
      const bgGrad = shapeGrads
        ? Float32Array.of(shapeGrads[0], shapeGrads[1], shapeGrads[2])
        : new Float32Array(3);
      return { ...innerGrads, bg: bgGrad };
    },

    harden() {
      inner.harden?.();
    },

    // The trained background color (for the final render, SVG, and report).
    background() {
      return [clamp01(bg[0]), clamp01(bg[1]), clamp01(bg[2])];
    },
  };
  return { model, lrs: { ...lrs, bg: { lr } } };
}
