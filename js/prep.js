// Split quadratic curves into compact xy-monotone pieces for the GPU.

const lerp = (a, b, t) => a + (b - a) * t;

// Clamped parameter of the axis extremum -a1/(2 a2), guarded so flat axes
// resolve to t = 0.
function axisRoot(p0, p1, p2) {
  const a1 = 2 * (p1 - p0);
  const denom = 2 * (p0 - 2 * p1 + p2);
  const eps = 1e-12 * (Math.abs(a1) + Math.abs(denom)) + 1e-30;
  const d = Math.abs(denom) < eps ? (denom < 0 ? -eps : eps) : denom;
  return Math.min(Math.max(-a1 / d, 0), 1);
}

function ensureScratch(scratch, shapeCount, curveCount) {
  if (!scratch.pieceData || scratch.pieceData.length !== curveCount * 18) {
    scratch.pieceData = new Float32Array(curveCount * 18);
  }
  if (!scratch.shapeData || scratch.shapeData.length !== shapeCount * 16) {
    scratch.shapeData = new Float32Array(shapeCount * 16);
    scratch.shapeMeta = new Uint32Array(scratch.shapeData.buffer);
  } else if (!scratch.shapeMeta || scratch.shapeMeta.buffer !== scratch.shapeData.buffer) {
    scratch.shapeMeta = new Uint32Array(scratch.shapeData.buffer);
  }
  if (!scratch.curveMetaData || scratch.curveMetaData.length !== curveCount * 4) {
    scratch.curveMetaData = new Float32Array(curveCount * 4);
    scratch.curveMeta = new Uint32Array(scratch.curveMetaData.buffer);
  } else if (!scratch.curveMeta || scratch.curveMeta.buffer !== scratch.curveMetaData.buffer) {
    scratch.curveMeta = new Uint32Array(scratch.curveMetaData.buffer);
  }
}

function fillRuleCode(fillRule, shapeIndex) {
  const rule = fillRule === undefined ? 'nonzero' : fillRule;
  if (rule === 'nonzero') return 0;
  if (rule === 'evenodd') return 1;
  throw new Error(`shape ${shapeIndex}: fillRule must be "nonzero" or "evenodd"`);
}

// shapes: [{ curves: K*6 coordinates, color: [r,g,b], alpha, fillRule?, s? }]
// pieceData is scratch-capacity; the first pieceCount*6 values are live.
export function packScene(shapes, scratch = {}) {
  scratch ??= {};
  let curveCount = 0;
  for (const shape of shapes) {
    if (shape.curves.length % 6 !== 0) throw new Error('curve data must contain 6 values per curve');
    curveCount += shape.curves.length / 6;
  }

  ensureScratch(scratch, shapes.length, curveCount);
  const { pieceData, shapeData, shapeMeta, curveMetaData, curveMeta } = scratch;

  let pieceCount = 0;
  // Skip pieces that collapse to a single f32 point; the write already
  // rounded, so equality on the array slots is the bitwise criterion.
  const emit = (qx0, qy0, qx1, qy1, qx2, qy2) => {
    const o = 6 * pieceCount;
    pieceData[o] = qx0;
    pieceData[o + 1] = qy0;
    pieceData[o + 2] = qx1;
    pieceData[o + 3] = qy1;
    pieceData[o + 4] = qx2;
    pieceData[o + 5] = qy2;
    const point = pieceData[o + 2] === pieceData[o] && pieceData[o + 3] === pieceData[o + 1] &&
      pieceData[o + 4] === pieceData[o] && pieceData[o + 5] === pieceData[o + 1];
    if (point) return 0;
    pieceCount++;
    return 1;
  };

  let curve = 0;
  for (let shapeIndex = 0; shapeIndex < shapes.length; shapeIndex++) {
    const shape = shapes[shapeIndex];
    const curves = shape.curves;
    const curveLength = curves.length / 6;
    const shapePieceStart = pieceCount;
    let lox = Infinity, loy = Infinity, hix = -Infinity, hiy = -Infinity;

    for (let k = 0; k < curveLength; k++, curve++) {
      const co = 6 * k;
      const x0 = curves[co], y0 = curves[co + 1];
      const x1 = curves[co + 2], y1 = curves[co + 3];
      const x2 = curves[co + 4], y2 = curves[co + 5];

      const tx = axisRoot(x0, x1, x2);
      const ty = axisRoot(y0, y1, y2);
      const t1 = Math.min(tx, ty);
      const t2 = Math.max(tx, ty);
      const mo = 4 * curve;
      curveMetaData[mo] = t1;
      curveMetaData[mo + 1] = t2;
      curveMeta[mo + 2] = pieceCount;

      // Blossom points shared between adjacent pieces: A = C(0,t), B = C(t,1),
      // P(t) = lerp(A,B,t), C(t1,t2) = lerp(A2,B2,t1). Endpoints stay exact.
      const ax1 = lerp(x0, x1, t1), ay1 = lerp(y0, y1, t1);
      const bx1 = lerp(x1, x2, t1), by1 = lerp(y1, y2, t1);
      const px1 = lerp(ax1, bx1, t1), py1 = lerp(ay1, by1, t1);
      const ax2 = lerp(x0, x1, t2), ay2 = lerp(y0, y1, t2);
      const bx2 = lerp(x1, x2, t2), by2 = lerp(y1, y2, t2);
      const px2 = lerp(ax2, bx2, t2), py2 = lerp(ay2, by2, t2);

      let mask = 0;
      if (t1 > 0) mask |= emit(x0, y0, ax1, ay1, px1, py1);
      if (t2 > t1) {
        mask |= emit(px1, py1, lerp(ax2, bx2, t1), lerp(ay2, by2, t1), px2, py2) << 1;
      }
      if (t2 < 1) mask |= emit(px2, py2, bx2, by2, x2, y2) << 2;
      curveMeta[mo + 3] = mask;

      lox = Math.min(lox, x0, x1, x2);
      hix = Math.max(hix, x0, x1, x2);
      loy = Math.min(loy, y0, y1, y2);
      hiy = Math.max(hiy, y0, y1, y2);
    }

    const o = shapeIndex * 16;
    shapeData[o] = lox;
    shapeData[o + 1] = loy;
    shapeData[o + 2] = hix;
    shapeData[o + 3] = hiy;
    shapeData[o + 4] = shape.color[0];
    shapeData[o + 5] = shape.color[1];
    shapeData[o + 6] = shape.color[2];
    shapeData[o + 7] = shape.alpha ?? 1;
    shapeMeta[o + 8] = shapePieceStart;
    shapeMeta[o + 9] = pieceCount - shapePieceStart;
    shapeMeta[o + 10] = fillRuleCode(shape.fillRule, shapeIndex);
    shapeMeta[o + 11] = 0;
    shapeData[o + 12] = 0;
    shapeData[o + 13] = 0;
    if (shape.s != null) {
      const sx = Array.isArray(shape.s) ? shape.s[0] : shape.s;
      const sy = Array.isArray(shape.s) ? shape.s[1] : shape.s;
      if (!(sx > 0) || !(sy > 0)) throw new Error(`shape ${shapeIndex}: s must be positive`);
      shapeData[o + 12] = sx;
      shapeData[o + 13] = sy;
    }
  }

  return { pieceData, shapeData, pieceCount, curveCount, curveMetaData };
}
