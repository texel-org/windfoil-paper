// Reference coverage: the box filter, point-sampled with ray-cast winding.

// Crossings of a rightward ray at height py, written into caller scratch.
function crossingsAt(py, quads, xs, sg) {
  let n = 0;
  for (let i = 0; i < quads.length; i += 6) {
    const x0 = quads[i], y0 = quads[i + 1], cx = quads[i + 2], cy = quads[i + 3];
    const x1 = quads[i + 4], y1 = quads[i + 5];
    if ((y0 < py && cy < py && y1 < py) || (y0 > py && cy > py && y1 > py)) continue; // hull y-reject
    const a = y0 - 2 * cy + y1, b = 2 * (cy - y0), c = y0 - py;
    let t0 = -1, t1 = -1;
    if (Math.abs(a) < 1e-9) {
      if (Math.abs(b) > 1e-12) t0 = -c / b;
    } else {
      const disc = b * b - 4 * a * c;
      if (disc >= 0) {
        const sq = Math.sqrt(disc);
        t0 = (-b + sq) / (2 * a);
        t1 = (-b - sq) / (2 * a);
      }
    }
    for (let k = 0; k < 2; k++) {
      const t = k === 0 ? t0 : t1;
      if (t < 0 || t > 1) continue;
      const u = 1 - t;
      xs[n] = u * u * x0 + 2 * u * t * cx + t * t * x1;
      sg[n] = 2 * a * t + b >= 0 ? 1 : -1;
      n++;
    }
  }
  return n;
}

/** Winding number W and crossing count K at one point, by a rightward ray cast. */
export function windingAt(px, py, quads) {
  const cap = (quads.length / 6) * 2;
  const xs = new Float64Array(cap), sg = new Int8Array(cap);
  const n = crossingsAt(py, quads, xs, sg);
  let W = 0, K = 0;
  for (let i = 0; i < n; i++) {
    if (xs[i] > px) {
      W += sg[i];
      K++;
    }
  }
  return { W, K };
}

/** Fraction of an N x N grid of sub-samples per pixel inside the shape. */
export function pointCoverage(quads, { size, evenodd = false, samples = 24 }) {
  const N = samples;
  const out = new Float64Array(size * size);
  const cap = (quads.length / 6) * 2;
  const xs = new Float64Array(cap), sg = new Int8Array(cap);
  const order = new Uint32Array(cap);
  const sx = new Float64Array(cap), ss = new Int8Array(cap);
  const off = new Float64Array(N);
  for (let i = 0; i < N; i++) off[i] = (i + 0.5) / N;

  for (let y = 0; y < size; y++) {
    const row = y * size;
    for (let j = 0; j < N; j++) {
      const n = crossingsAt(y + off[j], quads, xs, sg);
      const idx = order.subarray(0, n);
      for (let i = 0; i < n; i++) idx[i] = i;
      idx.sort((p, q) => xs[q] - xs[p]);
      for (let i = 0; i < n; i++) {
        sx[i] = xs[idx[i]];
        ss[i] = sg[idx[i]];
      }
      let ptr = 0, W = 0, K = 0;
      for (let x = size - 1; x >= 0; x--) {
        const inside = evenodd ? (K & 1) === 1 : W !== 0;
        if (ptr >= n || sx[ptr] <= x + off[0]) {
          if (inside) out[row + x] += N;
          continue;
        }
        let hit = 0;
        for (let i = N - 1; i >= 0; i--) {
          const px = x + off[i];
          while (ptr < n && sx[ptr] > px) {
            W += ss[ptr];
            K++;
            ptr++;
          }
          if (evenodd ? (K & 1) === 1 : W !== 0) hit++;
        }
        out[row + x] += hit;
      }
    }
  }
  // Divide, not multiply by 1/(N*N), to match the original rounding.
  for (let i = 0; i < out.length; i++) out[i] /= N * N;
  return out;
}

/** Mean and worst-pixel |a - b| over two coverage buffers. */
export function coverageStats(a, b) {
  let sum = 0, max = 0;
  for (let i = 0; i < a.length; i++) {
    const e = Math.abs(a[i] - b[i]);
    sum += e;
    if (e > max) max = e;
  }
  return { mean: sum / a.length, max };
}
