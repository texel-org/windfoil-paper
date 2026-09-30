// Tone mapping operators and their chain rules, shared by every host-side
// consumer (the CLIP loss boundary, PNG export, the oracle harness) so
// display-space semantics match the fused L2 kernel bit-for-bit in intent:
// the scene composites in linear light; the operator is applied only where an
// image is compared or shown.
//
// The family, T(W) = 1 for the white-point pair and T -> x/W as k -> 0:
//   reinhard        T(x) = k*x / (1 + k*x)                     x >= 0 only
//   reinhard-white  T(x) = R(k*x) / R(k*W), R(u) = u/(1+u)     x >= 0 only
//   smooth          T(x) = s(k*x) / s(k*W), s(u) = u/sqrt(1+u^2)   all of R
// The Reinhard pair has a pole at x = -1/k, so its domain is nonnegative
// linear light; 'smooth' is odd and C-infinity everywhere, the operator for
// signed (raw identity-transfer) scenes.

export const TONEMAP_MODES = Object.freeze({
  'none': 0,
  'reinhard': 1,
  'reinhard-white': 2,
  'smooth': 3,
});

// Operators with a learnable white point W (display reaches 1 exactly at W).
export const TONEMAP_HAS_WHITE = Object.freeze({
  'reinhard-white': true,
  'smooth': true,
});

// The linear-light domain each operator accepts; the renderer refuses signed
// color codecs under a nonneg operator so the Reinhard pole stays unreachable
// by construction.
export const TONEMAP_DOMAINS = Object.freeze({
  'none': 'signed',
  'reinhard': 'nonneg',
  'reinhard-white': 'nonneg',
  'smooth': 'signed',
});

function checkScalars(mode, k, w) {
  if (mode === 'none' || !(mode in TONEMAP_MODES)) {
    throw new Error(`no tonemap operator named '${mode}'`);
  }
  if (!(k > 0) || !Number.isFinite(k)) throw new Error('exposure must be positive and finite');
  if (TONEMAP_HAS_WHITE[mode] && (!(w > 0) || !Number.isFinite(w))) {
    throw new Error('white point must be positive and finite');
  }
}

// Map a linear RGBA image to display space; alpha passes through.
export function tonemapImage(mode, linear, k, w = null, out = new Float32Array(linear.length)) {
  checkScalars(mode, k, w);
  if (mode === 'reinhard') {
    for (let i = 0; i < linear.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const scaled = k * linear[i + c];
        out[i + c] = scaled / (1 + scaled);
      }
      out[i + 3] = linear[i + 3];
    }
  } else if (mode === 'reinhard-white') {
    const kw = k * w;
    const norm = (1 + kw) / kw;
    for (let i = 0; i < linear.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const scaled = k * linear[i + c];
        out[i + c] = norm * scaled / (1 + scaled);
      }
      out[i + 3] = linear[i + 3];
    }
  } else {
    const v = k * w;
    const sv = v / Math.sqrt(1 + v * v);
    for (let i = 0; i < linear.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const u = k * linear[i + c];
        out[i + c] = u / (Math.sqrt(1 + u * u) * sv);
      }
      out[i + 3] = linear[i + 3];
    }
  }
  return out;
}

// Chain a display-space cotangent back to linear light, in place, and return
// { kGrad, wGrad } (wGrad is null for operators without a white point).
// dLdDisplay alpha lanes are zeroed: the tonemap never touches alpha and the
// composite backward ignores it.
export function tonemapChain(mode, linear, k, w, dLdDisplay) {
  checkScalars(mode, k, w);
  if (dLdDisplay.length !== linear.length) throw new Error('cotangent/image size mismatch');
  let kGrad = 0;
  let wGrad = 0;
  if (mode === 'reinhard') {
    for (let i = 0; i < linear.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const x = linear[i + c];
        const inv = 1 / (1 + k * x);
        const cot = dLdDisplay[i + c];
        dLdDisplay[i + c] = cot * k * inv * inv;
        kGrad += cot * x * inv * inv;
      }
      dLdDisplay[i + 3] = 0;
    }
    return { kGrad, wGrad: null };
  }
  if (mode === 'reinhard-white') {
    const kw = k * w;
    const norm = (1 + kw) / kw;
    for (let i = 0; i < linear.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const x = linear[i + c];
        const inv = 1 / (1 + k * x);
        const r = k * x * inv;
        const cot = dLdDisplay[i + c];
        dLdDisplay[i + c] = cot * norm * k * inv * inv;
        // d(norm)/dk = -1/(k^2 W) and d(norm)/dW = -1/(k W^2), so
        // dT/dk = norm * x * inv^2 - r/(k*kw), dT/dW = -r/(kw*W).
        kGrad += cot * (norm * x * inv * inv - r / (k * kw));
        wGrad -= cot * r / (kw * w);
      }
      dLdDisplay[i + 3] = 0;
    }
    return { kGrad, wGrad };
  }
  const v = k * w;
  const iv = 1 / Math.sqrt(1 + v * v);
  const sv = v * iv;
  const spv = iv * iv * iv; // s'(k*W)
  for (let i = 0; i < linear.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const x = linear[i + c];
      const u = k * x;
      const iu = 1 / Math.sqrt(1 + u * u);
      const spu = iu * iu * iu; // s'(k*x)
      const t = u * iu / sv;
      const cot = dLdDisplay[i + c];
      dLdDisplay[i + c] = cot * k * spu / sv;
      kGrad += cot * (x * spu - t * w * spv) / sv;
      wGrad -= cot * t * k * spv / sv;
    }
    dLdDisplay[i + 3] = 0;
  }
  return { kGrad, wGrad };
}
