// Raw linear color+alpha codec: each shape's color is its own parameter,
// mapped through an elementwise transfer whose codomain *is* the range
// declaration -- identity (signed linear light), softplus (nonnegative HDR),
// or sigmoid (bounded unit, the backward-compatible baseline). No softmax, no
// gamut projection: the composite is linear in the identity-transfer
// parameters, so with an L2 loss (and any C-infinity tonemap) the color
// subproblem is exactly convex and smooth end to end.
//
// The renderer validates a codec's `range` against the blend mode
// (multiply/screen need 'unit') and the tonemap domain (the Reinhard pair
// needs nonnegative light), so invalid combinations are rejected at
// construction instead of guarded at run time.
//
// Alpha is a fixed constant by default (add-mode fits train the premultiplied
// quantity directly and the renderer freezes its alpha pipeline); 'learned'
// opts into a sigmoid-parameterized opacity with the same 0.999 ceiling as
// the anchor codec.

const ALPHA_MAX = 0.999;
// Init/write-only floor keeping inverse transforms inside an open codomain;
// decode and pullback never touch it.
const CODOMAIN_FLOOR = 1e-3;

function sigmoid(t) {
  if (t >= 0) return 1 / (1 + Math.exp(-t));
  const e = Math.exp(t);
  return e / (1 + e);
}

const logit = (c) => Math.log(c / (1 - c));

const RAW_TRANSFERS = Object.freeze({
  identity: {
    range: 'signed',
    forward: (t) => t,
    derivative: () => 1,
    inverse: (c) => c,
    clampToDomain: (c) => c,
  },
  softplus: {
    range: 'nonneg',
    // max(t,0) + log1p(exp(-|t|)) is the overflow-safe softplus.
    forward: (t) => Math.max(t, 0) + Math.log1p(Math.exp(-Math.abs(t))),
    derivative: (t) => sigmoid(t),
    // softplus^-1(c) = log(e^c - 1) = c + log1p(-exp(-c)), stable for all c > 0.
    inverse: (c) => c + Math.log1p(-Math.exp(-c)),
    clampToDomain: (c) => Math.max(c, CODOMAIN_FLOOR),
  },
  sigmoid: {
    range: 'unit',
    forward: sigmoid,
    derivative: (t) => {
      const s = sigmoid(t);
      return s * (1 - s);
    },
    inverse: logit,
    clampToDomain: (c) => Math.min(Math.max(c, CODOMAIN_FLOOR), 1 - CODOMAIN_FLOOR),
  },
});

// Learned opacity a = ALPHA_MAX * sigmoid(theta); the exact inverse of a
// target fraction f = a / ALPHA_MAX, floored like the color codomains.
function alphaParam(fraction) {
  return logit(Math.min(Math.max(fraction, CODOMAIN_FLOOR), 1 - CODOMAIN_FLOOR));
}

export function rawStyle({
  channels = 'gray',
  transfer = 'identity',
  alpha = 1,
  lr = 0.05,
  alphaLr = 0.05,
} = {}) {
  const tf = RAW_TRANSFERS[transfer];
  if (!tf) {
    throw new Error(`transfer must be one of ${Object.keys(RAW_TRANSFERS).join(', ')}`);
  }
  if (channels !== 'gray' && channels !== 'rgb') {
    throw new Error("channels must be 'gray' or 'rgb'");
  }
  const learnedAlpha = alpha === 'learned';
  if (!learnedAlpha && (typeof alpha !== 'number' || !(alpha > 0) || !(alpha <= 1))) {
    throw new Error("alpha must be 'learned' or a fixed opacity in (0, 1]");
  }
  const C = channels === 'gray' ? 1 : 3;
  const project = channels === 'gray'
    // decode expands gray to (g, g, g); the L2-nearest gray to an rgb target
    // is the channel mean, so init/write project through it.
    ? (rgb) => [(rgb[0] + rgb[1] + rgb[2]) / 3]
    : (rgb) => rgb;

  return {
    kind: 'raw',
    range: tf.range,
    channels,
    transfer,
    // A fixed alpha never creates the parameter group, so callers can build
    // the renderer with train: { alpha: false } and drop that pipeline.
    trainsAlpha: learnedAlpha,
    fixedAlpha: learnedAlpha ? null : alpha,
    groups: learnedAlpha ? ['rawColor', 'rawAlpha'] : ['rawColor'],
    lrs: learnedAlpha
      ? { rawColor: { lr }, rawAlpha: { lr: alphaLr } }
      : { rawColor: { lr } },

    init(colors, alphas) {
      const rawColor = new Float32Array(C * colors.length);
      for (let i = 0; i < colors.length; i++) {
        const c = project(colors[i]);
        for (let ch = 0; ch < C; ch++) {
          rawColor[C * i + ch] = tf.inverse(tf.clampToDomain(c[ch]));
        }
      }
      if (!learnedAlpha) return { rawColor };
      const rawAlpha = new Float32Array(colors.length);
      for (let i = 0; i < colors.length; i++) {
        rawAlpha[i] = alphaParam((alphas?.[i] ?? 0.9) / ALPHA_MAX);
      }
      return { rawColor, rawAlpha };
    },

    decode(params, colorIndex, alphaIndex, outColor) {
      const at = C * colorIndex;
      if (C === 1) {
        const g = tf.forward(params.rawColor[at]);
        outColor[0] = g;
        outColor[1] = g;
        outColor[2] = g;
      } else {
        outColor[0] = tf.forward(params.rawColor[at]);
        outColor[1] = tf.forward(params.rawColor[at + 1]);
        outColor[2] = tf.forward(params.rawColor[at + 2]);
      }
      return learnedAlpha ? ALPHA_MAX * sigmoid(params.rawAlpha[alphaIndex]) : alpha;
    },

    // cot[o..o+3] = (dColor, dAlpha) from the renderer; the transfer chain is
    // elementwise, and gray contracts the three color cotangents exactly.
    pullback(params, colorIndex, alphaIndex, cot, o, grads) {
      const at = C * colorIndex;
      if (C === 1) {
        grads.rawColor[at] +=
          (cot[o] + cot[o + 1] + cot[o + 2]) * tf.derivative(params.rawColor[at]);
      } else {
        grads.rawColor[at] += cot[o] * tf.derivative(params.rawColor[at]);
        grads.rawColor[at + 1] += cot[o + 1] * tf.derivative(params.rawColor[at + 1]);
        grads.rawColor[at + 2] += cot[o + 2] * tf.derivative(params.rawColor[at + 2]);
      }
      if (learnedAlpha) {
        const s = sigmoid(params.rawAlpha[alphaIndex]);
        grads.rawAlpha[alphaIndex] += cot[o + 3] * ALPHA_MAX * s * (1 - s);
      }
    },
  };
}
