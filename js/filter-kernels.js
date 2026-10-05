// Pixel filters: the analytic box of size s composed with per-axis taps
// (offset, weight) in units of s. Must match js/wgsl/coverage.wgsl and jax/oracle.py.

// 3-point Gauss-Legendre nodes/weights on [-1/2, 1/2].
export const GL3_NODE = 0.3872983346207417; // sqrt(3/5) / 2
export const GL3_EDGE = 0.2777777777777778; // 5/18
export const GL3_CENTER = 0.4444444444444444; // 8/18

// Weights of (GL3_EDGE z^-1 + GL3_CENTER + GL3_EDGE z)^3.
export const CUBIC_OUTER = 0.021433470507544582; // 125/5832
export const CUBIC_MID = 0.102880658436214; // 25/243
export const CUBIC_INNER = 0.22890946502057613; // 445/1944
export const CUBIC_CENTER = 0.2935528120713306; // 214/729

const freezeTaps = (...values) => Object.freeze(
  values.map((value) => Object.freeze(value)),
);

// radius: support half-width in units of s. boxPasses: box blurs for plot targets.
export const KERNELS = Object.freeze({
  box: Object.freeze({
    code: 0,
    radius: 0.5,
    boxPasses: 1,
    taps: freezeTaps([0, 1]),
  }),
  tent: Object.freeze({
    code: 1,
    radius: 1,
    boxPasses: 2,
    taps: freezeTaps(
      [-GL3_NODE, GL3_EDGE],
      [0, GL3_CENTER],
      [GL3_NODE, GL3_EDGE],
    ),
  }),
  cubic: Object.freeze({
    code: 2,
    radius: 2,
    boxPasses: 4,
    taps: freezeTaps(
      [-3 * GL3_NODE, CUBIC_OUTER],
      [-2 * GL3_NODE, CUBIC_MID],
      [-GL3_NODE, CUBIC_INNER],
      [0, CUBIC_CENTER],
      [GL3_NODE, CUBIC_INNER],
      [2 * GL3_NODE, CUBIC_MID],
      [3 * GL3_NODE, CUBIC_OUTER],
    ),
  }),
});

export function resolveKernel(name = 'box') {
  const kernel = Object.hasOwn(KERNELS, name) ? KERNELS[name] : null;
  if (!kernel) {
    throw new Error(
      `unknown filter kernel ${JSON.stringify(name)} (expected ${Object.keys(KERNELS).join(' or ')})`,
    );
  }
  return kernel;
}
