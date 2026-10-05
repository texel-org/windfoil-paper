import { existsSync } from 'node:fs';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { packScene } from '../../js/prep.js';
import { Renderer, requestDevice } from '../../js/renderer.js';
import { tonemapChain, tonemapImage } from '../../js/tonemap.js';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const WIDTH = 14;
const HEIGHT = 12;

function loop(anchors, controls) {
  return anchors.flatMap((anchor, i) => [
    ...anchor,
    ...controls[i],
    ...anchors[(i + 1) % anchors.length],
  ]);
}

function cotangent() {
  const values = [];
  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      for (let c = 0; c < 3; c++) {
        values.push(0.011 * (
          Math.sin(0.37 * (x + 1) + 0.19 * (y + 1) + 0.61 * c) +
          0.31 * Math.cos(0.13 * (x + 2 * y + 3 * c))
        ));
      }
    }
  }
  return values;
}

function nonzeroCase() {
  const curves = [
    loop(
      [[1.83, 4.17], [5.48, 1.61], [10.72, 3.86], [9.64, 9.79], [3.21, 9.14]],
      [[2.91, 1.92], [8.57, 1.38], [11.94, 6.71], [6.47, 10.83], [1.29, 6.88]],
    ),
    loop(
      [[4.13, 5.31], [7.92, 2.48], [12.16, 4.74], [11.38, 9.37], [6.06, 10.21]],
      [[5.47, 2.91], [10.63, 2.29], [13.06, 7.17], [8.71, 10.79], [3.72, 7.89]],
    ),
  ];
  return {
    name: 'nonzero',
    width: WIDTH,
    height: HEIGHT,
    settings: { s: [15, 13], scale: 1, origin: [0, 0], bg: [0.93, 0.89, 0.84] },
    curves,
    colors: [[0.81, 0.24, 0.19], [0.12, 0.56, 0.78]],
    alphas: [0.73, 0.82],
    fillRules: ['nonzero', 'nonzero'],
    cotangent: cotangent(),
  };
}

function pentagram(cx, cy, radius, rotation = -Math.PI / 2) {
  const outer = Array.from({ length: 5 }, (_, i) => {
    const angle = rotation + i * 2 * Math.PI / 5;
    return [cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)];
  });
  const anchors = [0, 2, 4, 1, 3].map((index) => outer[index]);
  const controls = anchors.map((anchor, i) => {
    const next = anchors[(i + 1) % anchors.length];
    return [0.5 * (anchor[0] + next[0]), 0.5 * (anchor[1] + next[1])];
  });
  return loop(anchors, controls);
}

function rectangle(x0, y0, x1, y1) {
  const anchors = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  const controls = anchors.map((anchor, i) => {
    const next = anchors[(i + 1) % anchors.length];
    return [0.5 * (anchor[0] + next[0]), 0.5 * (anchor[1] + next[1])];
  });
  return loop(anchors, controls);
}

function evenoddCase() {
  return {
    name: 'evenodd',
    width: WIDTH,
    height: HEIGHT,
    settings: { s: [2.7, 2.3], scale: 1, origin: [0, 0], bg: [0.93, 0.89, 0.84] },
    curves: [
      pentagram(6.7, 6.1, 5.0),
      rectangle(4.4, 3.2, 11.3, 9.7),
    ],
    colors: [[0.81, 0.24, 0.19], [0.12, 0.56, 0.78]],
    alphas: [0.73, 0.82],
    fillRules: ['evenodd', 'nonzero'],
    cotangent: cotangent(),
  };
}

// A 0.001-unit edge at x=335 is a real f32 segment, not a degenerate point.
// The former coordinate-relative gate discarded it, opening the contour and
// leaving a faint horizontal winding residue. The direct JAX unit test anchors
// the expected area; this case ensures WebGPU follows the corrected oracle in
// both the forward image and VJP.
function microEdgeCase() {
  const anchors = [[330, 2], [335, 2], [335, 7.999], [335, 8], [330, 8]];
  const controls = anchors.map((anchor, i) => {
    const next = anchors[(i + 1) % anchors.length];
    return [0.5 * (anchor[0] + next[0]), 0.5 * (anchor[1] + next[1])];
  });
  return {
    name: 'micro-edge',
    width: WIDTH,
    height: HEIGHT,
    settings: { s: [1, 1], scale: 1, origin: [329, 0], bg: [0, 0, 0] },
    curves: [loop(anchors, controls)],
    colors: [[1, 1, 1]],
    alphas: [1],
    fillRules: ['nonzero'],
    cotangent: cotangent(),
  };
}

// The order-independent blends reuse the overlapping nonzero geometry, which
// exercises the cross-shape terms (add has none; multiply's sibling products).
// add gets a dark background so the unclamped sum stays near [0, 1].
function blendCase(blend, bg) {
  const base = nonzeroCase();
  return {
    ...base,
    name: blend,
    blend,
    settings: { ...base.settings, bg },
  };
}

// Per-shape filters with blur training on: the GPU's filter-size adjoints
// (the Euler-identity contraction of the piece gradients) must match JAX
// differentiating the same render through shape_s.
function blurCase() {
  const base = nonzeroCase();
  return {
    ...base,
    name: 'blur',
    shapeS: [[3.1, 2.6], [2.2, 4.0]],
    train: { blur: true },
  };
}

function kernelCase(kernel, s) {
  const base = nonzeroCase();
  return {
    ...base,
    name: kernel,
    settings: { ...base.settings, s, kernel },
  };
}

function kernelBlurCase(kernel) {
  const base = blurCase();
  return {
    ...base,
    name: `blur-${kernel}`,
    settings: { ...base.settings, kernel },
  };
}

// HDR linear light through a tonemap operator: the host chains the
// display-space cotangent back to linear (tonemapChain) exactly as the
// CLIP path does; JAX differentiates the tonemapped render directly,
// including d/dk (and d/dW for the white-normalized operators). Values
// above 1 come from the add-mode sum, not clamping.
function tonemapCase(tonemap = 'reinhard', extra = {}) {
  const base = nonzeroCase();
  return {
    ...base,
    name: `tonemap-${tonemap}`,
    blend: 'add',
    settings: { ...base.settings, bg: [0.07, 0.11, 0.16] },
    tonemap,
    exposure: 1.7,
    ...extra,
  };
}

// The signed domain of 'smooth': one shape carries a negative color (add mode
// subtracts its light), so linear values cross zero under the odd operator --
// the case the Reinhard pair cannot express at all.
function tonemapSignedCase() {
  const base = tonemapCase('smooth', { white: 2.3 });
  return {
    ...base,
    name: 'tonemap-signed',
    colorRange: 'signed',
    colors: [[0.81, -0.44, 0.19], [-0.22, 0.56, 0.78]],
  };
}

function rgba(rgb, alpha = 0) {
  const out = new Float32Array((rgb.length / 3) * 4);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    out[j] = rgb[i];
    out[j + 1] = rgb[i + 1];
    out[j + 2] = rgb[i + 2];
    out[j + 3] = alpha;
  }
  return out;
}

function rgb(rgbaValues) {
  const out = new Array((rgbaValues.length / 4) * 3);
  for (let i = 0, j = 0; i < rgbaValues.length; i += 4, j += 3) {
    out[j] = rgbaValues[i];
    out[j + 1] = rgbaValues[i + 1];
    out[j + 2] = rgbaValues[i + 2];
  }
  return out;
}

function stats(actual, expected) {
  if (actual.length !== expected.length) throw new Error(`length mismatch: ${actual.length} != ${expected.length}`);
  let error2 = 0;
  let reference2 = 0;
  let maxAbs = 0;
  let referenceMax = 0;
  for (let i = 0; i < actual.length; i++) {
    const error = actual[i] - expected[i];
    if (!Number.isFinite(error)) throw new Error(`non-finite value at ${i}`);
    error2 += error * error;
    reference2 += expected[i] * expected[i];
    maxAbs = Math.max(maxAbs, Math.abs(error));
    referenceMax = Math.max(referenceMax, Math.abs(expected[i]));
  }
  return {
    maxAbs,
    relL2: Math.sqrt(error2) / Math.max(Math.sqrt(reference2), 1e-30),
    peakRelative: maxAbs / Math.max(referenceMax, 1e-30),
  };
}

// Contract per-curve gradients to the per-anchor/control gradients a model
// actually consumes. Both consumers do this: LoopModel.pullback adds curve c's
// b0 to curve c-1's b2, and GenericModel accumulates both into one tape node.
//
// A closed loop's curves share their end anchors, and only the *total*
// derivative at a shared point is determined: how it is split between curve c's
// b0 and curve c-1's b2 is an attribution choice. Where a control point lands on
// an exact coordinate tie -- as the pentagram's do -- WebGPU's strict piece
// culls and JAX's formulation attribute the boundary contribution to different
// sides of the anchor. The split differs, the sum does not, and the sum is what
// reaches the optimizer. Comparing raw per-curve values would fail on that gauge
// freedom while both implementations are correct.
function contractLoops(curveGrads, testCase) {
  const out = [];
  let base = 0;
  for (const curves of testCase.curves) {
    const k = curves.length / 6;
    for (let j = 0; j < k; j++) {
      const at = 6 * (base + j);
      const previous = 6 * (base + ((j + k - 1) % k));
      out.push(
        curveGrads[at] + curveGrads[previous + 4],       // anchor x
        curveGrads[at + 1] + curveGrads[previous + 5],   // anchor y
        curveGrads[at + 2],                              // control x
        curveGrads[at + 3],                              // control y
      );
    }
    base += k;
  }
  return out;
}

function compare(name, actual, expected, limits) {
  const value = stats(actual, expected);
  console.log(`${name.padEnd(8)} max ${value.maxAbs.toExponential(3)}  rel L2 ${value.relL2.toExponential(3)}`);
  for (const [metric, limit] of Object.entries(limits)) {
    if (value[metric] > limit) throw new Error(`${name} ${metric} ${value[metric]} > ${limit}`);
  }
}

async function gpuResult(testCase) {
  const shapes = testCase.curves.map((curves, i) => ({
    curves: Float64Array.from(curves),
    color: testCase.colors[i],
    alpha: testCase.alphas[i],
    fillRule: testCase.fillRules[i],
    s: testCase.shapeS?.[i],
  }));
  const scene = packScene(shapes);
  const device = await requestDevice();
  let renderer;
  try {
    renderer = await Renderer.create(device, {
      width: testCase.width,
      height: testCase.height,
      maxPieces: scene.pieceCount,
      maxShapes: shapes.length,
      maxCurves: scene.curveCount,
      blend: testCase.blend,
      train: testCase.train,
      colorRange: testCase.colorRange,
    });
    renderer.uploadScene(scene, testCase.settings);
    const linear = await renderer.forward();
    const cotangent = rgba(testCase.cotangent);
    let image = linear;
    let kGrad = null;
    let wGrad = null;
    if (testCase.tonemap) {
      image = tonemapImage(testCase.tonemap, linear, testCase.exposure, testCase.white ?? null);
      const chain = tonemapChain(
        testCase.tonemap, linear, testCase.exposure, testCase.white ?? null, cotangent);
      kGrad = chain.kGrad;
      wGrad = chain.wGrad;
    }
    const gradients = await renderer.backward(cotangent);
    const shape = Array.from(gradients.shapeGrads);
    return {
      image: rgb(image),
      curveGrads: Array.from(gradients.curveGrads),
      colorGrads: shape.flatMap((_, i) => i % 4 === 0 ? shape.slice(i, i + 3) : []),
      alphaGrads: shape.filter((_, i) => i % 4 === 3),
      blurGrads: gradients.blurGrads ? Array.from(gradients.blurGrads) : null,
      kGrad,
      wGrad,
    };
  } finally {
    renderer?.destroy();
    device.destroy();
  }
}

async function main() {
  const requested = process.argv[2];
  const temporary = !requested;
  const workDir = requested
    ? resolve(requested)
    : await mkdtemp(join(tmpdir(), 'windfoil-oracle-'));
  await mkdir(workDir, { recursive: true });
  try {
    const venvPython = join(ROOT, '.venv', 'bin', 'python');
    const python = process.env.PYTHON ?? (existsSync(venvPython) ? venvPython : 'python3');
    const script = fileURLToPath(new URL('./jax_reference.py', import.meta.url));
    const cases = [
      nonzeroCase(),
      microEdgeCase(),
      evenoddCase(),
      blendCase('add', [0.07, 0.11, 0.16]),
      blendCase('multiply', [0.93, 0.89, 0.84]),
      blendCase('screen', [0.07, 0.11, 0.16]),
      blurCase(),
      kernelCase('tent', [7.5, 6.5]),
      kernelCase('cubic', [5.5, 4.5]),
      kernelBlurCase('tent'),
      kernelBlurCase('cubic'),
      tonemapCase(),
      tonemapCase('reinhard-white', { white: 2.3 }),
      tonemapCase('smooth', { white: 2.3 }),
      tonemapSignedCase(),
    ];
    for (const testCase of cases) {
      console.log(`${testCase.name} ${testCase.blend ? 'blend' : 'fill-rule'} parity`);
      const exchangePath = join(workDir, `webgpu-${testCase.name}.json`);
      const referencePath = join(workDir, `jax-${testCase.name}.json`);
      const gpu = await gpuResult(testCase);
      await writeFile(exchangePath, JSON.stringify({ ...testCase, gpu }));
      const child = spawnSync(python, [script, exchangePath, referencePath], { stdio: 'inherit' });
      if (child.error) throw child.error;
      if (child.status !== 0) throw new Error(`JAX oracle exited with status ${child.status}`);
      const reference = JSON.parse(await readFile(referencePath, 'utf8'));

      compare('image', gpu.image, reference.image, { maxAbs: 6e-4 });
      compare('curves', contractLoops(gpu.curveGrads, testCase),
        contractLoops(reference.curveGrads, testCase), { relL2: 7e-3, peakRelative: 2e-2 });
      compare('colors', gpu.colorGrads, reference.colorGrads, { relL2: 4e-3, peakRelative: 1e-2 });
      compare('alphas', gpu.alphaGrads, reference.alphaGrads, { relL2: 4e-3, peakRelative: 1e-2 });
      if (testCase.shapeS) {
        compare('blur', gpu.blurGrads, reference.sGrads, { relL2: 4e-3, peakRelative: 1e-2 });
      }
      if (testCase.tonemap) {
        compare('kgrad', [gpu.kGrad], [reference.kGrad], { relL2: 1e-3 });
      }
      if (testCase.white) {
        compare('wgrad', [gpu.wGrad], [reference.wGrad], { relL2: 1e-3 });
      }
    }
    console.log('fill-rule, blend-mode, and filter-kernel oracle parity ok');
  } catch (error) {
    if (temporary) console.error(`oracle files kept at ${workDir}`);
    throw error;
  }

  if (temporary) await rm(workDir, { recursive: true });
}

try {
  await main();
  process.exit(0);
} catch (error) {
  console.error(error?.stack ?? error);
  process.exit(1);
}
