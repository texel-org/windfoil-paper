import { optimize } from '../../js/optimize.js';
import { packScene } from '../../js/prep.js';
import { getWebGPUHostInfo, Renderer, requestDevice } from '../../js/renderer.js';
import { parseFillSvg, rasterSize } from '../render/svg.js';
import { rgbToHex } from '../util/color.js';
import { imageToPng, l2Quality } from '../util/image.js';
import { annealedBlur } from '../util/model.js';
import { runPath } from '../util/output.js';
import { formatLoss, mkdir, readText, runMain, writeBytes, writeText } from '../util/runtime.js';
import { sceneToSVG } from '../util/svg.js';
import { buildRoundtripModel } from './model.js';

export const ROUNDTRIP_DEFAULTS = Object.freeze({
  svg: 'demos/roundtrip/star-evenodd.svg',
  optSize: 512,
  steps: 100,
  blur: 1,
  offset: null,
});

const OPTION_NAMES = new Set([
  'svg', 'opt-size', 'steps', 'blur', 'blur-start', 'offset', 'save-every', 'out', 'quiet',
]);

function positiveInteger(value, name) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < 1) {
    throw new Error(`--${name} must be a positive integer`);
  }
  return number;
}

function positiveNumber(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) {
    throw new Error(`--${name} must be a positive number`);
  }
  return number;
}

function requiredString(value, name) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`--${name} requires a value`);
  }
  return value;
}

export function parseOffset(value) {
  if (typeof value !== 'string') throw new Error('--offset requires x,y');
  const parts = value.split(',').map((part) => part.trim());
  if (parts.length !== 2 || parts.some((part) => part.length === 0)) {
    throw new Error('--offset must be two comma-separated numbers: x,y');
  }
  const offset = parts.map(Number);
  if (!offset.every(Number.isFinite)) {
    throw new Error('--offset must be two finite numbers: x,y');
  }
  return offset;
}

/** Strictly parse the small roundtrip CLI surface. */
export function parseRoundtripArgs(argv) {
  const raw = new Map();
  for (let at = 0; at < argv.length; at++) {
    const token = argv[at];
    if (typeof token !== 'string' || !token.startsWith('--') || token === '--') {
      throw new Error(`unexpected argument ${JSON.stringify(token)}`);
    }
    const equals = token.indexOf('=');
    const name = token.slice(2, equals < 0 ? undefined : equals);
    if (!OPTION_NAMES.has(name)) throw new Error(`unknown option --${name}`);
    if (name === 'quiet') {
      if (equals >= 0) throw new Error('--quiet does not take a value');
      raw.set(name, true);
      continue;
    }
    const value = equals >= 0 ? token.slice(equals + 1) : argv[++at];
    if (value === undefined || (equals < 0 && value.startsWith('--'))) {
      throw new Error(`--${name} requires a value`);
    }
    raw.set(name, requiredString(value, name));
  }

  // The same filter flags as the other demos: --blur throughout, or a wider
  // --blur-start that eases down to it.
  const blur = raw.has('blur')
    ? positiveNumber(raw.get('blur'), 'blur')
    : ROUNDTRIP_DEFAULTS.blur;
  const blurStart = raw.has('blur-start')
    ? positiveNumber(raw.get('blur-start'), 'blur-start')
    : blur;
  if (blurStart < blur) throw new Error('--blur-start must be at least --blur');

  return {
    svg: raw.has('svg') ? raw.get('svg') : ROUNDTRIP_DEFAULTS.svg,
    optSize: raw.has('opt-size')
      ? positiveInteger(raw.get('opt-size'), 'opt-size')
      : ROUNDTRIP_DEFAULTS.optSize,
    steps: raw.has('steps')
      ? positiveInteger(raw.get('steps'), 'steps')
      : ROUNDTRIP_DEFAULTS.steps,
    blur,
    blurStart,
    offset: raw.has('offset') ? parseOffset(raw.get('offset')) : ROUNDTRIP_DEFAULTS.offset,
    saveEvery: raw.has('save-every')
      ? positiveInteger(raw.get('save-every'), 'save-every')
      : null,
    out: raw.has('out') ? raw.get('out') : null,
    quiet: raw.has('quiet'),
  };
}

export function resolveRoundtripOffset(offset, viewBox) {
  if (offset) return Array.from(offset);
  if (!viewBox || !Number.isFinite(viewBox.width) || !Number.isFinite(viewBox.height) ||
      !(viewBox.width > 0) || !(viewBox.height > 0)) {
    throw new Error('viewBox dimensions must be positive and finite');
  }
  return [viewBox.width / 16, viewBox.height / 16];
}

/** Renderer settings expressed in the SVG's curve coordinate space. */
export function roundtripRenderSettings(raster, background, blurPixels = 1) {
  if (!Number.isFinite(raster?.scale) || raster.scale <= 0) {
    throw new Error('raster scale must be positive');
  }
  if (!raster.origin || raster.origin.length !== 2 || !raster.origin.every(Number.isFinite)) {
    throw new Error('raster origin must contain two finite coordinates');
  }
  if (!background || background.length !== 3 || !background.every(Number.isFinite)) {
    throw new Error('background must contain three finite channels');
  }
  const blur = positiveNumber(blurPixels, 'blur');
  return {
    s: [blur * raster.scale, blur * raster.scale],
    scale: raster.scale,
    origin: Array.from(raster.origin),
    bg: Array.from(background),
  };
}

export function summarizeLosses(losses) {
  if (!losses.length) return { start: null, end: null, min: null, bestStep: null };
  const min = Math.min(...losses);
  return {
    start: losses[0],
    end: losses.at(-1),
    min,
    bestStep: losses.indexOf(min) + 1,
  };
}

async function timedRender(renderer, scene, settings) {
  const started = performance.now();
  renderer.uploadScene(scene, settings);
  const image = await renderer.forward();
  return { image, ms: performance.now() - started };
}

export async function runRoundtrip(argv) {
  const options = parseRoundtripArgs(argv);
  const source = await readText(options.svg);
  const parsed = parseFillSvg(source);
  if (!parsed.shapes.length) throw new Error('SVG must contain at least one filled path');

  const raster = rasterSize(parsed.viewBox, options.optSize);
  const background = parsed.background ?? [1, 1, 1];
  const offset = resolveRoundtripOffset(options.offset, parsed.viewBox);
  const crisp = roundtripRenderSettings(raster, background);
  const targetScene = packScene(parsed.shapes);
  const { model, lrs } = buildRoundtripModel(parsed.shapes, {
    translation: offset,
    width: parsed.viewBox.width,
    height: parsed.viewBox.height,
  });

  const device = await requestDevice();
  const host = getWebGPUHostInfo();
  const root = runPath(host.environment, 'roundtrip', options.out);
  await mkdir(root);
  if (options.saveEvery) await mkdir(`${root}/frames`);

  let renderer;
  try {
    renderer = await Renderer.create(device, {
      width: raster.width,
      height: raster.height,
      maxShapes: model.maxShapes,
      maxPieces: model.maxPieces,
      maxCurves: model.maxCurves,
    });

    const targetRender = await timedRender(renderer, targetScene, crisp);
    const initialDecoded = model.decode();
    const initialSvg = sceneToSVG(
      initialDecoded.shapes,
      parsed.viewBox.width,
      parsed.viewBox.height,
      {
        background: parsed.background ? rgbToHex(parsed.background) : null,
        viewBox: parsed.viewBox,
      },
    );
    const initialRender = await timedRender(renderer, initialDecoded.scene, crisp);
    const initialQuality = l2Quality(initialRender.image, targetRender.image);

    const logEvery = Math.max(1, Math.floor(options.steps / 10));
    const settings = (step) => roundtripRenderSettings(
      raster,
      background,
      annealedBlur(step, options.steps, options.blurStart, options.blur),
    );
    const onStep = options.quiet && !options.saveEvery ? null : async ({ step, loss }) => {
      if (!options.quiet && (step === 1 || step % logEvery === 0)) {
        console.log(`roundtrip step ${step}: ${formatLoss(loss)}`);
      }
      if (options.saveEvery && step % options.saveEvery === 0) {
        const decoded = model.decode();
        const frame = await timedRender(renderer, decoded.scene, crisp);
        const name = `step-${String(step).padStart(8, '0')}.png`;
        await writeBytes(`${root}/frames/${name}`, imageToPng(frame.image, raster.width, raster.height));
      }
    };

    const run = await optimize({
      renderer,
      model,
      lrs,
      settings,
      steps: options.steps,
      target: targetRender.image,
      onStep,
    });

    const finalDecoded = model.decode();
    const finalSvg = sceneToSVG(
      finalDecoded.shapes,
      parsed.viewBox.width,
      parsed.viewBox.height,
      {
        background: parsed.background ? rgbToHex(parsed.background) : null,
        viewBox: parsed.viewBox,
      },
    );
    const finalRender = await timedRender(renderer, finalDecoded.scene, crisp);
    const finalQuality = l2Quality(finalRender.image, targetRender.image);
    const fillRules = parsed.shapes.map((shape) => shape.fillRule ?? 'nonzero');
    const config = {
      demo: 'roundtrip',
      loss: 'l2',
      svg: options.svg,
      viewBox: parsed.viewBox,
      requestedOptSize: options.optSize,
      optSize: [raster.width, raster.height],
      scale: raster.scale,
      origin: raster.origin,
      background,
      fillRules,
      shapes: parsed.shapes.length,
      curves: targetScene.curveCount,
      steps: options.steps,
      blur: options.blur,
      blurStart: options.blurStart,
      offset,
      saveEvery: options.saveEvery,
      fixedStyle: true,
    };
    const result = {
      schemaVersion: 1,
      status: 'ok',
      demo: 'roundtrip',
      engine: { name: 'windfoil', environment: host.environment, backend: host.backend },
      workload: {
        loss: 'l2',
        primitive: 'source-quadratic-loop',
        source: options.svg,
        fillRules,
        shapes: parsed.shapes.length,
        curves: targetScene.curveCount,
        optSize: [raster.width, raster.height],
        offset,
      },
      progress: { stepsCompleted: run.steps, stopReason: 'steps' },
      timing: {
        targetRenderMs: targetRender.ms,
        initialRenderMs: initialRender.ms,
        optimizeMs: run.optimizeMs,
        callbackMs: run.callbackMs ?? 0,
        finalRenderMs: finalRender.ms,
        msPerStep: run.msPerStep,
      },
      loss: summarizeLosses(run.losses),
      quality: { initial: initialQuality, final: finalQuality },
      system: {
        adapter: host.adapterInfo ?? null,
        limits: host.limits ?? null,
        culling: renderer.getCullingInfo(),
      },
      artifacts: {
        sourceSvg: 'source.svg',
        targetPng: 'target.png',
        initialPng: 'initial.png',
        initialSvg: 'initial.svg',
        finalPng: 'final.png',
        finalSvg: 'final.svg',
        trace: 'trace.jsonl',
        frames: options.saveEvery ? 'frames/' : null,
      },
    };

    await Promise.all([
      writeText(`${root}/source.svg`, source),
      writeBytes(`${root}/target.png`, imageToPng(targetRender.image, raster.width, raster.height)),
      writeBytes(`${root}/initial.png`, imageToPng(initialRender.image, raster.width, raster.height)),
      writeText(`${root}/initial.svg`, initialSvg),
      writeBytes(`${root}/final.png`, imageToPng(finalRender.image, raster.width, raster.height)),
      writeText(`${root}/final.svg`, finalSvg),
      writeText(`${root}/trace.jsonl`, run.timeline.map((item) => JSON.stringify(item)).join('\n') + '\n'),
      writeText(`${root}/config.json`, JSON.stringify(config, null, 2) + '\n'),
      writeText(`${root}/result.json`, JSON.stringify(result, null, 2) + '\n'),
    ]);
    if (!options.quiet) {
      console.log(`saved ${root} (${run.steps} steps, final PSNR ${finalQuality.psnrDb.toFixed(2)} dB)`);
    }
    return { root, result, config };
  } finally {
    renderer?.destroy();
    device.destroy();
  }
}

async function isMainModule() {
  if (typeof Deno !== 'undefined' && Deno.version?.deno) return import.meta.main;
  if (typeof process !== 'undefined' && process.versions?.node && process.argv[1]) {
    const { pathToFileURL } = await import('node:url');
    return pathToFileURL(process.argv[1]).href === import.meta.url;
  }
  return false;
}

if (await isMainModule()) await runMain(runRoundtrip);
