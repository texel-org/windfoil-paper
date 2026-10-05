#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { access, mkdir, readFile, unlink, writeFile } from 'node:fs/promises';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  imageToPng,
  l2Quality,
  loadImageSource,
  loadTarget,
  targetFromSource,
} from '../demos/util/image.js';
import { arg, argList, parseArgs, slug, stamp } from '../demos/util/runtime.js';
import { BLUR_ANNEAL_FRACTION, createInit, serializeInit } from '../demos/util/model.js';
import { resolveKernel } from '../js/filter-kernels.js';
import { parseOptSizes, resolveOptSize } from '../demos/util/opt-size.js';
import {
  WINDFOIL_ENVIRONMENTS,
  windfoilBackend,
  windfoilCommand,
} from '../tools/windfoil-runtime.js';

import { exportCsv } from './csv.js';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const VENV_PYTHON = join(REPO, '.venv', 'bin', 'python');
const PYTHON = process.env.PYTHON ?? (await exists(VENV_PYTHON) ? VENV_PYTHON : 'python3');
// Each comparison engine has its own venv (bench/pod-setup.sh builds both), and
// the base venv cannot import either engine. Find them where setup puts them,
// so the documented path needs no exports; the variables remain as overrides.
const enginePython = async (variable, venv) => {
  const python = join(REPO, venv, 'bin', 'python');
  return process.env[variable] ?? (await exists(python) ? python : PYTHON);
};
const DIFFVG_PYTHON = await enginePython('DIFFVG_PYTHON', '.venv-diffvg');
const BEZIER_PYTHON = await enginePython('BEZIER_PYTHON', '.venv-bezier');
const DEFAULT_N = [32, 64, 128, 512, 1024, 2048, 4096, 10000, 50000];
const DEFAULT_ENVS = ['node', 'diffvg', 'bezier'];
const SUPPORTED_ENVS = [...WINDFOIL_ENVIRONMENTS, 'diffvg', 'bezier'];
const WINDFOIL_VARIANTS = ['anneal', 'crisp'];
// Benchmarks use a constant one-pixel filter; annealing is an explicit opt-in.
const DEFAULT_WINDFOIL_VARIANTS = ['crisp'];
const DEFAULT_CAPS = { diffvg: 4096, bezier: 4096 };
const CLIP_MODEL = 'ViT-B-32-quickgelu';

const options = parseArgs();
const loss = String(arg(options, 'loss', 'l2')).toLowerCase();
if (!['l2', 'clip'].includes(loss)) fail('--loss must be l2 or clip');

const environments = nonEmpty(argList(options, 'env', DEFAULT_ENVS), 'env').map(String);
for (const environment of environments) {
  if (!SUPPORTED_ENVS.includes(environment)) fail(`unsupported environment: ${environment}`);
}
const windfoilVariants = nonEmpty(
  argList(options, 'windfoil-variant', DEFAULT_WINDFOIL_VARIANTS),
  'windfoil-variant',
).map(String);
for (const variant of windfoilVariants) {
  if (!WINDFOIL_VARIANTS.includes(variant)) fail(`unsupported Windfoil variant: ${variant}`);
}
const counts = nonEmpty(argList(options, 'n', DEFAULT_N), 'n').map((value) => integer(value, 'n'));
const repeats = integer(arg(options, 'repeats', 1), 'repeats');
const sizeSpec = parseOptSizes([arg(options, 'opt-size', 128)])[0];
const k = integer(arg(options, 'k', 8), 'k');
const kernel = String(arg(options, 'kernel', 'box'));
resolveKernel(kernel); // throws with the accepted kernel names
const scheduleSteps = integer(arg(options, 'schedule-steps', 800), 'schedule-steps');
const baseSeed = integer(arg(options, 'seed', 1), 'seed', { allowZero: true });
const force = boolean(arg(options, 'force', false));
const prompt = String(arg(options, 'prompt', 'a hot air balloon festival'));
const augs = integer(arg(options, 'augs', 4), 'augs', { allowZero: true });
const lossUrl = String(arg(options, 'loss-url', 'ws://127.0.0.1:8765'));
const clipWeights = String(arg(options, 'clip-weights', 'openai'));
const diffvgClipMax = integer(arg(options, 'diffvg-clip-max', 512), 'diffvg-clip-max');
// --windfoil-style=raw swaps the demos' default 9-anchor softmax colour codec
// for the raw RGB one, which is what DiffVG optimises: free RGB plus alpha, four
// parameters per shape, held in [0,1]. The anchor codec spends nine parameters
// per shape on a softmax over the premultiplied style body and couples colour
// to alpha through the transparent anchor, so at equal N it is not an equal
// colour parametrisation. Raw is the benchmark default because it removes that
// objection, not because it wins: over a corpus the two are level on quality,
// though a single cell can move either way. `--windfoil-style=anchor` selects
// the demos' codec.
const windfoilStyle = String(arg(options, 'windfoil-style', 'raw'));
if (!['anchor', 'raw'].includes(windfoilStyle)) {
  fail("--windfoil-style must be 'anchor' or 'raw'");
}
// Equal nominal learning rates are not equal steps here. The raw codec trains a
// colour's logit, and a sigmoid's slope is at most 1/4, so an Adam step of lr
// moves the colour by at most lr/4; DiffVG steps the linear colour itself, by
// 0.02 (bench/diffvg/run.py). 0.1 puts Windfoil's largest colour step beside
// DiffVG's instead of well under it. The demos keep the codec's own default.
const RAW_COLOR_LR = 0.1;
// Sweep-runtime policy, not a demonstrated ceiling of either engine; raise to
// compare at larger N (see bench/README.md).
const CAPS = {
  diffvg: integer(arg(options, 'diffvg-max', DEFAULT_CAPS.diffvg), 'diffvg-max'),
  bezier: integer(arg(options, 'bezier-max', DEFAULT_CAPS.bezier), 'bezier-max'),
};
const targetInput = arg(options, 'target', null);
const targetLabel = optionalString(arg(options, 'target-label', null));
const sourceId = optionalString(arg(options, 'source-id', null));
const sourceRevision = optionalString(arg(options, 'source-revision', null));
const dataset = optionalString(arg(options, 'dataset', null));
const protocol = optionalString(arg(options, 'protocol', null));
if (loss === 'l2' && !targetInput) fail('--target is required for L2');

const budgets = [];
if (options.steps !== undefined) {
  for (const value of nonEmpty(argList(options, 'steps'), 'steps')) {
    budgets.push({ mode: 'steps', value: integer(value, 'steps') });
  }
}
if (options.seconds !== undefined) {
  for (const value of nonEmpty(argList(options, 'seconds'), 'seconds')) {
    budgets.push({ mode: 'seconds', value: positive(value, 'seconds') });
  }
}
if (!budgets.length) budgets.push({ mode: 'steps', value: 800 });

const generatedSuiteId = `${stamp()}-bench-${slug(loss)}`;
const suiteRoot = resolve(String(arg(options, 'out', join('output', generatedSuiteId))));
const inputRoot = join(suiteRoot, 'inputs');
await mkdir(inputRoot, { recursive: true });

let target = null;
let targetSha256 = null;
let background = [1, 1, 1];
let dimensions;
if (loss === 'l2') {
  const source = await loadImageSource(resolve(String(targetInput)));
  dimensions = resolveOptSize(sizeSpec, loss, source);
  const loaded = targetFromSource(source, dimensions.width, dimensions.height);
  target = join(inputRoot, 'target.png');
  background = meanRgb8(loaded.u8);
  const targetPng = imageToPng(loaded.rgba, dimensions.width, dimensions.height);
  targetSha256 = sha256(targetPng);
  await writeFile(target, targetPng);
} else dimensions = resolveOptSize(sizeSpec, loss);
const { width, height } = dimensions;

const configBase = {
  loss,
  environments,
  windfoil_variants: windfoilVariants,
  n: counts,
  k,
  // Box stays out of the key so existing suites still match.
  ...(kernel === 'box' ? {} : { kernel }),
  schedule_steps: scheduleSteps,
  opt_size: [width, height],
  opt_size_requested: sizeSpec,
  budgets,
  repeats,
  seed: baseSeed,
  target: target ? pathInSuite(target) : null,
  target_label: targetLabel,
  target_sha256: targetSha256,
  source_id: sourceId,
  source_revision: sourceRevision,
  dataset,
  protocol,
  prompt: loss === 'clip' ? prompt : null,
  augs: loss === 'clip' ? augs : null,
  loss_url: loss === 'clip' ? lossUrl : null,
  clip_model: loss === 'clip' ? CLIP_MODEL : null,
  clip_weights: loss === 'clip' ? clipWeights : null,
  diffvg_clip_max: diffvgClipMax,
};
const configKey = sha256(JSON.stringify(configBase));
const previousConfig = force ? null : await readJson(join(suiteRoot, 'config.json'));
const suiteId = previousConfig?.config_key === configKey && typeof previousConfig.suite_id === 'string'
  ? previousConfig.suite_id
  : generatedSuiteId;
const config = {
  schema_version: 1,
  suite_id: suiteId,
  config_key: configKey,
  ...configBase,
};
await writeJson(join(suiteRoot, 'config.json'), config);

const records = [];
const initCache = new Map();
for (const environment of environments) {
  const variants = WINDFOIL_ENVIRONMENTS.includes(environment) ? windfoilVariants : [null];
  for (const variant of variants) {
    for (const n of counts) {
      for (const budget of budgets) {
        for (let repeat = 0; repeat < repeats; repeat++) {
          const seed = baseSeed + repeat;
          const budgetLabel = `${budget.mode}-${String(budget.value).replace('.', '_')}`;
          const variantLabel = variant ? `-${variant}` : '';
          const runId = `${suiteId}-${environment}${variantLabel}-n${n}-${budgetLabel}-r${repeat}`;
          const cell = variant
            ? join(suiteRoot, 'cells', environment, variant, `n${n}`, budgetLabel, `r${repeat}`)
            : join(suiteRoot, 'cells', environment, `n${n}`, budgetLabel, `r${repeat}`);
          const resultPath = join(cell, 'result.json');
          await mkdir(cell, { recursive: true });

          const meta = metadata({
            runId, environment, variant, loss, n, k, width, height, seed, repeat, budget, cell,
          });

          const cached = force ? null : await readJson(resultPath);
          if (await isReusable(cached, meta, cell)) {
            records.push(cached);
            await flushResults();
            continue;
          }

          const skip = skipReason(environment, loss, n);
          if (skip) {
            const result = { ...meta, status: 'skipped', skip };
            await writeJson(resultPath, result);
            records.push(result);
            await flushResults();
            console.log(`skip ${environment}${variantLabel} n=${n} ${budgetLabel}: ${skip.reason}`);
            continue;
          }

          const init = await sharedInit(n, seed);
          const invocation = commandFor({ environment, variant, loss, n, seed, budget, init, cell });
          console.log(`run  ${environment}${variantLabel} n=${n} ${budgetLabel} r=${repeat}`);
          await clearCellArtifacts(cell);
          const processResult = await runProcess(invocation);
          await writeFile(join(cell, 'stdout.log'), processResult.stdout);
          await writeFile(join(cell, 'stderr.log'), processResult.stderr);

          let raw = null;
          try { raw = JSON.parse(await readFile(resultPath, 'utf8')); } catch { /* reported below */ }
          let result = normalize(raw, meta, processResult);

          if (loss === 'l2' && result.status === 'ok') {
            result = await score(result, cell);
          }
          await writeJson(resultPath, result);
          records.push(result);
          await flushResults();
        }
      }
    }
  }
}

await exportCsv(suiteRoot);
console.log(`wrote ${relative(REPO, join(suiteRoot, 'results.jsonl'))} (${records.length} cells)`);
const errorCount = records.filter(({ status }) => status === 'error').length;
if (errorCount) {
  console.error(`${errorCount} benchmark cell${errorCount === 1 ? '' : 's'} failed`);
  process.exitCode = 1;
}

async function sharedInit(n, seed) {
  const key = `${n}:${seed}`;
  if (initCache.has(key)) return initCache.get(key);
  const path = join(inputRoot, `init-n${n}-s${seed}.json`);
  const initial = createInit({ n, width, height, k, seed, background });
  await writeJson(path, serializeInit(initial));
  initCache.set(key, path);
  return path;
}

function commandFor({ environment, variant, loss, n, seed, budget, init, cell }) {
  const common = [
    '--init', init,
    '--n', String(n),
    '--k', String(k),
    `--${budget.mode}`, String(budget.value),
    '--seed', String(seed),
    '--out', cell,
    '--benchmark',
  ];
  if (loss === 'l2') common.unshift('--target', target);
  else common.unshift(
    '--prompt', prompt,
    '--augs', String(augs),
    '--loss-url', lossUrl,
    '--clip-weights', clipWeights,
  );

  if (environment === 'diffvg') {
    return {
      command: DIFFVG_PYTHON,
      args: [
        join(REPO, 'bench/diffvg/run.py'), '--loss', loss,
        '--opt-width', String(width), '--opt-height', String(height),
        ...common,
      ],
      env: process.env,
    };
  }
  if (environment === 'bezier') {
    const args = [join(REPO, 'bench/bezier/run.py'), '--loss', loss, '--target', target,
      '--n', String(n), `--${budget.mode}`, String(budget.value), '--seed', String(seed), '--out', cell];
    return { command: BEZIER_PYTHON, args, env: process.env };
  }

  const windfoilCli = resolve(String(arg(options, 'windfoil-cli', 'demos/cli.js')));
  const windfoilSize = loss === 'l2' ? 'max' : String(width);
  // Both filter widths are explicit so the protocol does not depend on CLI
  // defaults. The final width stays at one pixel at every resolution.
  const windfoilArgs = [
    windfoilCli, loss, ...common,
    '--opt-size', windfoilSize,
    ...(kernel === 'box' ? [] : ['--kernel', kernel]),
    '--blur', '1',
    '--blur-start', variant === 'crisp' ? '1' : '7',
    // sigmoid rather than identity keeps every decode displayable, which is the
    // guarantee the anchor codec gives by construction and the reason the OKLab
    // codec it replaced was removed. It matches DiffVG's clamp to [0,1] without
    // that clamp's zero-gradient dead zone.
    ...(windfoilStyle === 'raw'
      ? ['--style', 'raw', '--channels', 'rgb', '--alpha', 'learned', '--transfer', 'sigmoid',
        '--color-lr', String(RAW_COLOR_LR)]
      : []),
    '--schedule-steps', String(scheduleSteps),
  ];
  return windfoilCommand(environment, windfoilArgs);
}

async function score(result, cell) {
  const finalPng = join(cell, 'final.png');
  if (!await exists(finalPng)) {
    return errorResult(result, 'artifacts', 'successful adapter did not write final.png');
  }
  const qualityPath = join(cell, 'quality.json');
  const started = performance.now();
  const [targetImage, renderImage] = await Promise.all([
    loadTarget(target, width, height),
    loadTarget(finalPng, width, height),
  ]);
  if (targetImage.sourceWidth !== renderImage.sourceWidth || targetImage.sourceHeight !== renderImage.sourceHeight) {
    return errorResult(
      result,
      'metrics',
      `dimension mismatch: target=${targetImage.sourceWidth}x${targetImage.sourceHeight} ` +
        `render=${renderImage.sourceWidth}x${renderImage.sourceHeight}`,
    );
  }
  const { mseRgb, psnrDb } = l2Quality(renderImage.rgba, targetImage.rgba);
  const quality = {
    width: targetImage.sourceWidth,
    height: targetImage.sourceHeight,
    mse_rgb: mseRgb,
    psnr_db: psnrDb,
  };
  await writeJson(qualityPath, quality);
  return {
    ...result,
    quality,
    timing: { ...(result.timing ?? {}), metrics_ms: performance.now() - started },
    artifacts: { ...(result.artifacts ?? {}), quality_json: pathInSuite(qualityPath) },
  };
}

function metadata({ runId, environment, variant, loss, n, k, width, height, seed, repeat, budget, cell }) {
  const engine = environment === 'diffvg' ? 'diffvg' : environment === 'bezier' ? 'bezier-splatting' : 'windfoil';
  const backend = WINDFOIL_ENVIRONMENTS.includes(environment)
    ? windfoilBackend(environment)
    : environment === 'diffvg' ? 'cuda-diffvg' : 'cuda-gsplat';
  return {
    schema_version: 1,
    suite_id: suiteId,
    config_key: configKey,
    run_id: runId,
    status: 'ok',
    engine: { name: engine, environment, backend },
    workload: {
      loss,
      primitive: environment === 'bezier' ? 'closed-degree4-curve' : 'closed-quadratic-loop',
      n,
      k: environment === 'bezier' ? null : k,
      opt_size: [width, height],
      seed,
      repeat,
      target: target ? pathInSuite(target) : null,
      target_label: targetLabel,
      source_id: sourceId,
      dataset,
      protocol,
      prompt: loss === 'clip' ? prompt : null,
      init: environment === 'bezier' ? { kind: 'native' } : { kind: 'shared-neutral' },
      variant,
      engine_options: variant ? {
        style: windfoilStyle,
        ...(kernel === 'box' ? {} : { kernel }),
        ...(windfoilStyle === 'raw' ? { color_lr: RAW_COLOR_LR } : {}),
        blur_anneal: variant === 'anneal',
        blur_start: variant === 'anneal' ? 7 : 1,
        blur_end: 1,
        anneal_fraction: variant === 'anneal' ? BLUR_ANNEAL_FRACTION : 0,
        schedule_steps: scheduleSteps,
        ...(loss === 'clip' ? {
          augs,
          model: CLIP_MODEL,
          pretrained: clipWeights,
        } : {}),
      } : undefined,
    },
    provenance: sourceRevision ? { source_revision: sourceRevision } : undefined,
    budget: { mode: budget.mode, requested: budget.value, clock: 'optimize' },
    artifacts: {
      result_json: pathInSuite(join(cell, 'result.json')),
      final_png: pathInSuite(join(cell, 'final.png')),
      // Windfoil and DiffVG export their vector scene exactly; Bézier
      // Splatting's closed mode has no exact SVG form, so it writes none.
      final_svg: environment === 'bezier' ? null : pathInSuite(join(cell, 'final.svg')),
      trace: pathInSuite(join(cell, 'trace.jsonl')),
    },
  };
}

function normalize(raw, meta, processResult) {
  const failed = processResult.code !== 0 || !raw;
  const status = failed ? 'error' : raw.status ?? 'ok';
  const {
    schemaVersion: _schemaVersion,
    runId: _runId,
    status: _status,
    engine: rawEngine,
    workload: rawWorkload,
    budget: _budget,
    progress: rawProgress,
    timing: rawTiming,
    loss: rawLoss,
    quality: rawQuality,
    artifacts: _artifacts,
    ...extra
  } = raw ?? {};
  const progress = normalizeFields(rawProgress, {
    steps_completed: 'stepsCompleted',
    stop_reason: 'stopReason',
    deadline_overshoot_ms: 'deadlineOvershootMs',
  });
  const timing = {
    ...normalizeFields(rawTiming, {
      process_internal_ms: 'processInternalMs',
      optimize_start_epoch_ms: 'optimizeStartEpochMs',
      setup_ms: 'setupMs',
      warmup_ms: 'warmupMs',
      optimize_ms: 'optimizeMs',
      callback_ms: 'callbackMs',
      final_render_ms: 'finalRenderMs',
      final_readback_ms: 'finalReadbackMs',
      output_ms: 'outputMs',
      metrics_ms: 'metricsMs',
      steps_timed: 'stepsTimed',
      step_ms: 'stepMs',
    }),
    process_ms: processResult.elapsedMs,
    spawn_epoch_ms: processResult.startedEpochMs,
  };
  if (timing.steps_timed === undefined && progress?.steps_completed !== undefined) {
    timing.steps_timed = progress.steps_completed;
  }
  const result = {
    ...extra,
    ...meta,
    status,
    engine: { ...(typeof rawEngine === 'object' ? rawEngine : {}), ...meta.engine },
    workload: {
      ...meta.workload,
      engine_options: mergeObjects(meta.workload.engine_options, rawWorkload?.engine_options),
    },
    budget: meta.budget,
    progress,
    timing,
    loss: normalizeFields(rawLoss, { start: 'start', end: 'end', min: 'min', best_step: 'bestStep' }),
    quality: normalizeFields(rawQuality, { mse_rgb: 'mseRgb', psnr_db: 'psnrDb' }),
    artifacts: meta.artifacts,
  };
  if (failed) {
    const adapterError = raw?.error;
    result.error = {
      stage: adapterError?.stage ?? 'adapter',
      message: adapterError?.message ?? processResult.error?.message ??
        (!raw ? 'adapter did not write result.json' : `exit ${processResult.code}`),
      exit_code: processResult.code,
      signal: processResult.signal,
      detail: adapterError?.detail ?? tail(processResult.stderr || processResult.stdout),
    };
  }
  return result;
}

function normalizeFields(source, fields) {
  if (!source) return undefined;
  const out = {};
  for (const [canonical, alias] of Object.entries(fields)) {
    const value = source[canonical] !== undefined ? source[canonical] : source[alias];
    if (value !== undefined) out[canonical] = value;
  }
  return out;
}

function mergeObjects(base, override) {
  if (!base && !override) return undefined;
  return { ...(base ?? {}), ...(override ?? {}) };
}

function skipReason(environment, loss, n) {
  if (n > (CAPS[environment] ?? Infinity)) {
    return { kind: 'cap', reason: `${environment} is capped at N=${CAPS[environment]}` };
  }
  if (loss === 'clip' && environment === 'diffvg' && n > diffvgClipMax) {
    return { kind: 'cap', reason: `DiffVG CLIP is capped at N=${diffvgClipMax}` };
  }
  if (loss === 'clip' && environment === 'bezier') {
    return { kind: 'unsupported-loss', reason: `${environment} adapter supports L2 only` };
  }
  return null;
}

async function runProcess({ command, args, env }) {
  const started = performance.now();
  // Wall-clock epoch of the spawn itself. Paired with the adapter's
  // optimize_start_epoch_ms this gives the startup cost -- interpreter,
  // imports, device init, setup, warmup -- that every engine pays before its
  // first step and that an optimise-relative clock hides entirely.
  const startedEpochMs = Date.now();
  return await new Promise((resolveRun) => {
    let stdout = '', stderr = '', settled = false;
    const child = spawn(command, args, { cwd: REPO, env, stdio: ['ignore', 'pipe', 'pipe'] });
    child.stdout.on('data', (data) => { stdout += data; });
    child.stderr.on('data', (data) => { stderr += data; });
    child.on('error', (error) => {
      if (settled) return;
      settled = true;
      resolveRun({ code: null, signal: null, stdout, stderr, error, elapsedMs: performance.now() - started, startedEpochMs });
    });
    child.on('close', (code, signal) => {
      if (settled) return;
      settled = true;
      resolveRun({ code, signal, stdout, stderr, error: null, elapsedMs: performance.now() - started, startedEpochMs });
    });
  });
}

async function flushResults() {
  const jsonl = records.map((record) => JSON.stringify(record)).join('\n') + (records.length ? '\n' : '');
  await writeFile(join(suiteRoot, 'results.jsonl'), jsonl);
}

async function isReusable(result, meta, cell) {
  const canonical = result?.schema_version === 1 &&
    result.suite_id === meta.suite_id &&
    result.config_key === meta.config_key &&
    result.run_id === meta.run_id &&
    (result.status === 'ok' || result.status === 'skipped');
  if (!canonical || result.status === 'skipped') return canonical;
  const required = ['final.png', 'trace.jsonl'];
  if (loss === 'l2') required.push('quality.json');
  // The two engines whose fit is a vector scene export it exactly; Bézier
  // Splatting's closed mode has no exact SVG form, so it writes none.
  if (['windfoil', 'diffvg'].includes(result.engine?.name)) required.push('final.svg');
  return (await Promise.all(required.map((name) => exists(join(cell, name))))).every(Boolean);
}

async function clearCellArtifacts(cell) {
  await Promise.all(
    ['result.json', 'final.png', 'final.svg', 'trace.jsonl', 'quality.json']
      .map((name) => removeIfExists(join(cell, name))),
  );
}

async function removeIfExists(path) {
  try {
    await unlink(path);
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
}

function errorResult(result, stage, message) {
  return { ...result, status: 'error', error: { stage, message } };
}

function pathInSuite(path) {
  return relative(suiteRoot, path).split('\\').join('/');
}

async function writeJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify(value, null, 2) + '\n');
}

async function exists(path) {
  try { await access(path); return true; } catch { return false; }
}

async function readJson(path) {
  try {
    return JSON.parse(await readFile(path, 'utf8'));
  } catch (error) {
    if (error?.code === 'ENOENT' || error instanceof SyntaxError) return null;
    throw error;
  }
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function integer(value, name, { allowZero = false } = {}) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < (allowZero ? 0 : 1)) {
    fail(`--${name} must be a ${allowZero ? 'non-negative' : 'positive'} integer`);
  }
  return number;
}

function positive(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) fail(`--${name} must be positive`);
  return number;
}

function nonEmpty(values, name) {
  if (!values.length) fail(`--${name} must not be empty`);
  return values;
}

function boolean(value) {
  return value === true || value === 'true' || value === '1';
}

function optionalString(value) {
  if (value === null || value === undefined) return null;
  if (value === true || !String(value).trim()) fail('metadata options require a value');
  return String(value);
}

function tail(value, length = 4000) {
  return String(value ?? '').slice(-length).trim();
}

function meanRgb8(rgba) {
  const mean = [0, 0, 0];
  const pixels = rgba.length / 4;
  for (let i = 0; i < pixels; i++) {
    for (let channel = 0; channel < 3; channel++) mean[channel] += rgba[4 * i + channel] / 255;
  }
  return mean.map((value) => value / pixels);
}

function fail(message) {
  throw new Error(message);
}
