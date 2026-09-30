// Compare committed Windfoil revisions using the original Färlev inputs and CLI.
// Each run gets a fresh Node process; both snapshots share installed dependencies.
import { spawn, execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { closeSync, openSync } from 'node:fs';
import { mkdir, readFile, symlink, writeFile } from 'node:fs/promises';
import { cpus, release } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = Object.fromEntries(process.argv.slice(2).map((arg) => {
  if (!arg.startsWith('--')) throw new Error('use --name=value options');
  const [key, ...value] = arg.slice(2).split('=');
  return [key, value.join('=') || '1'];
}));
const allowed = new Set(['inputs', 'out', 'baseline', 'candidate', 'steps', 'repeats', 'seconds', 'seconds-repeats', 'help']);
for (const key of Object.keys(args)) if (!allowed.has(key)) throw new Error(`unknown option --${key}`);
if (args.help || !args.inputs) {
  console.log('node tools/perf-farlev.js --inputs=<suite/inputs> [--baseline=88d6d7c] [--candidate=HEAD] [--out=output/farlev-perf] [--steps=800] [--repeats=4] [--seconds=300] [--seconds-repeats=1]\nInputs must contain target.png and init-n512-s1.json. Use --seconds=0 for only the repeated fixed-step comparison.');
  process.exit(args.help ? 0 : 2);
}
const integer = (key, fallback, minimum = 1) => {
  const value = Number(args[key] ?? fallback);
  if (!Number.isInteger(value) || value < minimum) throw new Error(`--${key} must be an integer >= ${minimum}`);
  return value;
};
const steps = integer('steps', 800), repeats = integer('repeats', 4);
const seconds = integer('seconds', 300, 0), secondsRepeats = integer('seconds-repeats', 1);
const out = resolve(args.out ?? `output/farlev-perf-${new Date().toISOString().replace(/[:.]/g, '-')}`);
const inputs = resolve(args.inputs);
const revisions = Object.fromEntries(['baseline', 'candidate'].map((label) => [label,
  execFileSync('git', ['rev-parse', '--verify', `${args[label] ?? (label === 'baseline' ? '88d6d7c' : 'HEAD')}^{commit}`],
    { cwd: repo, encoding: 'utf8' }).trim()]));
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const json = async (path, value, options) => writeFile(path, JSON.stringify(value, null, 2) + '\n', options);
const target = await readFile(join(inputs, 'target.png'));
const initBytes = await readFile(join(inputs, 'init-n512-s1.json'));
const init = JSON.parse(initBytes);
if (init.width !== 512 || init.height !== 288 || init.n !== 512 || init.k !== 8 ||
    !target.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) ||
    target.readUInt32BE(16) !== 512 || target.readUInt32BE(20) !== 288) {
  throw new Error('expected the 512x288, N=512, K=8 Färlev target and shared initialization');
}
await mkdir(out, { recursive: true });
const config = { revisions, node: process.version, nodeExecutable: process.execPath,
  machine: { cpu: cpus()[0].model, platform: process.platform, arch: process.arch, os: release() },
  dependencies: Object.fromEntries(await Promise.all(['webgpu', 'webgpu-legacy', 'fast-png'].map(async (name) =>
    [name, JSON.parse(await readFile(join(repo, 'node_modules', name, 'package.json'))).version]))),
  inputs: { source: inputs, targetSha256: hash(target), initSha256: hash(initBytes) },
  steps, repeats, seconds, secondsRepeats, scheduleSteps: 800,
  method: 'Fresh Node processes; sequential runs; alternate revision order per repeat. Includes CLI warmup. Time budget counts optimization only; startup is recorded separately. Committed sources only.' };
// Refuse to mix a previous comparison with a new one.
await json(join(out, 'comparison-config.json'), config, { flag: 'wx' });
await mkdir(join(out, 'inputs'));
await writeFile(join(out, 'inputs', 'target.png'), target);
await writeFile(join(out, 'inputs', 'init-n512-s1.json'), initBytes);
for (const label of Object.keys(revisions)) {
  const source = join(out, 'sources', label);
  await mkdir(source, { recursive: true });
  const archive = join(out, `${label}.tar`);
  await writeFile(archive, execFileSync('git', ['archive', revisions[label], 'js', 'demos', 'package.json'],
    { cwd: repo, maxBuffer: 32 * 1024 * 1024 }));
  execFileSync('tar', ['-xf', archive, '-C', source]);
  await symlink(join(repo, 'node_modules'), join(source, 'node_modules'), 'dir');
}
const records = [];
const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b), mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};
async function run(label, mode, value, repeat) {
  const cell = join(out, 'runs', mode, label, `r${repeat}`);
  await mkdir(cell, { recursive: true });
  const source = join(out, 'sources', label);
  const command = ['demos/cli.js', 'l2', '--target', join(out, 'inputs', 'target.png'),
    '--init', join(out, 'inputs', 'init-n512-s1.json'), '--n', '512', '--k', '8',
    `--${mode}`, String(value), '--seed', '1', '--out', cell, '--benchmark', '--opt-size', 'max',
    '--blur', '1', '--blur-floor', '1', '--style', 'raw', '--channels', 'rgb', '--alpha', 'learned',
    '--transfer', 'sigmoid', '--color-lr', '0.1', '--schedule-steps', '800'];
  console.log(`${label} ${revisions[label].slice(0, 7)}: ${mode}=${value}, repeat ${repeat + 1}`);
  const stdout = openSync(join(cell, 'stdout.log'), 'w'), stderr = openSync(join(cell, 'stderr.log'), 'w');
  const started = performance.now(), spawnEpochMs = Date.now();
  let code;
  try {
    code = await new Promise((resolveRun, reject) => {
      const child = spawn(process.execPath, command, { cwd: source,
        env: { ...process.env, WF_WEBGPU_BACKEND: 'dawn' }, stdio: ['ignore', stdout, stderr] });
      child.once('error', reject);
      child.once('close', (status) => resolveRun(status));
    });
  } finally { closeSync(stdout); closeSync(stderr); }
  const processMs = performance.now() - started;
  if (code !== 0) throw new Error(`${label} failed (${code}): ${await readFile(join(cell, 'stderr.log'), 'utf8')}`);
  const result = JSON.parse(await readFile(join(cell, 'result.json')));
  const trace = (await readFile(join(cell, 'trace.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
  if (result.status !== 'ok' || (mode === 'steps' && trace.length !== value) ||
      (mode === 'seconds' && result.progress.stopReason !== 'seconds')) throw new Error(`incomplete run: ${cell}`);
  const startupMs = result.timing.optimizeStartEpochMs - spawnEpochMs;
  const record = { label, revision: revisions[label], mode, value, repeat, cell,
    command: [process.execPath, ...command], processMs, spawnEpochMs, startupMs,
    optimizeMs: result.timing.optimizeMs, steps: trace.length, stepMs: result.timing.stepMs,
    first800Ms: trace.length >= 800 ? startupMs + trace[799].elapsedMs : null,
    quality: result.quality, loss: result.loss, system: result.system,
    finalPngSha256: hash(await readFile(join(cell, 'final.png'))),
    lossTraceSha256: hash(JSON.stringify(trace.map((row) => row.loss))),
    psnrBars: Object.fromEntries([20, 21, 21.4, 22].map((bar) => {
      const row = trace.find((row) => -10 * Math.log10(row.loss) >= bar);
      return [bar, row ? { step: row.step, fromLaunchMs: startupMs + row.elapsedMs } : null];
    })) };
  await json(join(cell, 'measurement.json'), record);
  records.push(record);
  await json(join(out, 'measurements.json'), records);
  console.log(`  ${record.stepMs.mean.toFixed(3)} ms/step; startup ${startupMs} ms; ${record.steps} steps; final ${record.quality.psnrDb.toFixed(3)} dB`);
}
for (const [mode, value, count] of [['steps', steps, repeats], ['seconds', seconds, secondsRepeats]]) {
  if (!value) continue;
  for (let repeat = 0; repeat < count; repeat++) {
    for (const label of repeat % 2 ? ['candidate', 'baseline'] : ['baseline', 'candidate']) {
      await run(label, mode, value, repeat);
    }
  }
}
const summarize = (mode) => Object.fromEntries(['baseline', 'candidate'].map((label) => {
  const rows = records.filter((r) => r.mode === mode && r.label === label);
  return [label, rows.length ? Object.fromEntries([
    ['runs', rows.length], ...['processMs', 'startupMs', 'optimizeMs', 'steps', 'first800Ms'].map((key) =>
      [key, median(rows.map((r) => r[key]))]), ['meanStepMs', median(rows.map((r) => r.stepMs.mean))],
    ['finalPsnrDb', median(rows.map((r) => r.quality.psnrDb))],
  ]) : null];
}));
const fixed = summarize('steps'), timed = summarize('seconds');
const fixedRows = records.filter((r) => r.mode === 'steps');
const summary = { config, fixed, timed,
  speedup: { fixedOptimize: fixed.baseline.optimizeMs / fixed.candidate.optimizeMs,
    fixedProcess: fixed.baseline.processMs / fixed.candidate.processMs,
    startup: fixed.baseline.startupMs / fixed.candidate.startupMs,
    timedSteps: timed.baseline ? timed.candidate.steps / timed.baseline.steps : null },
  fixedOutputsIdentical: new Set(fixedRows.map((r) => r.finalPngSha256)).size === 1,
  fixedLossTrajectoriesIdentical: new Set(fixedRows.map((r) => r.lossTraceSha256)).size === 1,
  psnrBarsMeaning: 'First observed training-loss PSNR crossing, measured from process launch. Loss is evaluated before the Adam update; finalPsnrDb scores the final floating-point render.' };
await json(join(out, 'summary.json'), summary);
console.log(JSON.stringify(summary, null, 2));
console.log(`Artifacts: ${out}`);
