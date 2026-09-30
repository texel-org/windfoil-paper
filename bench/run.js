#!/usr/bin/env node
// The benchmark entry point: one resumable command that plans every suite,
// fetches and verifies its fixtures, runs the matrix through bench/suite.js,
// and records the machine it ran on. It writes raw artifacts only -- tables,
// figures and perceptual metrics come from `bench/report.py`, a separate step
// so a long sweep never depends on torch or matplotlib being installed.

import { execFileSync, spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream, existsSync } from 'node:fs';
import { connect } from 'node:net';
import {
  lstat, mkdir, readFile, readdir, rename, stat, unlink, writeFile,
} from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { arg, argList, parseArgs, stamp } from '../demos/util/runtime.js';
import { run as fetchFixtureManifest } from '../tools/fetch-fixtures.js';
import { WINDFOIL_ENVIRONMENTS } from '../tools/windfoil-runtime.js';
import {
  ALL_STAGES,
  BENCH_ENVIRONMENTS,
  DEFAULT_ENVIRONMENTS,
  DEFAULT_STAGES,
  createBenchmarkPlan,
  expandStageSelectors,
} from './plan.js';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SOURCE_INPUTS = [
  'package.json',
  'package-lock.json',
  'deno.json',
  'deno.lock',
  'requirements-clip.txt',
  'js',
  'demos/cli.js',
  'demos/util',
  'tools/windfoil-runtime.js',
  'bench/run.js',
  'bench/plan.js',
  'bench/suite.js',
  'bench/optimizer_state.py',
  'bench/diffvg/run.py',
  'bench/diffvg/requirements.txt',
  'bench/bezier/run.py',
  'bench/bezier/requirements.txt',
];
// Mirrors bench/suite.js's --loss-url default; WF_LOSS_URL overrides both the
// probe here and nothing else, so keep them in step if that default moves.
const DEFAULT_LOSS_URL = 'ws://127.0.0.1:8765';
const GPU_HEADER = [
  'timestamp', 'index', 'name', 'uuid', 'memory_used_mib', 'memory_total_mib',
  'utilization_gpu_percent', 'utilization_memory_percent',
].join(',');

export function parseBenchOptions(argv, now = new Date()) {
  const options = parseArgs(argv);
  const allowed = new Set([
    'out', 'only', 'skip', 'env', 'repeats', 'seed', 'force', 'dry-run', 'help',
    // `kodak-long` only: which Kodak images and how many steps.
    'images', 'steps', 'seconds',
  ]);
  for (const key of Object.keys(options)) {
    if (!allowed.has(key)) throw new Error(`unknown option: --${key}`);
  }
  if (options.help) return { help: true };

  const only = options.only === undefined
    ? [...DEFAULT_STAGES]
    : expandStageSelectors(argList(options, 'only').map(String), []);
  const skipped = new Set(options.skip === undefined
    ? []
    : expandStageSelectors(argList(options, 'skip').map(String), []));
  const stages = only.filter((stage) => !skipped.has(stage));
  if (!stages.length) throw new Error('stage selection is empty');

  const environments = unique(argList(options, 'env', DEFAULT_ENVIRONMENTS).map(String));
  if (!environments.length) throw new Error('--env must not be empty');
  for (const environment of environments) {
    if (!BENCH_ENVIRONMENTS.includes(environment)) {
      throw new Error(`unsupported environment: ${environment}`);
    }
  }

  const outValue = arg(options, 'out', join('output', `${stamp(now)}-bench`));
  if (outValue === true || !String(outValue).trim()) throw new Error('--out requires a path');
  return {
    help: false,
    out: String(outValue),
    stages,
    environments,
    repeats: integer(arg(options, 'repeats', 1), 'repeats'),
    seed: integer(arg(options, 'seed', 1), 'seed', true),
    force: flag(arg(options, 'force', false)),
    dryRun: flag(arg(options, 'dry-run', false)),
    ...(options.images === undefined ? {} : { images: kodakImages(options) }),
    ...(options.steps === undefined ? {} : { steps: integer(arg(options, 'steps', 0), 'steps') }),
    ...(options.seconds === undefined ? {} : { seconds: integer(arg(options, 'seconds', 0), 'seconds') }),
  };
}

// `--images=5` or `--images=1-24` or `--images=4,9,13`. Only Kodak indices, so
// anything outside 1..24 is a typo rather than a request.
function kodakImages(options) {
  const values = [];
  for (const entry of argList(options, 'images').map(String)) {
    for (const part of entry.split(',')) {
      const range = part.trim().match(/^(\d+)-(\d+)$/);
      if (range) {
        const [from, to] = [Number(range[1]), Number(range[2])];
        if (from > to) throw new Error(`--images range is inverted: ${part}`);
        for (let index = from; index <= to; index += 1) values.push(index);
      } else if (part.trim()) {
        values.push(integer(part.trim(), 'images'));
      }
    }
  }
  if (!values.length) throw new Error('--images must not be empty');
  for (const index of values) {
    if (index < 1 || index > 24) throw new Error(`--images is 1..24, got ${index}`);
  }
  return unique(values);
}

export async function sourceFingerprint(repo = REPO) {
  const paths = [];
  for (const input of SOURCE_INPUTS) await collectFiles(resolve(repo, input), repo, paths);
  paths.sort();
  const hash = createHash('sha256');
  for (const path of paths) {
    const bytes = await readFile(resolve(repo, path));
    hash.update(path).update('\0').update(String(bytes.length)).update('\0').update(bytes);
  }
  return `sha256:${hash.digest('hex')}`;
}

export function suiteArguments(suite, { repo = REPO, outputRoot, sourceRevision, force = false }) {
  const budgets = suite.budgets.flatMap(({ mode, value }) => [`--${mode}=${value}`]);
  const args = [
    'bench/suite.js',
    `--loss=${suite.loss ?? 'l2'}`,
    ...(suite.target ? [
      `--target=${resolve(repo, suite.target.path)}`,
      `--target-label=${suite.target.label}`,
      `--source-id=${suite.target.id}`,
    ] : []),
    ...(suite.prompt ? [`--prompt=${suite.prompt}`] : []),
    `--source-revision=${sourceRevision}`,
    `--protocol=${suite.protocol}`,
    `--env=${suite.environments.join(',')}`,
    `--n=${suite.n.join(',')}`,
    `--opt-size=${suite.opt_size}`,
    `--repeats=${suite.repeats}`,
    `--seed=${suite.seed}`,
    '--schedule-steps=800',
    `--out=${resolve(outputRoot, suite.output)}`,
    ...budgets,
  ];
  if (suite.dataset) args.push(`--dataset=${suite.dataset}`);
  if (suite.windfoil_variants.length) {
    args.push(`--windfoil-variant=${suite.windfoil_variants.join(',')}`);
  }
  if (force) args.push('--force');
  return args;
}

export async function verifyPlanFixtures(plan, repo = REPO) {
  const manifests = new Map();
  const checked = [];
  for (const target of uniqueTargets(plan.suites)) {
    let expected = null;
    if (target.fixture) {
      let manifest = manifests.get(target.fixture);
      if (!manifest) {
        const path = resolve(repo, 'fixtures', 'manifests', `${target.fixture}.json`);
        manifest = JSON.parse(await readFile(path, 'utf8'));
        manifests.set(target.fixture, manifest);
      }
      const destination = resolve(repo, manifest.destination);
      expected = manifest.files.find((file) => resolve(destination, file.path) === resolve(repo, target.path));
      if (!expected) throw new Error(`${target.path} is not declared by ${target.fixture}`);
      expected = { ...expected, fixture: target.fixture };
    }
    checked.push(await verifyFixtureFile(resolve(repo, target.path), expected, repo));
  }
  const invalid = checked.filter((entry) => entry.status !== 'ok');
  return {
    ok: invalid.length === 0,
    files: checked,
    commands: unique(invalid
      .map((entry) => entry.fixture)
      .filter(Boolean)
      .map((id) => `node tools/fetch-fixtures.js ${id}`)),
  };
}

export async function verifyFixtureFile(path, expected = null, repo = REPO) {
  const result = {
    path: relative(repo, path).split('\\').join('/'),
    fixture: expected?.fixture ?? null,
    status: 'ok',
  };
  let info;
  try {
    info = await lstat(path);
  } catch (error) {
    if (error?.code === 'ENOENT') return { ...result, status: 'missing' };
    throw error;
  }
  if (info.isSymbolicLink()) return { ...result, status: 'symlink' };
  if (!info.isFile()) return { ...result, status: 'not-a-file' };
  const pathFromRepo = relative(resolve(repo), resolve(path));
  if (pathFromRepo === '..' || pathFromRepo.startsWith(`..${sep}`) || isAbsolute(pathFromRepo)) {
    return { ...result, status: 'outside-repo' };
  }
  let parent = resolve(repo);
  for (const part of pathFromRepo.split(sep).slice(0, -1)) {
    parent = join(parent, part);
    const parentInfo = await lstat(parent);
    if (parentInfo.isSymbolicLink()) return { ...result, status: 'symlink-parent' };
    if (!parentInfo.isDirectory()) return { ...result, status: 'invalid-parent' };
  }
  if (expected?.bytes !== undefined && info.size !== expected.bytes) {
    return { ...result, status: 'wrong-size', bytes: info.size, expected_bytes: expected.bytes };
  }
  if (expected?.sha256) {
    const digest = await hashFile(path);
    if (digest !== expected.sha256.toLowerCase()) {
      return { ...result, status: 'wrong-sha256', sha256: digest };
    }
  }
  return { ...result, bytes: info.size, sha256: expected?.sha256?.toLowerCase() };
}

export function countPlanCells(plan) {
  return plan.suites.reduce((total, suite) => {
    const windfoil = suite.environments
      .filter((environment) => WINDFOIL_ENVIRONMENTS.includes(environment)).length;
    const competitors = suite.environments.length - windfoil;
    const engines = windfoil * suite.windfoil_variants.length + competitors;
    return total + suite.n.length * suite.budgets.length * suite.repeats * engines;
  }, 0);
}

export async function fetchMissingPlanFixtures(verification, fetchFixture = fetchFixtureManifest) {
  const invalid = verification.files.filter((entry) => entry.status !== 'ok');
  if (invalid.some((entry) => entry.status !== 'missing' || !entry.fixture)) return [];
  const ids = unique(invalid.map((entry) => entry.fixture));
  for (const id of ids) {
    console.log(`fetching fixture ${id}`);
    await fetchFixture([id]);
  }
  return ids;
}

export async function runBench(argv = process.argv.slice(2), dependencies = {}) {
  const repo = dependencies.repo ?? REPO;
  const options = parseBenchOptions(argv, dependencies.now?.() ?? new Date());
  if (options.help) {
    usage();
    return 0;
  }

  const outputRoot = resolve(repo, options.out);
  await mkdir(outputRoot, { recursive: true });
  const sourceRevision = dependencies.sourceRevision ?? await sourceFingerprint(repo);
  const plan = createBenchmarkPlan({
    stages: options.stages,
    environments: options.environments,
    repeats: options.repeats,
    seed: options.seed,
    force: options.force,
    ...(options.images === undefined ? {} : { images: options.images }),
    ...(options.steps === undefined ? {} : { steps: options.steps }),
    ...(options.seconds === undefined ? {} : { seconds: options.seconds }),
    sourceRevision,
  });
  if (!plan.suites.length) throw new Error('the selected stages have no suites for --env');
  const planPath = join(outputRoot, options.dryRun ? 'plan.dry-run.json' : 'plan.json');
  await writeJsonAtomic(planPath, plan);

  const planHash = digestJson(plan);
  const statePath = join(outputRoot, options.dryRun ? 'state.dry-run.json' : 'state.json');
  const previous = await readJson(statePath);
  const state = makeState(plan, planHash, previous, options.dryRun);
  await writeJsonAtomic(statePath, state);

  const verifyFixtures = dependencies.verifyFixtures ?? verifyPlanFixtures;
  let verification;
  try {
    verification = await verifyFixtures(plan, repo);
    state.fixtures = verification;
    state.updated_at = isoNow(dependencies);
    if (options.dryRun) {
      state.status = 'dry-run';
      await writeJsonAtomic(statePath, state);
      console.log(`wrote ${relative(repo, planPath)} ` +
        `(${plan.suites.length} suites, ${countPlanCells(plan)} cells)`);
      if (!verification.ok) console.warn(fixtureError(verification).message);
      return 0;
    }
    if (!verification.ok) {
      const fetched = await fetchMissingPlanFixtures(
        verification,
        dependencies.fetchFixtures ?? fetchFixtureManifest,
      );
      if (fetched.length) {
        verification = await verifyFixtures(plan, repo);
        verification.fetched = fetched;
        state.fixtures = verification;
        state.updated_at = isoNow(dependencies);
        await writeJsonAtomic(statePath, state);
      }
    }
  } catch (error) {
    state.status = 'error';
    state.fixture_error = error?.message ?? String(error);
    state.updated_at = isoNow(dependencies);
    await writeJsonAtomic(statePath, state);
    throw error;
  }
  if (!verification.ok) {
    state.status = 'error';
    state.fixture_error = fixtureError(verification).message;
    await writeJsonAtomic(statePath, state);
    throw new Error(state.fixture_error);
  }

    // The CLIP stage needs a loss server that nothing else in the sweep does, and
  // it runs after an hour of L2 work -- so check it now rather than letting a
  // reviewer discover it from two failed cells much later.
  if (plan.suites.some((entry) => entry.loss === 'clip')) {
    const url = process.env.WF_LOSS_URL ?? DEFAULT_LOSS_URL;
    if (!await canConnect(url)) {
      throw new Error(`the clip stage needs a loss server at ${url}, which is not accepting ` +
        'connections.\nStart it with `npm run clip:server`, or re-run with `--skip=clip`.');
    }
  }

  // Written before the first cell so an interrupted sweep is still attributable
  // to a machine. It reads the host it runs on, which is the only place the
  // GPU, driver and Vulkan device are visible -- result.json records none of
  // them, and a pod without a working NVIDIA ICD silently reports `llvmpipe`.
  await writeJsonAtomic(join(outputRoot, 'env.json'), {
    schema_version: 1,
    ...(dependencies.environment ?? environment)(repo, options, isoNow(dependencies)),
  });

  const runSuite = dependencies.runSuite ?? ((suite) => runSuiteProcess(suite, {
    repo,
    outputRoot,
    sourceRevision,
    force: options.force,
    spawnImpl: dependencies.spawnImpl ?? spawn,
  }));
  const failures = [];
  for (const suite of plan.suites) {
    const entry = state.suites[suite.id];
    entry.status = 'running';
    entry.attempts += 1;
    entry.started_at = isoNow(dependencies);
    delete entry.completed_at;
    delete entry.exit_code;
    await writeJsonAtomic(statePath, state);

    console.log(`\n[${suite.id}] ${suite.environments.join(',')} N=${suite.n.join(',')} @ ${suite.opt_size}`);
    const started = performance.now();
    let result;
    try {
      result = await runSuite(suite);
    } catch (error) {
      result = { code: null, error };
    }
    entry.duration_ms = performance.now() - started;
    entry.completed_at = isoNow(dependencies);
    entry.exit_code = result.code;
    if (result.code === 0) entry.status = 'ok';
    else {
      entry.status = 'error';
      entry.error = result.error?.message ?? `benchmark runner exited with ${result.code}`;
      failures.push(suite.id);
    }
    state.updated_at = entry.completed_at;
    await writeJsonAtomic(statePath, state);
  }

  state.status = failures.length ? 'error' : 'ok';
  state.completed_at = isoNow(dependencies);
  state.updated_at = state.completed_at;
  state.failures = failures;
  await writeJsonAtomic(statePath, state);
  console.log(`\nwrote ${relative(repo, outputRoot)}`);
  console.log(`next: ${reportPython(repo)} bench/report.py ${relative(repo, outputRoot)}`);
  if (failures.length) console.error(`${failures.length} suite${failures.length === 1 ? '' : 's'} failed`);
  return failures.length ? 1 : 0;
}

async function runSuiteProcess(suite, { repo, outputRoot, sourceRevision, force, spawnImpl }) {
  const suiteRoot = resolve(outputRoot, suite.output);
  await mkdir(suiteRoot, { recursive: true });
  const stdout = createWriteStream(join(suiteRoot, 'runner.stdout.log'), { flags: 'a' });
  const stderr = createWriteStream(join(suiteRoot, 'runner.stderr.log'), { flags: 'a' });
  const header = `\n== ${new Date().toISOString()} ==\n`;
  stdout.write(header);
  stderr.write(header);
  const monitor = await startGpuMonitor(join(suiteRoot, 'gpu.csv'), spawnImpl);
  const child = spawnImpl(process.execPath, suiteArguments(suite, {
    repo, outputRoot, sourceRevision, force,
  }), { cwd: repo, env: process.env, stdio: ['ignore', 'pipe', 'pipe'] });
  child.stdout?.on('data', (chunk) => {
    stdout.write(chunk);
    process.stdout.write(chunk);
  });
  child.stderr?.on('data', (chunk) => {
    stderr.write(chunk);
    process.stderr.write(chunk);
  });
  const result = await processResult(child);
  await monitor.stop();
  await Promise.all([closeStream(stdout), closeStream(stderr)]);
  return result;
}

export async function startGpuMonitor(out, spawnImpl = spawn) {
  const temporary = `${out}.part`;
  const stream = createWriteStream(temporary, { flags: 'w' });
  stream.write(`${GPU_HEADER}\n`);
  let samples = false;
  let unavailable = false;
  const child = spawnImpl('nvidia-smi', [
    '--query-gpu=timestamp,index,name,uuid,memory.used,memory.total,utilization.gpu,utilization.memory',
    '--format=csv,noheader,nounits',
    '--loop-ms=1000',
  ], { stdio: ['ignore', 'pipe', 'ignore'] });
  child.stdout?.on('data', (chunk) => {
    samples = true;
    stream.write(chunk);
  });
  child.on('error', () => { unavailable = true; });

  return {
    async stop() {
      if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM');
      await waitForChild(child, 2000);
      await closeStream(stream);
      if (samples && !unavailable) await rename(temporary, out);
      else await unlink(temporary).catch(() => {});
    },
  };
}

// Everything here is best-effort: a missing tool means a null field, never a
// failed sweep. `sh` swallows non-zero exits for exactly that reason.
export function environment(repo = REPO, options = null, date = new Date().toISOString()) {
  const gpu = sh('nvidia-smi', [
    '--query-gpu=name,memory.total,driver_version,compute_cap', '--format=csv,noheader',
  ]);
  const memKb = firstLine(sh('bash', ['-lc', "grep -m1 MemTotal /proc/meminfo | awk '{print $2}'"]));
  const torchVersion = (python) => firstLine(sh(python, ['-c',
    'import torch;print(f"torch {torch.__version__} / CUDA {torch.version.cuda}")']));
  const submodule = (path) => firstLine(sh('git', ['-C', repo, 'rev-parse', `HEAD:${path}`]));
  return {
    date,
    argv: process.argv.slice(2),
    selection: options && {
      stages: options.stages,
      environments: options.environments,
      repeats: options.repeats,
      seed: options.seed,
      out: options.out,
    },
    os: firstLine(sh('bash', ['-lc', '. /etc/os-release && echo "$PRETTY_NAME"'])),
    gpu: gpu ? gpu.split(',').map((part) => part.trim()) : null,
    // Dawn falls back to the llvmpipe CPU rasterizer when the pod exposes only
    // the compute half of the driver, which makes every Windfoil timing a CPU
    // number. This field is how a sweep is caught after the fact.
    vulkan_device: firstLine(sh('bash', ['-lc',
      'vulkaninfo --summary 2>/dev/null | grep -m1 deviceName | cut -d= -f2'])),
    cpu: firstLine(sh('bash', ['-lc', "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"])),
    cpu_cores: Number(sh('nproc', [])) || null,
    memory_gb: memKb ? Math.round(Number(memKb) / 1024 / 1024) : null,
    nvcc: firstLine(sh('bash', ['-lc',
      "nvcc --version 2>/dev/null | grep -o 'release [0-9.]*, V[0-9.]*'"])),
    node: process.version,
    windfoil_commit: firstLine(sh('git', ['-C', repo, 'rev-parse', '--short', 'HEAD'])),
    diffvg_torch: torchVersion(process.env.DIFFVG_PYTHON ?? join(repo, '.venv-diffvg/bin/python')),
    bezier_torch: torchVersion(process.env.BEZIER_PYTHON ?? join(repo, '.venv-bezier/bin/python')),
    bezier_upstream: submodule('bench/bezier/upstream'),
    bezier_gsplat: submodule('bench/bezier/gsplat'),
  };
}

// The report needs matplotlib and torch, which live in a venv rather than the
// system Python. Name an interpreter that can actually run it, so the printed
// command can be pasted: a bare `python3` is usually wrong on a pod, and a
// relative `.venv/bin/python` is wrong from anywhere but the repo root.
// Existence is not enough -- pod-setup.sh puts the report dependencies in
// `.venv-bezier` (it already has CUDA torch) while `.venv` holds the CLIP
// server, so picking the first venv that exists names one that cannot import
// matplotlib. Probe instead of guessing at the layout.
function reportPython(repo) {
  if (process.env.REPORT_PYTHON) return process.env.REPORT_PYTHON;
  const candidates = [
    join(repo, '.venv', 'bin', 'python'),
    join(repo, '.venv-bezier', 'bin', 'python'),
    join(repo, '.venv-diffvg', 'bin', 'python'),
    'python3',
  ].filter((path) => path === 'python3' || existsSync(path));
  const usable = candidates.find((path) => {
    try {
      execFileSync(path, ['-c', 'import matplotlib'], { stdio: 'ignore', timeout: 20_000 });
      return true;
    } catch {
      return false;
    }
  });
  // Nothing can run it yet: name the most likely venv anyway, so the message
  // points somewhere useful and report.py explains the missing dependency.
  return usable ?? candidates[0];
}

// A TCP connect, not a websocket handshake: enough to prove something is
// listening, with no dependency and no protocol assumptions.
async function canConnect(url, timeoutMs = 2000) {
  let target;
  try {
    target = new URL(url);
  } catch {
    return false;
  }
  const port = Number(target.port) || (target.protocol === 'wss:' ? 443 : 80);
  return await new Promise((settle) => {
    const socket = connect({ host: target.hostname, port });
    const finish = (ok) => {
      socket.destroy();
      settle(ok);
    };
    socket.setTimeout(timeoutMs);
    socket.once('connect', () => finish(true));
    socket.once('timeout', () => finish(false));
    socket.once('error', () => finish(false));
  });
}

function sh(command, args) {
  try {
    return execFileSync(command, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
  } catch {
    return null;
  }
}

function firstLine(value) {
  return value ? value.split('\n')[0].trim() : null;
}

function makeState(plan, planHash, previous, dryRun) {
  const keep = previous?.plan_sha256 === planHash ? previous.suites ?? {} : {};
  const suites = {};
  for (const suite of plan.suites) {
    suites[suite.id] = {
      output: suite.output,
      status: dryRun ? 'planned' : 'pending',
      attempts: Number(keep[suite.id]?.attempts ?? 0),
    };
  }
  return {
    schema_version: 1,
    plan_sha256: planHash,
    status: dryRun ? 'planning' : 'running',
    updated_at: new Date().toISOString(),
    suites,
  };
}

function fixtureError(verification) {
  const invalid = verification.files
    .filter((entry) => entry.status !== 'ok')
    .map((entry) => `- ${entry.path}: ${entry.status}`);
  const force = verification.files.some((entry) => entry.status !== 'ok' && entry.status !== 'missing');
  const commands = verification.commands.map((command) => `  ${command}${force ? ' --force' : ''}`);
  const hint = commands.length ? ['Fetch and verify them with:', ...commands] : [];
  return new Error(['benchmark fixtures are not ready:', ...invalid, ...hint].join('\n'));
}

function uniqueTargets(suites) {
  const targets = new Map();
  for (const suite of suites) {
    if (suite.target) targets.set(suite.target.path, suite.target);
  }
  return [...targets.values()];
}

async function collectFiles(path, repo, out) {
  const info = await stat(path);
  if (info.isFile()) {
    out.push(relative(repo, path).split('\\').join('/'));
    return;
  }
  if (!info.isDirectory()) throw new Error(`unsupported source fingerprint input: ${path}`);
  for (const entry of await readdir(path, { withFileTypes: true })) {
    if (entry.name.startsWith('.') || entry.name === '__pycache__') continue;
    await collectFiles(join(path, entry.name), repo, out);
  }
}

async function hashFile(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

function digestJson(value) {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

async function processResult(child) {
  return await new Promise((resolveResult) => {
    let settled = false;
    child.on('error', (error) => {
      if (settled) return;
      settled = true;
      resolveResult({ code: null, signal: null, error });
    });
    child.on('close', (code, signal) => {
      if (settled) return;
      settled = true;
      resolveResult({ code, signal, error: null });
    });
  });
}

async function waitForChild(child, timeoutMs) {
  if (child.exitCode !== null || child.signalCode !== null) return;
  await Promise.race([
    new Promise((resolveWait) => child.once('close', resolveWait)),
    new Promise((resolveWait) => setTimeout(resolveWait, timeoutMs)),
  ]);
  if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
}

async function closeStream(stream) {
  if (stream.closed) return;
  await new Promise((resolveClose, reject) => {
    stream.once('error', reject);
    stream.end(resolveClose);
  });
}

async function writeJsonAtomic(path, value) {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`);
  await rename(temporary, path);
}

async function readJson(path) {
  try {
    return JSON.parse(await readFile(path, 'utf8'));
  } catch (error) {
    if (error?.code === 'ENOENT' || error instanceof SyntaxError) return null;
    throw error;
  }
}

function isoNow(dependencies) {
  return (dependencies.now?.() ?? new Date()).toISOString();
}

function integer(value, name, allowZero = false) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < (allowZero ? 0 : 1)) {
    throw new Error(`--${name} must be a ${allowZero ? 'non-negative' : 'positive'} integer`);
  }
  return number;
}

function flag(value) {
  if ([true, 'true', '1'].includes(value)) return true;
  if ([false, 'false', '0'].includes(value)) return false;
  throw new Error(`expected a boolean flag, received ${value}`);
}

function unique(values) {
  return [...new Set(values)];
}

function usage() {
  console.log(`usage: npm run bench -- [options]\n\n` +
    `  --out=PATH              default: output/<timestamp>-bench\n` +
    `  --only=${ALL_STAGES.join(',')}\n` +
    `  --skip=STAGE[,STAGE]\n` +
    `  --env=${BENCH_ENVIRONMENTS.join(',')}\n` +
    `  --repeats=N --seed=N --force --dry-run\n` +
    `  --images=1-24  --steps=N  --seconds=N   (kodak-long only)\n\n` +
    `then: python3 bench/report.py <out>`);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  runBench().then((code) => { process.exitCode = code; }).catch((error) => {
    console.error(error?.stack ?? error);
    process.exitCode = 1;
  });
}
