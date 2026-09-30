#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { lstat, mkdir, readFile, readdir, rename, stat, unlink } from 'node:fs/promises';
import path from 'node:path';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const fixtureRoot = path.join(repoRoot, 'fixtures');
const manifestRoot = path.join(fixtureRoot, 'manifests');
const LARGE_BYTES = 1024 ** 3;

function inside(root, target) {
  const relative = path.relative(root, target);
  return relative !== '' && !relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative);
}

function assertObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
}

export function validateFixtureManifest(value, expectedId = value?.id) {
  assertObject(value, 'manifest');
  if (value.version !== 1) throw new Error('manifest.version must be 1');
  if (!/^[a-z0-9-]+$/.test(value.id) || value.id !== expectedId) {
    throw new Error(`invalid manifest id: ${JSON.stringify(value.id)}`);
  }
  if (typeof value.destination !== 'string') throw new Error('manifest.destination must be a string');

  const destination = path.resolve(repoRoot, value.destination);
  if (!inside(fixtureRoot, destination) || destination === manifestRoot || inside(manifestRoot, destination)) {
    throw new Error('manifest.destination must be below fixtures/');
  }
  if (!Array.isArray(value.files) || value.files.length === 0) {
    throw new Error('manifest.files must be a non-empty array');
  }

  const targets = new Set();
  const files = value.files.map((file, index) => {
    const label = `manifest.files[${index}]`;
    assertObject(file, label);
    if (typeof file.path !== 'string' || file.path.length === 0) throw new Error(`${label}.path is invalid`);
    const target = path.resolve(destination, file.path);
    if (!inside(destination, target)) throw new Error(`${label}.path escapes its destination`);
    if (targets.has(target)) throw new Error(`${label}.path is duplicated`);
    targets.add(target);

    let url;
    try {
      url = new URL(file.url);
    } catch {
      throw new Error(`${label}.url is invalid`);
    }
    if (url.protocol !== 'https:') throw new Error(`${label}.url must use HTTPS`);
    if (!Number.isSafeInteger(file.bytes) || file.bytes <= 0) throw new Error(`${label}.bytes is invalid`);
    if (file.sha256 !== undefined && !/^[a-f0-9]{64}$/i.test(file.sha256)) {
      throw new Error(`${label}.sha256 is invalid`);
    }
    return { ...file, url: url.href, target };
  });

  return { ...value, destination, files };
}

async function sha256(file) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}

async function verify(file, spec) {
  let info;
  try {
    info = await stat(file);
  } catch (error) {
    if (error.code === 'ENOENT') return { valid: false, reason: 'missing' };
    throw error;
  }
  if (!info.isFile()) return { valid: false, reason: 'not a regular file' };
  if (info.size !== spec.bytes) return { valid: false, reason: `${info.size} bytes, expected ${spec.bytes}` };
  if (spec.sha256) {
    const actual = await sha256(file);
    if (actual !== spec.sha256.toLowerCase()) return { valid: false, reason: `sha256 ${actual}` };
  }
  return { valid: true };
}

async function makeSafeDirectory(target) {
  await mkdir(fixtureRoot, { recursive: true });
  const relative = path.relative(fixtureRoot, target);
  let current = fixtureRoot;
  for (const part of relative.split(path.sep)) {
    current = path.join(current, part);
    try {
      const info = await lstat(current);
      if (!info.isDirectory() || info.isSymbolicLink()) throw new Error(`unsafe fixture directory: ${current}`);
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      await mkdir(current);
    }
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KiB', 'MiB', 'GiB'];
  let value = bytes;
  let unit = 'B';
  for (const next of units) {
    value /= 1024;
    unit = next;
    if (value < 1024) break;
  }
  return `${value.toFixed(value >= 10 ? 1 : 2)} ${unit}`;
}

async function download(spec, { force }) {
  await makeSafeDirectory(path.dirname(spec.target));
  try {
    const info = await lstat(spec.target);
    if (info.isSymbolicLink()) throw new Error(`refusing symlink: ${spec.target}`);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }

  const current = await verify(spec.target, spec);
  if (current.valid) {
    console.log(`ok   ${path.relative(repoRoot, spec.target)}`);
    return;
  }
  if (current.reason !== 'missing' && !force) {
    throw new Error(`${path.relative(repoRoot, spec.target)} is invalid (${current.reason}); pass --force to replace it`);
  }

  const response = await fetch(spec.url, { headers: { 'accept-encoding': 'identity' } });
  if (!response.ok || !response.body) throw new Error(`${spec.url}: HTTP ${response.status}`);
  if (new URL(response.url).protocol !== 'https:') throw new Error(`${spec.url}: redirected away from HTTPS`);
  const contentLength = response.headers.get('content-length');
  const announced = contentLength === null ? null : Number(contentLength);
  if (announced !== null && Number.isFinite(announced) && announced !== spec.bytes) {
    throw new Error(`${spec.url}: server announced ${announced} bytes, expected ${spec.bytes}`);
  }

  const temporary = `${spec.target}.part-${process.pid}`;
  let received = 0;
  let lastUpdate = 0;
  const hash = createHash('sha256');
  const meter = new Transform({
    transform(chunk, _encoding, callback) {
      received += chunk.length;
      hash.update(chunk);
      const now = Date.now();
      if (now - lastUpdate >= 5000) {
        process.stdout.write(`get  ${path.basename(spec.target)} ${formatBytes(received)} / ${formatBytes(spec.bytes)}\n`);
        lastUpdate = now;
      }
      callback(null, chunk);
    },
  });

  try {
    await pipeline(Readable.fromWeb(response.body), meter, createWriteStream(temporary, { flags: 'wx' }));
    if (received !== spec.bytes) throw new Error(`${spec.url}: received ${received} bytes, expected ${spec.bytes}`);
    const digest = hash.digest('hex');
    if (spec.sha256 && digest !== spec.sha256.toLowerCase()) {
      throw new Error(`${spec.url}: sha256 ${digest}, expected ${spec.sha256}`);
    }
    await rename(temporary, spec.target);
  } catch (error) {
    await unlink(temporary).catch(() => {});
    throw error;
  }
  console.log(`done ${path.relative(repoRoot, spec.target)} (${formatBytes(received)})`);
}

export async function manifestIds() {
  const entries = await readdir(manifestRoot);
  return entries.filter((name) => name.endsWith('.json'))
    .map((name) => name.slice(0, -'.json'.length))
    .sort();
}

function usage() {
  console.log('usage: node tools/fetch-fixtures.js <manifest-id|all>... [--accept-large] [--force]\n\n' +
    '  all   every declared fixture; re-verifies size and sha256 of what is\n' +
    '        already present, and downloads only what is missing');
}

export async function run(argv) {
  if (argv.includes('--help') || argv.includes('-h')) {
    usage();
    return;
  }
  const options = new Set(argv.filter((arg) => arg.startsWith('--')));
  for (const option of options) {
    if (!['--accept-large', '--force'].includes(option)) throw new Error(`unknown option: ${option}`);
  }
  const requested = argv.filter((arg) => !arg.startsWith('--'));
  if (!requested.length || requested.some((id) => !/^[a-z0-9-]+$/.test(id))) {
    usage();
    throw new Error('expected at least one manifest id');
  }
  const declared = await manifestIds();
  const ids = [...new Set(requested.flatMap((id) => (id === 'all' ? declared : [id])))];

  for (const id of ids) {
    const manifestFile = path.join(manifestRoot, `${id}.json`);
    let source;
    try {
      source = await readFile(manifestFile, 'utf8');
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      throw new Error(`unknown fixture: ${id}\navailable: ${declared.join(', ')}, all`);
    }
    const manifest = validateFixtureManifest(JSON.parse(source), id);
    const large = manifest.files.filter((file) => file.bytes > LARGE_BYTES);
    if (large.length > 0 && !options.has('--accept-large')) {
      throw new Error(`${id} contains ${large.map((file) => formatBytes(file.bytes)).join(', ')} download(s); pass --accept-large`);
    }

    for (const file of manifest.files) await download(file, { force: options.has('--force') });
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  run(process.argv.slice(2)).catch((error) => {
    console.error(`Error: ${error.message}`);
    process.exitCode = 1;
  });
}
