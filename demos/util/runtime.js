const deno = typeof Deno !== 'undefined' && !!Deno.version?.deno;
const node = typeof process !== 'undefined' && !!process.versions?.node && !deno;

export const isNode = () => node;
export const args = () => deno ? Deno.args : node ? process.argv.slice(2) : [];

export function parseArgs(argv = args()) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    const eq = token.indexOf('=');
    const key = token.slice(2, eq < 0 ? undefined : eq);
    const value = eq >= 0
      ? token.slice(eq + 1)
      : argv[i + 1] && !argv[i + 1].startsWith('--')
      ? argv[++i]
      : true;
    out[key] = key in out ? [...(Array.isArray(out[key]) ? out[key] : [out[key]]), value] : value;
  }
  return out;
}

export function arg(args, key, fallback) {
  const value = args[key];
  return Array.isArray(value) ? value.at(-1) : value ?? fallback;
}

export function argList(args, key, fallback = [], { comma = true } = {}) {
  const value = args[key] ?? fallback;
  const values = Array.isArray(value) ? value : [value];
  return values.flatMap((item) => {
    if (typeof item !== 'string') return [item];
    const text = item.trim();
    if (text.startsWith('[')) {
      const parsed = JSON.parse(text);
      if (!Array.isArray(parsed)) throw new Error(`--${key} JSON value must be an array`);
      return parsed;
    }
    if (!comma) return [item];
    return text.split(',').map((part) => part.trim()).filter(Boolean);
  });
}

// Scalar option validators shared by the CLI runner and the model/loss
// descriptors. Each reads a single `--key` value and enforces its shape.
export function numberArg(options, key, fallback) {
  const raw = arg(options, key, fallback);
  if (typeof raw === 'boolean') throw new Error(`--${key} requires a value`);
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`--${key} must be a number`);
  return value;
}

export function integerArg(options, key, fallback) {
  const value = numberArg(options, key, fallback);
  if (!Number.isInteger(value) || value < 1)
    throw new Error(`--${key} must be a positive integer`);
  return value;
}

export function nonNegativeIntegerArg(options, key, fallback) {
  const value = numberArg(options, key, fallback);
  if (!Number.isInteger(value) || value < 0)
    throw new Error(`--${key} must be a non-negative integer`);
  return value;
}

export function positiveArg(options, key, fallback) {
  const value = numberArg(options, key, fallback);
  if (value <= 0) throw new Error(`--${key} must be positive`);
  return value;
}

export function booleanArg(options, key, fallback) {
  const value = arg(options, key, fallback);
  if (value === true || value === 'true') return true;
  if (value === false || value === 'false') return false;
  throw new Error(`--${key} must be true or false`);
}

// --color-count: 1 (a single/fixed palette) or the learned-palette size (>= 2).
export function paletteCount(options) {
  if (!('color-count' in options)) return 1;
  const count = integerArg(options, 'color-count');
  if (count < 2) throw new Error('--color-count must be an integer >= 2');
  return count;
}

// --palette-fidelity: 0 (best fit) .. higher pulls the learned palette toward
// the image's actual colors (perceptual, OKLab). Off by default.
export function paletteFidelity(options) {
  if (!('palette-fidelity' in options)) return 0;
  const value = numberArg(options, 'palette-fidelity');
  if (value < 0) throw new Error('--palette-fidelity must be >= 0');
  return value;
}

// Resolve shape opacity for a learned palette: --opaque / --translucent override
// the per-model default (plot opaque, others translucent).
export function opaqueMode(options, defaultOpaque) {
  const opaque = 'opaque' in options;
  const translucent = 'translucent' in options;
  if (opaque && translucent) {
    throw new Error('--opaque and --translucent are mutually exclusive');
  }
  return opaque ? true : translucent ? false : defaultOpaque;
}

export async function mkdir(path) {
  if (deno) return Deno.mkdir(path, { recursive: true });
  const { mkdir } = await import('node:fs/promises');
  return mkdir(path, { recursive: true });
}

export async function readBytes(path) {
  if (deno) return Deno.readFile(path);
  const { readFile } = await import('node:fs/promises');
  const data = await readFile(path);
  return new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
}

export async function readText(path) {
  if (deno) return Deno.readTextFile(path);
  const { readFile } = await import('node:fs/promises');
  return readFile(path, 'utf8');
}

// Sorted entry names of a directory, or null when `path` is not a directory.
export async function listDirectory(path) {
  try {
    if (deno) {
      if (!(await Deno.stat(path)).isDirectory) return null;
      return (await Array.fromAsync(Deno.readDir(path))).map((entry) => entry.name).sort();
    }
    const { readdir, stat } = await import('node:fs/promises');
    if (!(await stat(path)).isDirectory()) return null;
    return (await readdir(path)).sort();
  } catch (error) {
    if (error?.code === 'ENOENT' || error?.name === 'NotFound') return null;
    throw error;
  }
}

export async function writeBytes(path, data) {
  if (deno) return Deno.writeFile(path, data);
  const { writeFile } = await import('node:fs/promises');
  return writeFile(path, data);
}

export async function writeText(path, data) {
  if (deno) return Deno.writeTextFile(path, data);
  const { writeFile } = await import('node:fs/promises');
  return writeFile(path, data);
}

export function stamp(date = new Date()) {
  const p = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}${p(date.getMonth() + 1)}${p(date.getDate())}-` +
    `${p(date.getHours())}${p(date.getMinutes())}${p(date.getSeconds())}`;
}

export function slug(value, fallback = 'run') {
  return String(value ?? fallback).toLowerCase().replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '').slice(0, 48) || fallback;
}

// Enough decimals to stay readable as a loss shrinks toward zero.
export function formatLoss(value) {
  if (!Number.isFinite(value) || value === 0) return String(value);
  const magnitude = Math.abs(value);
  if (magnitude < 5e-16) return value < 0 ? '>-0.000000000000001' : '<0.000000000000001';
  const decimals = Math.min(15, Math.max(6, 5 - Math.floor(Math.log10(magnitude))));
  return value.toFixed(decimals).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
}

export function exitCleanly() {
  if (node) setTimeout(() => process.exit(0), 0);
}

export async function runMain(main) {
  try {
    await main(args());
    exitCleanly();
  } catch (error) {
    console.error(error?.stack ?? error);
    if (node) process.exit(1);
    Deno.exit(1);
  }
}
