// Target preparation benchmark, optionally compared against a saved image.js.
// Usage: node tools/perf-image.js --baseline=PATH --out=output/perf/image.json
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { resizeImage } from '../demos/util/image.js';

const args = Object.fromEntries(process.argv.slice(2).map((arg) => {
  const [key, ...value] = arg.replace(/^--/, '').split('=');
  return [key, value.join('=')];
}));
let original;
if (args.baseline) {
  // The resize function is self-contained. Extract it so a baseline copy need
  // not resolve the image module's dependencies from its temporary directory.
  const source = await readFile(args.baseline, 'utf8');
  const start = source.indexOf('export function resizeImage(');
  const end = source.indexOf('export function boxBlurImage(', start);
  if (start < 0 || end < 0) throw new Error('baseline must be a Windfoil image.js');
  original = Function(source.slice(start, end).replace('export ', '') + '\nreturn resizeImage;')();
}
const repetitions = Number(args.repeats ?? 5);
if (!Number.isInteger(repetitions) || repetitions < 1) throw new Error('repeats must be positive');
const sizes = (args.sizes ?? '64,128,512,1024').split(',').map(Number);
if (sizes.some((n) => !Number.isInteger(n) || n < 1)) throw new Error('sizes must be positive integers');
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
function time(fn, source, width, height) {
  const started = performance.now();
  const image = fn(source, width, height);
  return { ms: performance.now() - started, image };
}
const cases = [];
for (const depth of [8, 16]) {
  for (const channels of [1, 2, 3, 4]) {
    const Samples = depth === 16 ? Uint16Array : Uint8Array;
    const data = Samples.from({ length: 768 * 512 * channels }, (_, i) => i * 103 + i % 47);
    const source = { width: 768, height: 512, depth, channels, data };
    // Noninteger scaling, single-pixel images, and clamped borders are checked
    // separately from the timed workload, which keeps source aspect ratio.
    if (original) {
      for (const [width, height] of [[1, 1], [35, 29], [769, 513]]) {
        check(resizeImage(source, width, height), original(source, width, height));
      }
    }
    for (const width of sizes) {
      const height = Math.max(1, Math.round(width * 2 / 3));
      for (let i = 0; i < 3; i++) {
        resizeImage(source, width, height);
        original?.(source, width, height);
      }
      const current = [], baseline = [];
      let a, b;
      for (let i = 0; i < repetitions; i++) {
        // Alternate order to reduce the effect of transient CPU clock changes.
        if (original && i % 2) { b = time(original, source, width, height); baseline.push(b.ms); }
        a = time(resizeImage, source, width, height); current.push(a.ms);
        if (original && !(i % 2)) { b = time(original, source, width, height); baseline.push(b.ms); }
      }
      if (b) check(a.image, b.image);
      cases.push({ width, height, depth, channels, ms: median(current),
        ...(b ? { baselineMs: median(baseline), speedup: median(baseline) / median(current), exact: true } : {}) });
    }
  }
}
function check(a, b) {
  if (a.length !== b.length) throw new Error('resized image length changed');
  for (let i = 0; i < a.length; i++) {
    if (!Object.is(a[i], b[i])) throw new Error(`resized image changed at ${i}: ${a[i]} vs ${b[i]}`);
  }
}
const result = { date: new Date().toISOString(), node: process.version, repetitions, cases };
const file = resolve(args.out ?? 'output/perf/image.json');
await mkdir(dirname(file), { recursive: true });
await writeFile(file, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(cases, null, 2));
console.log(`wrote ${file}`);
