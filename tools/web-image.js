#!/usr/bin/env node

// Derives the web demo's default target from the Färlev fixture, capped at
// 4096 px on its long side so the page doesn't ship the 14 MB original. The
// CLI demos and benchmarks keep using the full-resolution file.

import { rename, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import jpeg from 'jpeg-js';
import { loadImageSource, resizeImage } from '../demos/util/image.js';

const LONG_SIDE = 4096;
const QUALITY = 90;

const script = fileURLToPath(import.meta.url);
const directory = path.resolve(path.dirname(script), '../fixtures/wikimedia');
const input = path.join(directory, 'farlev-dip-in-road.jpg');
const output = path.join(directory, `farlev-dip-in-road-${LONG_SIDE}.jpg`);

const mtime = (file) => stat(file).then((info) => info.mtimeMs, () => -Infinity);

const inputTime = await mtime(input);
if (inputTime === -Infinity) {
  console.log('web image: no Färlev fixture (npm run fixtures:farlev); the demo starts without a target');
} else if (await mtime(output) < Math.max(inputTime, await mtime(script))) {
  const source = await loadImageSource(input);
  const scale = Math.min(1, LONG_SIDE / Math.max(source.width, source.height));
  const width = Math.round(source.width * scale);
  const height = Math.round(source.height * scale);
  const rgba = resizeImage(source, width, height);
  const data = new Uint8Array(rgba.length);
  for (let i = 0; i < rgba.length; i++) data[i] = Math.round(Math.min(1, Math.max(0, rgba[i])) * 255);
  await writeFile(`${output}.partial`, jpeg.encode({ data, width, height }, QUALITY).data);
  await rename(`${output}.partial`, output);
  console.log(`web image: ${width} × ${height} → ${path.relative(process.cwd(), output)}`);
}
