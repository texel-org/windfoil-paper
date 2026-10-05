import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { unzlibSync } from 'fflate';
import {
  ChunkType,
  ColorType,
  decode,
  decode_pHYs_PPI,
  readChunks,
  readIHDR,
} from 'png-tools';

import {
  blitRgba8,
  blitRgba16,
  blurredTargetProvider,
  boxBlurImage,
  imageToPng,
  letterboxRgba,
  loadImageSource,
  rgbaToPng,
} from '../../demos/util/image.js';

const JPEG = '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAABAAIDAREAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDs/gx/yR7wn/2LVh/6TpX+Yeef8jvFf9fJ/wDpTP8AMj6UX/KTPG//AGN8y/8AU2sf/9k=';
const jpegAvailable = await import('jpeg-js').then(() => true).catch(() => false);

test('letterbox paints outside the content rect and blends partial edge pixels', () => {
  // Opaque 4x1 canvas; content covers x in [1, 2.5), so pixel 2 is half
  // content and half bar.
  const canvas = Uint8Array.of(
    9, 9, 9, 255,
    255, 255, 255, 255,
    255, 255, 255, 255,
    9, 9, 9, 255,
  );
  letterboxRgba(canvas, 4, 1, { x: 1, y: 0, width: 1.5, height: 1 }, [1, 0, 0]);
  assert.deepEqual([...canvas], [
    255, 0, 0, 255,
    255, 255, 255, 255,
    255, 128, 128, 255,
    255, 0, 0, 255,
  ]);

  const transparent = new Uint16Array(4 * 3);
  letterboxRgba(transparent, 1, 3, { x: 0, y: 1, width: 1, height: 1.5 }, [0, 0, 1], { depth: 16 });
  assert.deepEqual([...transparent.slice(0, 4)], [0, 0, 65535, 65535]);
  assert.deepEqual([...transparent.slice(4, 8)], [0, 0, 0, 0]);
  assert.deepEqual([...transparent.slice(8)], [0, 0, 65535, 32768]);
});

test('blit quantizes an f32 region into an opaque 8-bit canvas', () => {
  const canvas = new Uint8Array(4 * 3 * 4);
  const tile = Float32Array.of(
    1, 0, 0, 0.5, 0, 2, 0, 1, // top row: red, over-range green
    0, 0, 0.5, 1, 0.2, 0.2, 0.2, 1, // clipped by the 2x1 blit region
  );
  blitRgba8(canvas, 4, 1, 1, tile, 2, 2, 1);
  const pixel = (x, y) => [...canvas.subarray(4 * (y * 4 + x), 4 * (y * 4 + x) + 4)];
  assert.deepEqual(pixel(1, 1), [255, 0, 0, 255]);
  assert.deepEqual(pixel(2, 1), [0, 255, 0, 255]);
  for (const [x, y] of [[0, 1], [3, 1], [1, 0], [1, 2]]) assert.deepEqual(pixel(x, y), [0, 0, 0, 0]);
});

test('transparent blits unpremultiply color and preserve analytic alpha', () => {
  const canvas = new Uint8Array(4);
  blitRgba8(
    canvas,
    1,
    0,
    0,
    Float32Array.of(0.125, 0.25, 0.375, 0.5),
    1,
    1,
    1,
    { transparent: true },
  );
  assert.deepEqual([...canvas], [64, 128, 191, 128]);
});

test('16-bit transparent blits retain high-precision straight-alpha samples', () => {
  const canvas = new Uint16Array(4);
  blitRgba16(
    canvas,
    1,
    0,
    0,
    Float32Array.of(0.125, 0.25, 0.375, 0.5),
    1,
    1,
    1,
    { transparent: true },
  );
  assert.deepEqual([...canvas], [16384, 32768, 49151, 32768]);
});

test('PNG output carries requested print resolution metadata', () => {
  const png = rgbaToPng(Uint8Array.of(0, 0, 0, 0), 1, 1, { ppi: 300 });
  assert.deepEqual([...decode(png, unzlibSync).data], [0, 0, 0, 0]);
  const chunks = readChunks(png);
  const physical = chunks.find(({ type }) => type === ChunkType.pHYs);
  assert.ok(physical);
  assert.ok(Math.abs(decode_pHYs_PPI(physical.data) - 300) < 0.01);
  assert.throws(
    () => rgbaToPng(Uint8Array.of(0, 0, 0, 0), 1, 1, { ppi: 0 }),
    /positive and finite/,
  );
});

test('16-bit PNG output preserves samples and print resolution metadata', () => {
  const samples = Uint16Array.of(1, 257, 32768, 65535);
  const png = rgbaToPng(samples, 1, 1, { depth: 16, ppi: 300 });
  assert.deepEqual(readIHDR(png), {
    width: 1,
    height: 1,
    depth: 16,
    colorType: ColorType.RGBA,
    compression: 0,
    filter: 0,
    interlace: 0,
  });
  assert.deepEqual([...decode(png, unzlibSync).data], [...samples]);
  const physical = readChunks(png).find(({ type }) => type === ChunkType.pHYs);
  assert.ok(physical);
  assert.ok(Math.abs(decode_pHYs_PPI(physical.data) - 300) < 0.01);
});

test('image source loads PNG by magic bytes', async () => {
  const root = await mkdtemp(join(tmpdir(), 'windfoil-png-'));
  const path = join(root, 'image.data');
  try {
    await writeFile(path, imageToPng(new Float32Array([1, 0, 0, 1]), 1, 1));
    const source = await loadImageSource(path);
    assert.deepEqual([source.width, source.height, source.channels, source.depth], [1, 1, 4, 8]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('box blur matches the renderer filter footprint', () => {
  // A width-2 box centered on a pixel takes half of each neighbor.
  const impulse = new Float32Array(5 * 1 * 4);
  impulse[4 * 2] = 1;
  const blurred = boxBlurImage(impulse, 5, 1, 2);
  const red = [0, 1, 2, 3, 4].map((x) => blurred[4 * x]);
  assert.deepEqual(red, [0, 0.25, 0.5, 0.25, 0]);
  assert.equal(blurred[3], 1);

  // Constant images are unchanged for any filter size, thanks to edge clamp.
  const flat = new Float32Array(4 * 3 * 4).fill(0.6);
  for (const value of boxBlurImage(flat, 4, 3, 3.5)) {
    assert.ok(Math.abs(value - (value > 0.9 ? 1 : 0.6)) < 1e-6);
  }

  // Size 1 and below reduce to the identity kernel.
  const same = boxBlurImage(impulse, 5, 1, 1);
  assert.deepEqual(Array.from(same.filter((_, i) => i % 4 === 0)), [0, 0, 1, 0, 0]);
});

test('blurred target provider buckets and caches filter sizes', () => {
  const rgba = new Float32Array(4 * 1 * 4).fill(0.5);
  const provider = blurredTargetProvider(rgba, 4, 1);
  assert.equal(provider(1), rgba);
  assert.equal(provider(1.1), rgba);
  const coarse = provider(3.01);
  assert.equal(provider(2.99), coarse);
  assert.notEqual(coarse, rgba);
  assert.notEqual(provider(3.6), coarse);
  assert.throws(() => blurredTargetProvider(rgba, 4, 1, 0.5, 'gauss'), /unknown filter kernel/);
});

test('blurred targets match each kernel convolution order', () => {
  const impulse = new Float32Array(9 * 4);
  impulse[4 * 4] = 1;
  const red = (kernel) => Array.from(
    blurredTargetProvider(impulse, 9, 1, 0.5, kernel)(2)
      .filter((_, index) => index % 4 === 0),
  );
  const box = red('box');
  const tent = red('tent');
  const cubic = red('cubic');
  assert.deepEqual(box, [0, 0, 0, 0.25, 0.5, 0.25, 0, 0, 0]);
  assert.notDeepEqual(tent, box);
  assert.notDeepEqual(cubic, tent);
  for (const values of [box, tent, cubic]) {
    assert.ok(Math.abs(values.reduce((sum, value) => sum + value, 0) - 1) < 1e-7);
  }
});

test('image source loads JPEG by magic bytes', { skip: jpegAvailable ? false : 'run npm install' }, async () => {
  const root = await mkdtemp(join(tmpdir(), 'windfoil-jpeg-'));
  const path = join(root, 'image.data');
  try {
    await writeFile(path, Buffer.from(JPEG, 'base64'));
    const source = await loadImageSource(path);
    assert.deepEqual([source.width, source.height, source.channels, source.depth], [2, 1, 4, 8]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
