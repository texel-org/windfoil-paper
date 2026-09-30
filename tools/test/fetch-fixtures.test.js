import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { createBenchmarkPlan } from '../../bench/plan.js';
import { LOSSES } from '../../demos/util/losses.js';
import { manifestIds, validateFixtureManifest } from '../fetch-fixtures.js';

async function manifest(id) {
  const json = await readFile(new URL(`../../fixtures/manifests/${id}.json`, import.meta.url), 'utf8');
  return validateFixtureManifest(JSON.parse(json), id);
}

test('Kodak manifest records the complete official set', async () => {
  const value = await manifest('kodak24');
  assert.equal(value.files.length, 24);
  assert.equal(value.files.reduce((sum, file) => sum + file.bytes, 0), 15_394_305);
  assert.deepEqual(value.files.map((file) => file.path),
    Array.from({ length: 24 }, (_, index) => `kodim${String(index + 1).padStart(2, '0')}.png`));
});

test('every declared manifest is valid and reachable through `all`', async () => {
  const ids = await manifestIds();
  assert.ok(ids.length, 'expected at least one manifest');
  for (const id of ids) await manifest(id);
});

test('high-resolution manifest pins the Färlev source', async () => {
  const value = await manifest('farlev-highres');
  assert.deepEqual(value.protocol.nativeResolution, [4925, 2770]);
  assert.deepEqual(value.protocol.benchmarkLongSides, [512, 1024, 2048, 4096]);
  assert.equal(value.files[0].sha256,
    '242277090148fdc8b984f6b4b4fa43cd9a299396eb7dd656d5c00f6e44badada');
});

// No imagery is tracked, so the demos and the benchmark can only find their
// target where the manifest puts it.
test('the L2 default target and the benchmark target are the file the Färlev manifest downloads', async () => {
  const [file] = (await manifest('farlev-highres')).files;
  const fromRoot = (relative) => fileURLToPath(new URL(`../../${relative}`, import.meta.url));
  assert.equal(fromRoot(LOSSES.l2.subjectDefault), file.target);
  assert.match(LOSSES.l2.subjectDefaultHint, /npm run fixtures:farlev/);

  const plan = createBenchmarkPlan({
    stages: ['opt512', 'full-schedule'], environments: ['node'], repeats: 1, seed: 7,
  });
  const imageTargets = plan.suites.map((suite) => suite.target).filter((target) => target?.id === 'farlev');
  assert.ok(imageTargets.length >= 2);
  for (const target of imageTargets) {
    assert.equal(target.fixture, 'farlev-highres');
    assert.equal(fromRoot(target.path), file.target);
  }
});

test('manifest validation rejects paths outside fixtures', () => {
  assert.throws(() => validateFixtureManifest({
    version: 1,
    id: 'bad',
    destination: '../bad',
    files: [{ path: 'x', url: 'https://example.com/x', bytes: 1 }],
  }, 'bad'), /below fixtures/);
});
