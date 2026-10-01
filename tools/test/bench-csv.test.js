import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { csvLine, exportCsv } from '../../bench/csv.js';

test('benchmark CSV export preserves measured times, precision, traces and missing values', async () => {
  const root = await mkdtemp(join(tmpdir(), 'windfoil-csv-'));
  try {
    const suite = join(root, 'suites', 'example');
    await mkdir(suite, { recursive: true });
    await writeFile(join(suite, 'config.json'), JSON.stringify({ source_id: 'farlev' }));
    const record = {
      status: 'ok', run_id: 'w', engine: { name: 'windfoil', environment: 'node' },
      workload: { loss: 'l2', n: 512, opt_size: [512, 288], target_label: 'road, "photo"' },
      budget: { mode: 'steps', requested: 800 }, progress: { steps_completed: 800 },
      timing: { spawn_epoch_ms: 1000, optimize_start_epoch_ms: 1750, optimize_ms: 1600 },
      quality: { psnr_db: 21.83019768184741 }, artifacts: { trace: 'trace.jsonl' },
    };
    const missing = { ...record, run_id: 'missing', timing: {}, artifacts: {} };
    await writeFile(join(suite, 'results.jsonl'), [record, missing, { status: 'error' }].map(JSON.stringify).join('\n'));
    await writeFile(join(suite, 'trace.jsonl'), '{"step":1,"elapsedMs":2,"loss":0.01}\n');
    assert.equal(await exportCsv(root), 2);
    const cells = await readFile(join(root, 'cells.csv'), 'utf8');
    assert.match(cells, /"road, ""photo"""/);
    assert.match(cells, /750,1600,,2,/);
    assert.match(cells, /21\.83019768184741,ok/);
    assert.match(cells, /missing\n/);
    assert.equal(await readFile(join(root, 'traces.csv'), 'utf8'), 'run_id,step,elapsed_ms,psnr_db\nw,1,2,20\n');
    assert.equal(csvLine(['a\nb', null]), '"a\nb",\n');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
