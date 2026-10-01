#!/usr/bin/env node
import { existsSync } from 'node:fs';
import { mkdir, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import { join, relative, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const CELL_COLUMNS = [
  'suite', 'run_id', 'engine', 'environment', 'variant', 'style', 'loss',
  'source_id', 'target_label', 'target_sha256', 'protocol', 'n', 'opt_width', 'opt_height',
  'budget', 'repeat', 'seed', 'steps_completed', 'startup_ms', 'optimize_ms',
  'process_ms', 'ms_per_step', 'mse_rgb', 'psnr_db', 'trace_status',
];
const TRACE_COLUMNS = ['run_id', 'step', 'elapsed_ms', 'psnr_db'];

export function csvLine(values) {
  return values.map((value) => {
    const text = value == null ? '' : String(value);
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  }).join(',') + '\n';
}

async function suites(root) {
  if (existsSync(join(root, 'results.jsonl')) && existsSync(join(root, 'config.json'))) return [root];
  const found = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    if (entry.isDirectory() && !['cells', 'report', 'inputs'].includes(entry.name)) {
      found.push(...await suites(join(root, entry.name)));
    }
  }
  return found.sort();
}

function jsonl(text) {
  return text.split('\n').filter((line) => line.trim()).map((line) => JSON.parse(line));
}

// CSVs contain only measured data; report.py derives table aggregates from them.
export async function exportCsv(root, out = root) {
  root = resolve(root);
  await mkdir(out, { recursive: true });
  const cells = [];
  const traces = [csvLine(TRACE_COLUMNS)];
  for (const suite of await suites(root)) {
    const config = JSON.parse(await readFile(join(suite, 'config.json'), 'utf8'));
    const name = (relative(root, suite) || config.suite_id || 'suite').replace(/^suites\//, '');
    const records = jsonl(await readFile(join(suite, 'results.jsonl'), 'utf8'));
    for (const record of records) {
      if (record.status !== 'ok') continue;
      const w = record.workload ?? {}, t = record.timing ?? {}, q = record.quality ?? {};
      const steps = record.progress?.steps_completed;
      const startup = Number.isFinite(t.spawn_epoch_ms) && Number.isFinite(t.optimize_start_epoch_ms)
        ? Math.max(0, t.optimize_start_epoch_ms - t.spawn_epoch_ms) : null;
      const tracePath = record.artifacts?.trace && join(suite, record.artifacts.trace);
      const hasTrace = tracePath && existsSync(tracePath);
      cells.push({
        suite: name, run_id: record.run_id, engine: record.engine?.name,
        environment: record.engine?.environment, variant: w.variant, style: w.engine_options?.style,
        loss: w.loss, source_id: w.source_id ?? config.source_id, target_label: w.target_label,
        target_sha256: config.target_sha256, protocol: w.protocol ?? config.protocol,
        n: w.n, opt_width: w.opt_size?.[0], opt_height: w.opt_size?.[1],
        budget: `${record.budget?.mode}-${record.budget?.requested}`, repeat: w.repeat, seed: w.seed,
        steps_completed: steps, startup_ms: startup, optimize_ms: t.optimize_ms,
        process_ms: t.process_ms, ms_per_step: steps ? t.optimize_ms / steps : null,
        mse_rgb: q.mse_rgb, psnr_db: q.psnr_db, trace_status: hasTrace ? 'ok' : 'missing',
      });
      if (hasTrace && w.loss === 'l2') {
        for (const point of jsonl(await readFile(tracePath, 'utf8'))) {
          const elapsed = point.elapsed_ms ?? point.elapsedMs;
          const psnr = record.engine?.name === 'bezier-splatting' ? point.psnr_self
            : point.loss > 0 ? -10 * Math.log10(point.loss) : null;
          if (Number.isFinite(elapsed) && Number.isFinite(psnr)) {
            traces.push(csvLine([record.run_id, point.step, elapsed, psnr]));
          }
        }
      }
    }
  }
  for (const [name, contents] of [
    ['cells.csv', csvLine(CELL_COLUMNS) + cells.map((row) => csvLine(CELL_COLUMNS.map((key) => row[key]))).join('')],
    ['traces.csv', traces.join('')],
  ]) {
    const path = join(out, name);
    await writeFile(`${path}.tmp`, contents);
    await rename(`${path}.tmp`, path);
  }
  return cells.length;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const [root, out = root] = process.argv.slice(2);
  if (!root) throw new Error('usage: node bench/csv.js RUN [OUT]');
  console.log(`exported ${await exportCsv(root, out)} cells to ${out}`);
}
