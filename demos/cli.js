import { runCli } from './util/run.js';
import { arg, parseArgs, runMain } from './util/runtime.js';

// The geometry axis, resolved once: --primitive names the geometry, and each
// command declares which model draws each primitive it offers. Raster commands
// share one table — filled `shape` loops train as the quadratic-loop model;
// `capsule`/`line` strokes and `point` discs train as the stroke model. The
// pen-plotter command draws its round-capped strokes and stipple discs with the
// cm-based plot model, and offers only those two.
const RASTER = { shape: 'shape', capsule: 'line', line: 'line', point: 'line' };
const PLOTTER = { line: 'plot', point: 'plot' };

// Each command pairs a default objective (--loss overrides it) with a default
// primitive (--primitive overrides it) and the primitive→model table above.
const commands = {
  l2: { loss: 'l2', primitive: 'shape', models: RASTER },
  clip: { loss: 'clip', primitive: 'shape', models: RASTER },
  lines: { loss: 'l2', primitive: 'capsule', models: RASTER },
  plot: { loss: 'l2', primitive: 'line', models: PLOTTER },
};
const LOSS_KINDS = ['l2', 'clip'];

await runMain(async ([command, ...options]) => {
  const selected = commands[command];
  if (!selected) {
    throw new Error(`usage: demos/cli.js <${Object.keys(commands).join('|')}> [options]`);
  }
  // Loss, geometry, and command defaults are orthogonal: --loss fits the model
  // to any objective (`lines --loss=clip`) and --primitive fits any of the
  // command's geometries under that objective (`clip --primitive=capsule`).
  const parsed = parseArgs(options);
  const loss = 'loss' in parsed ? String(arg(parsed, 'loss')) : selected.loss;
  if (!LOSS_KINDS.includes(loss)) {
    throw new Error(`--loss must be one of ${LOSS_KINDS.join(', ')}`);
  }
  const primitive = 'primitive' in parsed
    ? String(arg(parsed, 'primitive'))
    : selected.primitive;
  const modelKind = selected.models[primitive];
  if (!modelKind) {
    throw new Error(
      `--primitive for ${command} must be one of ${Object.keys(selected.models).join(', ')}`,
    );
  }
  await runCli(loss, options, { modelKind, invocation: [command, ...options] });
});
