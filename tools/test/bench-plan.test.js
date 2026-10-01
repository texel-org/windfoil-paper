import assert from 'node:assert/strict';
import test from 'node:test';

import { DEFAULT_ENVIRONMENTS, DEFAULT_STAGES, createBenchmarkPlan } from '../../bench/plan.js';

const plan = createBenchmarkPlan({
  stages: DEFAULT_STAGES, environments: DEFAULT_ENVIRONMENTS, repeats: 1, seed: 7,
});

// Every L2 image-fitting result is reported on the equal-step protocol, so the
// default sweep has to produce the 4K one for all three engines unprompted.
test('the default sweep runs the 4K Färlev cell at 800 steps on every engine', () => {
  const suite = plan.suites.find((entry) => entry.id === 'big-image/farlev-4096');
  assert.ok(suite, 'big-image/farlev-4096 is missing from the default sweep');
  assert.deepEqual(suite.environments, DEFAULT_ENVIRONMENTS);
  assert.deepEqual(suite.budgets, [{ mode: 'steps', value: 800 }]);
  assert.equal(suite.opt_size, 4096);
  assert.deepEqual(suite.n, [256]);
});

// bench/report.py averages cells that share a suite, an engine and a shape
// count as repeats of one another, so two protocols must never share a suite.
test('no suite mixes a step budget with a wall-clock budget', () => {
  for (const suite of plan.suites) {
    const modes = new Set(suite.budgets.map((budget) => budget.mode));
    assert.equal(modes.size, 1, `${suite.id} mixes ${[...modes].join(' and ')} budgets`);
  }
});

test('suite ids and output directories are unique', () => {
  const ids = plan.suites.map((suite) => suite.id);
  const outputs = plan.suites.map((suite) => suite.output);
  assert.equal(new Set(ids).size, ids.length);
  assert.equal(new Set(outputs).size, outputs.length);
});

test('the paper convergence stage uses 512px, N=512 and a 300-second budget', () => {
  const selected = createBenchmarkPlan({
    stages: ['farlev-wallclock'], environments: DEFAULT_ENVIRONMENTS, repeats: 1, seed: 7,
  });
  assert.equal(selected.suites.length, 1);
  const [suite] = selected.suites;
  assert.equal(suite.opt_size, 512);
  assert.deepEqual(suite.n, [512]);
  assert.deepEqual(suite.budgets, [{ mode: 'seconds', value: 300 }]);
  assert.deepEqual(suite.environments, DEFAULT_ENVIRONMENTS);
});

test('suite launch contains one CLI entrypoint followed by flags', async () => {
  const { suiteArguments } = await import('../../bench/run.js');
  const args = suiteArguments(plan.suites[0], {
    repo: '/repo', outputRoot: '/output', sourceRevision: 'test',
  });
  assert.equal(args[0], 'bench/suite.js');
  assert.ok(args.slice(1).every((arg) => arg.startsWith('--')));
});
