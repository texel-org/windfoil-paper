import { WINDFOIL_ENVIRONMENTS } from '../tools/windfoil-runtime.js';

export const BENCH_ENVIRONMENTS = [...WINDFOIL_ENVIRONMENTS, 'diffvg', 'bezier'];
export const DEFAULT_ENVIRONMENTS = ['node', 'diffvg', 'bezier'];
// Cheapest and highest-value first; `kodak` last because it dominates runtime.
export const DEFAULT_STAGES = [
  'opt512',
  'big-image',
  'large-n',
  'clip',
  'kodak',
];
export const OPTIONAL_STAGES = ['kodak-long', 'full-schedule', 'farlev-wallclock', 'probes'];
export const ALL_STAGES = [...DEFAULT_STAGES, ...OPTIONAL_STAGES];

const FARLEV = {
  id: 'farlev',
  label: 'Färlev road',
  path: 'fixtures/wikimedia/farlev-dip-in-road.jpg',
  fixture: 'farlev-highres',
};
const KODAK_ALL = Array.from({ length: 24 }, (_, index) => index + 1);
const KODAK_N = [512];
const KODAK_SUBSET = [4, 9, 13, 23];
// Bézier's pruning/densification window and LR schedule require the full budget.
const KODAK_LONG_IMAGES = [5];
const KODAK_LONG_STEPS = 10000;
const CLIP_PROMPT = 'a photo of a lighthouse on a cliff at dusk';

export function expandStageSelectors(values, fallback = DEFAULT_STAGES) {
  const selected = new Set();
  for (const value of values?.length ? values : fallback) {
    if (value === 'all') {
      for (const stage of DEFAULT_STAGES) selected.add(stage);
    } else if (ALL_STAGES.includes(value)) {
      selected.add(value);
    } else {
      throw new Error(`unknown benchmark stage: ${value}`);
    }
  }
  return ALL_STAGES.filter((stage) => selected.has(stage));
}

export function createBenchmarkPlan({
  stages, environments, repeats, seed, force, sourceRevision,
  // `kodak-long` only. Defaults reproduce the trial cell; the full corpus run
  // passes --images=1..24 with the same step budget.
  images = KODAK_LONG_IMAGES, steps: longSteps = KODAK_LONG_STEPS, seconds: longSeconds = null,
}) {
  const selected = new Set(stages);
  const suites = [];
  if (selected.has('opt512')) {
    for (const target of [FARLEV]) {
      suites.push(suite({
        id: `opt512/${target.id}`,
        stage: 'opt512',
        target,
        protocol: 'controlled-opt512',
        optSize: 512,
        environments,
        variants: ['crisp'],
        n: [256, 512, 4096],
        steps: [800],
      }));
    }
  }

  // Separate suites keep fixed-step and fixed-time protocols separate.
  if (selected.has('big-image')) {
    suites.push(suite({
      id: 'big-image/farlev-4096-60s',
      stage: 'big-image',
      target: FARLEV,
      protocol: 'controlled-4k-wall-clock',
      optSize: 4096,
      environments,
      variants: ['crisp'],
      n: [256],
      seconds: [60],
    }));
    suites.push(suite({
      id: 'big-image/farlev-4096',
      stage: 'big-image',
      target: FARLEV,
      protocol: 'controlled-4k-800-step',
      optSize: 4096,
      environments,
      variants: ['crisp'],
      n: [256],
      steps: [800],
    }));
  }

  if (selected.has('large-n')) {
    suites.push(suite({
      id: 'large-n/farlev-4096',
      stage: 'large-n',
      target: FARLEV,
      protocol: 'windfoil-large-n-4k',
      optSize: 4096,
      environments: environments.filter((value) => WINDFOIL_ENVIRONMENTS.includes(value)),
      variants: ['crisp'],
      n: [10000, 50000],
      steps: [800],
    }));
  }

  // Bézier is excluded because its adapter is L2-only.
  if (selected.has('clip')) {
    suites.push(suite({
      id: 'clip/lighthouse',
      stage: 'clip',
      target: null,
      loss: 'clip',
      prompt: CLIP_PROMPT,
      protocol: 'controlled-clip-128',
      optSize: 128,
      environments: environments.filter((value) => value !== 'bezier'),
      n: [256],
      steps: [800],
    }));
  }

  // Last of the default stages: 24 images x 3 engines, with a DiffVG cell in
  // each, is most of the sweep's wall clock. Ordering it after the cheap ones
  // means an interrupted run still yields every other headline number.
  if (selected.has('kodak')) {
    for (const index of KODAK_ALL) {
      const target = kodakTarget(index);
      suites.push(suite({
        id: `kodak/${target.id}`,
        stage: 'kodak',
        target,
        dataset: 'kodak24',
        protocol: 'controlled-800-step-native',
        optSize: 'max',
        environments,
        n: KODAK_N,
        steps: [800],
      }));
    }
  }

  // Same corpus and shape count as `kodak`, but on a budget every engine's own
  // schedule was written for, and timed from process spawn rather than from the
  // first optimiser step. One image by default for a look at the figures; the
  // full 24 for the published mean.
  if (selected.has('kodak-long')) {
    for (const index of images) {
      const target = kodakTarget(index);
      suites.push(suite({
        id: `kodak-long/${target.id}`,
        stage: 'kodak-long',
        target,
        dataset: 'kodak24',
        protocol: longSeconds
          ? `controlled-${longSeconds}s-wall-clock-native`
          : `controlled-${longSteps}-step-native`,
        optSize: 'max',
        environments,
        n: KODAK_N,
        ...(longSeconds ? { seconds: [longSeconds] } : { steps: [longSteps] }),
      }));
    }
  }

  if (selected.has('full-schedule')) {
    suites.push(suite({
      id: `full-schedule/${FARLEV.id}-10k`,
      stage: 'full-schedule',
      target: FARLEV,
      protocol: 'full-schedule-10k',
      optSize: 512,
      environments: environments.filter((value) => value !== 'diffvg'),
      variants: ['crisp'],
      n: [512, 4096],
      steps: [10000],
    }));
    for (const index of KODAK_SUBSET) {
      const target = kodakTarget(index);
      suites.push(suite({
        id: `full-schedule/${target.id}-10k`,
        stage: 'full-schedule',
        target,
        dataset: 'kodak24',
        protocol: 'full-schedule-10k',
        optSize: 'max',
        environments: environments.filter((value) => value === 'bezier'),
        variants: [],
        n: [256, 512],
        steps: [10000],
      }));
    }
  }

  if (selected.has('farlev-wallclock')) {
    suites.push(suite({
      id: 'farlev-wallclock/farlev-300s', stage: 'farlev-wallclock',
      target: FARLEV, protocol: 'convergence-wallclock', optSize: 512,
      environments, variants: ['crisp'], n: [512], seconds: [300],
    }));
  }

  if (selected.has('probes')) {
    for (const size of [512, 1024, 2048, 4096]) {
      suites.push(suite({
        id: `probes/farlev-${size}`,
        stage: 'probes',
        target: FARLEV,
        protocol: 'resolution-one-step-probe',
        optSize: size,
        environments,
        variants: ['crisp'],
        n: [256, 4096],
        steps: [1],
      }));
    }
  }

  const activeSuites = suites.filter((entry) => entry.environments.length > 0);
  const activeStages = new Set(activeSuites.map((entry) => entry.stage));
  const warnings = stages
    .filter((stage) => !activeStages.has(stage))
    .map((stage) => `${stage} has no work for the selected environments`);
  return {
    schema_version: 1,
    source_revision: sourceRevision,
    selection: { stages, environments, repeats, seed, force },
    warnings,
    suites: activeSuites.map((entry) => ({ ...entry, repeats, seed })),
  };
}

function suite({
  id, stage, target, dataset = null, loss = 'l2', prompt = null, protocol, optSize, environments,
  // Crisp only: see bench/suite.js on why the annealed variant is not the
  // benchmarked configuration. Pass variants explicitly to compare both.
  variants = ['crisp'], n, steps = [], seconds = [],
}) {
  return {
    id,
    stage,
    output: `suites/${id}`,
    target,
    dataset,
    loss,
    prompt,
    protocol,
    opt_size: optSize,
    environments: [...environments],
    windfoil_variants: variants,
    n,
    budgets: [
      ...steps.map((value) => ({ mode: 'steps', value })),
      ...seconds.map((value) => ({ mode: 'seconds', value })),
    ],
  };
}

function kodakTarget(index) {
  const suffix = String(index).padStart(2, '0');
  return {
    id: `kodim${suffix}`,
    label: `Kodak ${suffix}`,
    path: `fixtures/kodak/kodim${suffix}.png`,
    fixture: 'kodak24',
  };
}
