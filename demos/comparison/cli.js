// Coverage error against the box filter (the paper's comparison figure).

import { getWebGPUHostInfo, requestDevice } from '../../js/renderer.js';
import { rgbaToPng } from '../util/image.js';
import { mkdir, parseArgs, runMain, stamp, writeBytes, writeText } from '../util/runtime.js';
import { cropRGBA, diffRGBA, grayRGBA, WHITE } from './images.js';
import { comparisonOptions } from './options.js';
import { skiaCoverage, windfoilCoverage } from './renderers.js';
import { loadFont, resolveScene, sceneKind } from './scenes.js';
import { slugCoverage } from './slug.js';
import { coverageStats, pointCoverage } from './truth.js';

await runMain(async (argv) => {
  const options = comparisonOptions(parseArgs(argv));
  const { size, samples, amp, offset, fit, crop } = options;
  const needsFont = options.scenes.some((spec) => sceneKind(spec).kind === 'glyph');
  const font = needsFont ? await loadFont() : null;
  const scenes = [];
  for (const spec of options.scenes) {
    const scene = await resolveScene(spec, { font, size, offset, fit });
    if (scenes.some((other) => other.key === scene.key || other.slug === scene.slug)) {
      throw new Error(`--scene ${spec} shares its output names with another scene`);
    }
    for (const warning of scene.warnings) console.warn(`${spec}: ${warning}`);
    scenes.push(scene);
  }

  const device = await requestDevice();
  const host = getWebGPUHostInfo();
  const root = options.out ?? `output/comparison/${stamp()}-${host.environment}`;
  await mkdir(root);

  try {
    for (const scene of scenes) {
      const { quads, evenodd } = scene;
      const common = { size, evenodd };
      // name/title are the stats.json schema; figure is the file name in the paper's figure.
      const renderers = [
        {
          name: 'truth',
          title: `ideal box filter, ${samples}×${samples} point-sampled`,
          reference: true,
          run: () => pointCoverage(quads, { ...common, samples }),
        },
        {
          name: 'binary',
          title: 'winding at the pixel centre, no AA',
          run: () => pointCoverage(quads, { ...common, samples: 1 }),
        },
        {
          name: 'windfoil',
          title: 'js/renderer.js, box filter, s = 1px',
          figure: 'wf',
          run: () => windfoilCoverage(device, quads, common),
        },
        {
          name: 'slug',
          title: 'demos/comparison/slug.wgsl',
          figure: 'slug',
          run: () => slugCoverage(device, quads, common),
        },
        {
          name: 'chrome',
          title: '@napi-rs/canvas (Skia)',
          figure: 'chrome',
          run: () => skiaCoverage(quads, common),
        },
      ];

      const results = [];
      for (const r of renderers) {
        const started = performance.now();
        const cov = await r.run();
        results.push({ ...r, cov, ms: performance.now() - started });
      }
      const truth = results.find((r) => r.reference).cov;
      for (const r of results) r.stats = r.reference ? null : coverageStats(r.cov, truth);

      const prefix = `${root}/cmp-${scene.key}`;
      const png = (path, rgba) => writeBytes(path, rgbaToPng(rgba, size, size));
      const cropped = (rgba) => cropRGBA(rgba, size, size, crop);
      const truthRGBA = grayRGBA(truth);
      const writes = [
        png(`${prefix}-truth.png`, truthRGBA),
        png(`${prefix}-truth-crop.png`, cropped(truthRGBA)),
        png(`${prefix}-wf-crop.png`, cropped(grayRGBA(results.find((r) => r.name === 'windfoil').cov))),
      ];
      for (const r of results.filter((r) => r.figure)) {
        writes.push(png(`${prefix}-${r.figure}-diff.png`, cropped(diffRGBA(r.cov, truth, amp, WHITE))));
      }
      writes.push(writeText(`${root}/${scene.slug}__stats.json`, JSON.stringify({
        scene: scene.spec,
        label: scene.label,
        slug: scene.slug,
        size,
        samples,
        amp,
        offset,
        exact: false, // schema parity: the renderer has no exact-sampling mode
        quads: quads.length / 6,
        renderers: results.map((r) => ({
          name: r.name,
          title: r.title,
          reference: !!r.reference,
          mean: r.stats?.mean ?? null,
          max: r.stats?.max ?? null,
          ms: +r.ms.toFixed(1),
        })),
      }, null, 2) + '\n'));
      await Promise.all(writes);

      console.log(`${scene.label} · ${size}x${size} · ${quads.length / 6} quads`);
      for (const r of results) {
        const mean = r.stats ? r.stats.mean.toFixed(6) : '-';
        const max = r.stats ? r.stats.max.toFixed(6) : '-';
        console.log(`  ${r.name.padEnd(9)} mean ${mean.padStart(8)}  max ${max.padStart(8)}  ` +
          `${r.ms.toFixed(0).padStart(5)} ms`);
      }
    }
  } finally {
    device.destroy();
  }
  console.log(`wrote ${scenes.length * 6} PNGs and ${scenes.length} stats files to ${root}`);
});
