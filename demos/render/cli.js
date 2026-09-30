import { getWebGPUHostInfo, requestDevice } from '../../js/renderer.js';
import { packScene } from '../../js/prep.js';
import { parseColor } from '../util/color.js';
import { letterboxRgba, rgbaToPng } from '../util/image.js';
import { runPath } from '../util/output.js';
import {
  arg,
  argList,
  booleanArg,
  listDirectory,
  mkdir,
  parseArgs,
  readText,
  runMain,
  writeBytes,
  writeText,
} from '../util/runtime.js';
import { parseFillSvg, rasterDimensions, rasterSize, SCALE_MODES } from './svg.js';
import { renderChunked } from './tiled.js';

const DEFAULT_SVG = 'demos/render/spokes.svg';

await runMain(async (argv) => {
  const options = parseArgs(argv);
  const widthOption = arg(options, 'width', null);
  const heightOption = arg(options, 'height', null);
  if ((widthOption === null) !== (heightOption === null)) {
    throw new Error('--width and --height must be provided together');
  }
  if (widthOption !== null && 'dimension' in options) {
    throw new Error('--dimension cannot be combined with --width and --height');
  }
  const dimension = widthOption === null
    ? positiveInteger(arg(options, 'dimension', 1024), 'dimension')
    : null;
  const outputWidth = widthOption === null ? null : positiveInteger(widthOption, 'width');
  const outputHeight = heightOption === null ? null : positiveInteger(heightOption, 'height');
  const scaleMode = String(arg(options, 'scale-mode', 'expand'));
  if (!SCALE_MODES.includes(scaleMode)) {
    throw new Error(`--scale-mode must be one of ${SCALE_MODES.join(', ')}`);
  }
  const letterboxColor = parseColor(arg(options, 'letterbox-color', '#000000'));
  const dpi = positiveNumber(arg(options, 'dpi', 300), 'dpi');
  const depth = pngDepth(arg(options, 'depth', 8));
  const debug = booleanArg(options, 'debug', false);
  const chunkOption = arg(options, 'chunk', null);
  const chunk = chunkOption === null ? null : positiveInteger(chunkOption, 'chunk');
  const fallbackBackground = parseBackground(arg(options, 'background', null));
  const { sources, batch } = await resolveSources(argList(options, 'svg', [DEFAULT_SVG]));
  // Parse every input before touching the GPU so a bad file fails the batch early.
  const jobs = await Promise.all(sources.map(async (sourcePath) => {
    const source = await readText(sourcePath);
    let parsed;
    try {
      parsed = parseFillSvg(source);
    } catch (error) {
      throw new Error(`${sourcePath}: ${error.message}`);
    }
    if (!parsed.shapes.length) throw new Error(`${sourcePath}: SVG must contain at least one filled path`);
    return { sourcePath, source, ...parsed };
  }));

  const device = await requestDevice();
  const host = getWebGPUHostInfo();
  const root = runPath(host.environment, 'render', arg(options, 'out', null));
  await mkdir(root);

  try {
    for (const { sourcePath, source, viewBox, background, shapes } of jobs) {
      const name = batch ? basename(sourcePath) : 'render';
      const scene = packScene(shapes);
      const raster = widthOption === null
        ? rasterSize(viewBox, dimension)
        : rasterDimensions(viewBox, outputWidth, outputHeight, { mode: scaleMode });
      const renderBackground = background ?? fallbackBackground;
      const transparent = renderBackground === null;
      if (debug) {
        console.log(
          `${sourcePath}: parsed ${shapes.length} paths, ${scene.curveCount} curves; ` +
            `${transparent ? 'transparent' : 'opaque'} ${raster.width}x${raster.height} ${scaleMode} output`,
        );
      }
      const started = performance.now();
      const { rgba, plan } = await renderChunked(device, scene, raster, {
        s: [raster.scale, raster.scale],
        bg: renderBackground ?? [0, 0, 0],
        transparent,
        depth,
        chunk,
        onChunk: debug
          ? ({ index, total, x, y, width, height, ms }) =>
              console.log(`chunk ${index + 1}/${total} at ${x},${y} ${width}x${height} in ${ms.toFixed(1)}ms`)
          : null,
      });
      if (scaleMode === 'letterbox') {
        letterboxRgba(rgba, raster.width, raster.height, raster.content, letterboxColor, { depth });
      }
      const renderMs = performance.now() - started;
      if (debug) console.log(`encoding ${raster.width}x${raster.height} ${depth}-bit png at ${dpi} dpi`);
      const writes = [
        writeBytes(`${root}/${name}.png`, rgbaToPng(rgba, raster.width, raster.height, { depth, ppi: dpi })),
      ];
      if (debug) {
        const metadata = {
          source: sourcePath,
          environment: host.environment,
          backend: host.backend,
          adapter: host.adapterInfo,
          limits: host.limits,
          viewBox,
          dimension,
          dpi,
          depth,
          width: raster.width,
          height: raster.height,
          scale: raster.scale,
          scaleMode,
          content: raster.content,
          letterboxColor: scaleMode === 'letterbox' ? letterboxColor : null,
          transparent,
          background: renderBackground,
          chunkSize: [plan.tileWidth, plan.tileHeight],
          chunkGrid: [plan.columns, plan.rows],
          shapes: shapes.length,
          fillRules: shapes.map((shape) => shape.fillRule),
          curves: scene.curveCount,
          pieces: scene.pieceCount,
          renderMs,
        };
        writes.push(
          writeText(`${root}/${batch ? name : 'source'}.svg`, source),
          writeText(`${root}/${name}.json`, JSON.stringify(metadata, null, 2) + '\n'),
        );
      }
      await Promise.all(writes);
      if (debug) console.log(`rendered ${raster.width}x${raster.height} to ${root}/${name}.png`);
    }
  } finally {
    device.destroy();
  }
});

// --svg accepts files and directories (every *.svg inside, sorted), repeated
// or comma-separated. A single file keeps the render.png output name; a batch
// writes one <basename>.png per source.
async function resolveSources(paths) {
  const sources = [];
  let directory = false;
  for (const path of paths.map(String)) {
    const entries = await listDirectory(path);
    if (entries === null) {
      sources.push(path);
      continue;
    }
    directory = true;
    const svgs = entries.filter((entry) => entry.toLowerCase().endsWith('.svg'));
    if (!svgs.length) throw new Error(`--svg directory ${path} contains no .svg files`);
    sources.push(...svgs.map((entry) => `${path.replace(/\/+$/, '')}/${entry}`));
  }
  const names = sources.map(basename);
  const duplicate = names.find((name, index) => names.indexOf(name) !== index);
  if (duplicate !== undefined) throw new Error(`--svg inputs share the output name ${duplicate}.png`);
  return { sources, batch: directory || sources.length > 1 };
}

function basename(path) {
  return path.split(/[\\/]/).at(-1).replace(/\.svg$/i, '');
}

function positiveInteger(value, name) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < 1) throw new Error(`--${name} must be a positive integer`);
  return number;
}

function positiveNumber(value, name) {
  const number = Number(value);
  if (!(number > 0) || !Number.isFinite(number)) throw new Error(`--${name} must be positive`);
  return number;
}

function pngDepth(value) {
  const depth = Number(value);
  if (depth !== 8 && depth !== 16) throw new Error('--depth must be 8 or 16');
  return depth;
}

function parseBackground(value) {
  const keyword = typeof value === 'string' ? value.trim().toLowerCase() : null;
  if (value === null || keyword === 'transparent' || keyword === 'none') return null;
  if (typeof value === 'boolean') {
    throw new Error('--background requires a color, none, or transparent');
  }
  return parseColor(value);
}
