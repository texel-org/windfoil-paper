import { imageToPng } from './image.js';
import { mkdir, stamp, writeBytes, writeText } from './runtime.js';
import { sceneToSVG } from './svg.js';
import { rgbToHex } from './color.js';

export function runPath(environment, loss, override = null) {
  return override ?? `output/${stamp()}-${environment}-${loss}`;
}

export async function writeCase(path, {
  finalImage,
  targetImage = null,
  width,
  height,
  shapes,
  background,
  result,
  config,
  timeline,
  svg = null,
}) {
  await mkdir(path);
  await Promise.all([
    writeBytes(`${path}/final.png`, imageToPng(finalImage, width, height)),
    writeText(`${path}/final.svg`, svg ?? sceneToSVG(shapes, width, height, {
      background: rgbToHex(background),
    })),
    writeText(`${path}/result.json`, JSON.stringify(result, null, 2) + '\n'),
    writeText(`${path}/config.json`, JSON.stringify(config, null, 2) + '\n'),
    writeText(`${path}/trace.jsonl`, timeline.map((item) => JSON.stringify(item)).join('\n') + '\n'),
    targetImage
      ? writeBytes(`${path}/target.png`, imageToPng(targetImage, width, height))
      : Promise.resolve(),
  ]);
}
