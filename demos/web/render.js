import { Renderer } from '../../js/renderer.js';
import { imageToPng } from '../util/image.js';
import { sceneToSVG } from '../util/svg.js';
import { exportSize } from '../util/dom.js';

export async function createSceneRenderer(device, snapshot, width, height) {
  return Renderer.create(device, {
    width, height, blend: snapshot.blend,
    maxShapes: snapshot.maxShapes, maxPieces: snapshot.maxPieces, maxCurves: snapshot.maxCurves,
    train: { geometry: false, colour: true, alpha: false },
  });
}

export function renderSettings(snapshot, width, height) {
  const scale = Math.max(snapshot.width / width, snapshot.height / height);
  return {
    scale, s: [scale, scale], bg: snapshot.background,
    origin: [(snapshot.width - width * scale) / 2, (snapshot.height - height * scale) / 2],
  };
}

export async function pngBlob(device, snapshot) {
  const { width, height } = exportSize(snapshot.width, snapshot.height);
  const renderer = await createSceneRenderer(device, snapshot, width, height);
  try {
    renderer.uploadScene(snapshot.scene, renderSettings(snapshot, width, height));
    const image = await renderer.forward();
    return new Blob([imageToPng(image, width, height)], { type: 'image/png' });
  } finally {
    renderer.destroy();
  }
}

export function svgBlob(snapshot) {
  if (snapshot.blend !== 'src-over') throw new Error('SVG export requires normal blending');
  if (snapshot.learnBlur) throw new Error('SVG cannot represent Windfoil per-shape blur');
  const background = '#' + snapshot.background.map((v) => Math.max(0, Math.min(255, Math.round(v * 255)))
    .toString(16).padStart(2, '0')).join('');
  return new Blob([sceneToSVG(snapshot.shapes, snapshot.width, snapshot.height, { background })],
    { type: 'image/svg+xml' });
}

export function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
