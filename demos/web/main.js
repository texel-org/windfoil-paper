import { requestDevice } from '../../js/renderer.js';
import { annealSettings } from '../util/model.js';
import { buildFitModel, Engine } from '../util/engine.js';
import { $, boundedInteger, cover, integerParam } from '../util/dom.js';

const params = new URLSearchParams(location.search);
const MAX_SIZE = 2048;
const MAX_COUNT = 10_000;
const MAX_STEPS = 1_000_000;
const size = integerParam(params, 'opt-size', 256, MAX_SIZE);
let count = integerParam(params, 'n', 1000, MAX_COUNT);
let steps = integerParam(params, 'steps', 500, MAX_STEPS);
let seed = integerParam(params, 'seed', 7, 0xffff_ffff, 0);
let mode = params.get('mode') === 'line' ? 'line' : 'shape';
// No imagery is tracked, so the default target is the Färlev fixture and may
// not have been fetched yet. A glob is simply empty when the file is absent,
// where a plain asset URL would leave the build with a dangling reference.
const [defaultImage] = Object.values(import.meta.glob(
  '../../fixtures/wikimedia/farlev-dip-in-road.jpg',
  { eager: true, query: '?url', import: 'default' },
));
let background = [1, 1, 1];
let engine;
let target;
let dirty = false;
let stream = null;
let live = false;
let videoTime = -1;
let sourceGeneration = 0;

const source = $('source');
const result = $('result');
const sourceContext = source.getContext('2d');
const resultContext = result.getContext('2d');
const work = document.createElement('canvas');
const workContext = work.getContext('2d', { willReadFrequently: true });
const video = $('camera');

function resizeCanvases() {
  source.width = source.height = result.width = result.height = 512;
  work.width = work.height = size;
}

function drawSource(input, width, height, mirror = false) {
  const box = cover(width, height, source.width);
  sourceContext.save();
  sourceContext.fillStyle = '#fff';
  sourceContext.fillRect(0, 0, source.width, source.height);
  if (mirror) {
    sourceContext.translate(source.width, 0);
    sourceContext.scale(-1, 1);
  }
  sourceContext.drawImage(input, box.x, box.y, box.w, box.h);
  sourceContext.restore();
}

function captureTarget(input, width, height, mirror = false) {
  const box = cover(width, height, size);
  workContext.save();
  workContext.fillStyle = '#fff';
  workContext.fillRect(0, 0, size, size);
  if (mirror) {
    workContext.translate(size, 0);
    workContext.scale(-1, 1);
  }
  workContext.drawImage(input, box.x, box.y, box.w, box.h);
  workContext.restore();
  const targetBytes = workContext.getImageData(0, 0, size, size).data;
  target ??= new Float32Array(size * size * 4);
  background = [0, 0, 0];
  for (let i = 0; i < size * size; i++) {
    for (let c = 0; c < 3; c++) {
      const value = targetBytes[4 * i + c] / 255;
      target[4 * i + c] = value;
      background[c] += value;
    }
    target[4 * i + 3] = 1;
  }
  for (let c = 0; c < 3; c++) background[c] /= size * size;
}

async function loadImage(url) {
  const ticket = ++sourceGeneration;
  const image = new Image();
  image.decoding = 'async';
  image.src = url;
  await image.decode();
  if (ticket !== sourceGeneration) return;
  stopCamera();
  captureTarget(image, image.width, image.height);
  drawSource(image, image.width, image.height);
  $('source-label').textContent = `${image.width}×${image.height} → ${size}²`;
  engine.setTarget(target, background);
  await engine.reset(true);
}

function drawResult() {
  if (!engine.shapes) return;
  const scale = result.width / size;
  resultContext.fillStyle = `rgb(${background.map((v) => Math.round(v * 255)).join(',')})`;
  resultContext.fillRect(0, 0, result.width, result.height);
  for (const shape of engine.shapes) {
    const curves = shape.curves;
    resultContext.beginPath();
    resultContext.moveTo(curves[0] * scale, curves[1] * scale);
    for (let j = 0; j < curves.length / 6; j++) {
      resultContext.quadraticCurveTo(
        curves[6 * j + 2] * scale,
        curves[6 * j + 3] * scale,
        curves[6 * j + 4] * scale,
        curves[6 * j + 5] * scale,
      );
    }
    const [r, g, b] = shape.color.map((v) => Math.round(v * 255));
    resultContext.fillStyle = `rgba(${r},${g},${b},${shape.alpha})`;
    resultContext.fill();
  }
}

function frameLoop() {
  requestAnimationFrame(frameLoop);
  if (live) refreshCamera();
  if (!dirty) return;
  dirty = false;
  drawResult();
  if (live && video.videoWidth) drawSource(video, video.videoWidth, video.videoHeight, true);
  $('step').textContent = live ? 'live' : `${engine.step} / ${steps}`;
  $('loss').textContent = engine.loss == null ? '–' : engine.loss.toExponential(3);
  $('psnr').textContent = engine.loss > 0 ? `${(-10 * Math.log10(engine.loss)).toFixed(2)} dB` : '–';
  $('rate').textContent = engine.averageMs ? `${engine.averageMs.toFixed(1)} ms` : '–';
  $('progress').style.width = live ? '100%' : `${Math.min(100, 100 * engine.step / steps)}%`;
  updateControls();
}

async function startCamera() {
  if (live) {
    captureTarget(video, video.videoWidth, video.videoHeight, true);
    drawSource(video, video.videoWidth, video.videoHeight, true);
    stopCamera();
    engine.updateTarget(target, background);
    $('source-label').textContent = 'webcam · frozen';
    return;
  }
  sourceGeneration++;
  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  try {
    await video.play();
    live = true;
    engine.continuous = true;
    videoTime = -1;
    captureTarget(video, video.videoWidth, video.videoHeight, true);
    drawSource(video, video.videoWidth, video.videoHeight, true);
    $('source-label').textContent = 'webcam · live';
    engine.setTarget(target, background);
    await engine.reset(!engine.model);
    updateControls();
  } catch (error) {
    stopCamera();
    throw error;
  }
}

function refreshCamera() {
  if (!video.videoWidth || video.currentTime === videoTime) return;
  videoTime = video.currentTime;
  captureTarget(video, video.videoWidth, video.videoHeight, true);
  engine.updateTarget(target, background);
}

function stopCamera() {
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  live = false;
  engine.continuous = false;
  updateControls();
}

function updateControls() {
  $('play').textContent = engine.playing ? 'pause' : 'play';
  $('camera-button').textContent = live ? 'freeze camera' : 'webcam';
}

function showError(error) {
  $('error').textContent = error?.message ?? String(error);
  console.error(error);
}

function bindUi() {
  $('play').onclick = () => {
    engine.playing = !engine.playing;
    updateControls();
  };
  $('reset').onclick = () => engine.reset(true).catch(showError);
  $('camera-button').onclick = () => startCamera().catch(showError);
  $('count').onchange = () => {
    count = boundedInteger($('count').value, count, 1, MAX_COUNT);
    $('count').value = count;
    engine.reset(true).catch(showError);
  };
  $('steps-input').onchange = () => {
    steps = boundedInteger($('steps-input').value, steps, 1, MAX_STEPS);
    $('steps-input').value = steps;
    engine.steps = steps;
    engine.reset(false).catch(showError);
  };
  $('mode').onchange = () => {
    mode = $('mode').value;
    $('model-label').textContent = mode === 'line' ? 'round capsules' : 'quadratic loops';
    engine.reset(true).catch(showError);
  };
  $('file').onchange = ({ target: input }) => {
    const file = input.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    loadImage(url).catch(showError).finally(() => URL.revokeObjectURL(url));
    input.value = '';
  };
  const drop = $('drop');
  drop.onclick = () => $('file').click();
  window.ondragover = (event) => event.preventDefault();
  window.ondrop = (event) => {
    event.preventDefault();
    const file = [...event.dataTransfer.files].find((item) => item.type.startsWith('image/'));
    if (!file) return;
    const url = URL.createObjectURL(file);
    loadImage(url).catch(showError).finally(() => URL.revokeObjectURL(url));
  };
}

async function main() {
  resizeCanvases();
  $('count').value = count;
  $('steps-input').value = steps;
  $('mode').value = mode;
  $('model-label').textContent = mode === 'line' ? 'round capsules' : 'quadratic loops';
  bindUi();
  const device = await requestDevice();
  engine = new Engine(device, {
    size,
    build: () => buildFitModel({ mode, n: count, size, seed: seed++, target: engine.target, background: engine.background }),
    settings: (step) => annealSettings(step, steps, engine.background),
    onFrame: () => { dirty = true; },
    onError: showError,
  });
  engine.continuous = false;
  engine.steps = steps;
  if (defaultImage) await loadImage(defaultImage);
  else $('source-label').textContent = 'drop an image, or run `npm run fixtures:farlev`';
  frameLoop();
  engine.run();
}

if (!navigator.gpu) showError('WebGPU is unavailable. Use current Chrome on localhost or HTTPS.');
else main().catch(showError);
