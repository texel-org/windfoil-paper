import { requestDevice } from "../../js/renderer.js";
import { buildFitModel, Engine } from "../util/engine.js";

import { Preview } from "./preview.js";
import { download, pngBlob, svgBlob } from "./render.js";

const sourceCanvas = document.querySelector("#source");
const resultCanvas = document.querySelector("#result");
const video = document.querySelector("#camera");
const fileInput = document.querySelector("#file");
const dropZone = document.querySelector("#drop");

const modeSelect = document.querySelector("#mode");
const countInput = document.querySelector("#count");
const optSizeSelect = document.querySelector("#opt-size");
const colorCountSelect = document.querySelector("#color-count");
const blendSelect = document.querySelector("#blend");
const backgroundSelect = document.querySelector("#background");
const alphaCheckbox = document.querySelector("#optimise-alpha");
const learnBlurCheckbox = document.querySelector("#learn-blur");

const playButton = document.querySelector("#play");
const resetButton = document.querySelector("#reset");
const cameraButton = document.querySelector("#camera-button");
const closeCameraButton = document.querySelector("#close-camera");
const pngButton = document.querySelector("#download-png");
const svgButton = document.querySelector("#download-svg");

const sourceLabel = document.querySelector("#source-label");
const modelLabel = document.querySelector("#model-label");
const renderSizeLabel = document.querySelector("#render-size");
const stepText = document.querySelector("#step");
const psnrText = document.querySelector("#psnr");
const lossText = document.querySelector("#loss");
const rateText = document.querySelector("#rate");
const errorText = document.querySelector("#error");

const MAX_COUNT = 100_000;
// Under "auto", each blend gets its natural backdrop.
const AUTO_BACKGROUND = {
  "src-over": "mean",
  multiply: "white",
  add: "black",
  screen: "black",
};
// A 4096 px copy of the Färlev fixture, made by tools/web-image.js. The fixture
// is optional; builds also work before it is downloaded.
const [defaultImage] = Object.values(
  import.meta.glob("../../fixtures/wikimedia/farlev-dip-in-road-4096.jpg", {
    eager: true,
    query: "?url",
    import: "default",
  }),
);

const params = new URLSearchParams(location.search);
let seed = integerParam(params.get("seed"), 0, 0xffff_ffff, 7);
const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
let target;
let targetMean = [1, 1, 1];
let engine;
let preview;
let dirty = false;
let rendering = false;
let downloading = false;
let lastPreview = -Infinity;
// The still input (an image or a frozen camera frame), kept so a new opt size
// can resample it.
let inputImage = null;
let live = false;
let cameraSource = false;
let videoTime = -1;
let sourceGeneration = 0;

// An integer up to `max`, or the fallback when `value` is not one >= `min`.
function integerParam(value, min, max, fallback) {
  const number = Number.parseInt(value, 10);
  return number >= min ? Math.min(number, max) : fallback;
}

// The controls hold the fit options; the URL can preset them, e.g.
// ?mode=line&n=1000&blend=add&bg=black&color-count=4&opacity=opaque.
function presetControls() {
  const selects = {
    mode: modeSelect,
    "opt-size": optSizeSelect,
    "color-count": colorCountSelect,
    blend: blendSelect,
    bg: backgroundSelect,
  };
  for (const [name, select] of Object.entries(selects)) {
    const value = params.get(name);
    if ([...select.options].some((option) => option.value === value)) {
      select.value = value;
    }
  }
  countInput.value = integerParam(params.get("n"), 1, MAX_COUNT, 256);
  alphaCheckbox.checked = params.get("opacity") !== "opaque";
  learnBlurCheckbox.checked = params.get("learn-blur") === "true";
}

const shapeCount = () => integerParam(countInput.value, 1, MAX_COUNT, 256);

function backgroundColor() {
  const choice =
    backgroundSelect.value === "auto"
      ? AUTO_BACKGROUND[blendSelect.value]
      : backgroundSelect.value;
  return { mean: targetMean, white: [1, 1, 1], black: [0, 0, 0] }[choice];
}

// Size the target canvas for an input: the opt size caps the longest side, and
// smaller inputs are not enlarged.
function resizeTarget(width = 512, height = 288) {
  const optSize = Number(optSizeSelect.value);
  const scale = Math.min(1, optSize / Math.max(width, height));
  sourceCanvas.width = Math.max(1, Math.round(width * scale));
  sourceCanvas.height = Math.max(1, Math.round(height * scale));
  document.documentElement.style.setProperty(
    "--image-aspect",
    `${sourceCanvas.width} / ${sourceCanvas.height}`,
  );
  target = new Float32Array(sourceCanvas.width * sourceCanvas.height * 4);
  dirty = true;
  if (engine) {
    engine.width = sourceCanvas.width;
    engine.height = sourceCanvas.height;
  }
}

// Draw the input onto the target canvas (over white, mirrored for the camera)
// and read it back as the RGBA target.
function captureTarget(input, mirror) {
  const { width, height } = sourceCanvas;
  sourceContext.save();
  sourceContext.fillStyle = "#fff";
  sourceContext.fillRect(0, 0, width, height);
  if (mirror) {
    sourceContext.translate(width, 0);
    sourceContext.scale(-1, 1);
  }
  sourceContext.drawImage(input, 0, 0, width, height);
  sourceContext.restore();
  const bytes = sourceContext.getImageData(0, 0, width, height).data;
  for (let i = 0; i < bytes.length; i++) target[i] = bytes[i] / 255;
  const pixels = width * height;
  targetMean = [0, 0, 0];
  for (let i = 0; i < target.length; i += 4) {
    for (let c = 0; c < 3; c++) targetMean[c] += target[i + c] / pixels;
  }
}

// Fit a new input from scratch.
async function useInput(input, width, height, mirror) {
  resizeTarget(width, height);
  captureTarget(input, mirror);
  updateSourceLabel(width, height);
  engine.setTarget(target, backgroundColor());
  await engine.reset(true);
}

async function loadImage(url) {
  const ticket = ++sourceGeneration;
  const image = new Image();
  image.src = url;
  await image.decode();
  if (ticket !== sourceGeneration) return;
  stopCamera();
  cameraSource = false;
  inputImage = image;
  await useInput(image, image.width, image.height, false);
}

function loadFile(file) {
  const url = URL.createObjectURL(file);
  loadImage(url)
    .catch(showError)
    .finally(() => URL.revokeObjectURL(url));
}

async function resizeInput() {
  if (live) {
    await useInput(video, video.videoWidth, video.videoHeight, true);
  } else if (inputImage) {
    await useInput(inputImage, inputImage.width, inputImage.height, cameraSource);
  }
}

// Rebuild the model after an option changes.
function restart() {
  engine.blend = blendSelect.value;
  updateControls();
  if (!engine.target) return;
  engine.background = backgroundColor();
  errorText.textContent = "";
  engine.reset(true).catch(showError);
}

function updateSourceLabel(width, height) {
  const size = `${sourceCanvas.width} × ${sourceCanvas.height}`;
  sourceLabel.textContent = cameraSource
    ? `webcam · ${live ? "live" : "frozen"} · ${size}`
    : `${width} × ${height} → ${size}`;
}

async function startCamera() {
  sourceGeneration++;
  video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
  try {
    await video.play();
    live = true;
    cameraSource = true;
    videoTime = -1;
    await useInput(video, video.videoWidth, video.videoHeight, true);
  } catch (error) {
    stopCamera();
    throw error;
  }
}

// Keep optimizing toward the current camera frame, without a reset.
function freezeCamera() {
  // Keep the full-size frame so changing the opt size can resample it.
  inputImage = document.createElement("canvas");
  inputImage.width = video.videoWidth;
  inputImage.height = video.videoHeight;
  inputImage.getContext("2d").drawImage(video, 0, 0);
  captureTarget(video, true);
  stopCamera();
  engine.updateTarget(target, backgroundColor());
  updateSourceLabel(inputImage.width, inputImage.height);
}

async function closeCamera() {
  sourceGeneration++;
  stopCamera();
  cameraSource = false;
  updateControls();
  if (defaultImage) return loadImage(defaultImage);
  await engine.clear();
  inputImage = null;
  resizeTarget();
  preview.clear();
  sourceLabel.textContent = "choose an image";
}

function refreshCamera() {
  if (!video.videoWidth || video.currentTime === videoTime) return;
  videoTime = video.currentTime;
  captureTarget(video, true);
  engine.updateTarget(target, backgroundColor());
  dirty = true;
}

function stopCamera() {
  video.srcObject?.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
  live = false;
  updateControls();
}

function frameLoop(now = 0) {
  requestAnimationFrame(frameLoop);
  if (live) refreshCamera();
  // The preview renders at the canvas's display resolution.
  const box = resultCanvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(box.width * devicePixelRatio));
  const height = Math.max(1, Math.round(box.height * devicePixelRatio));
  if (width !== resultCanvas.width || height !== resultCanvas.height) {
    dirty = true;
  }
  if (!dirty || rendering || now - lastPreview < 1000 / 30) return;
  dirty = false;
  lastPreview = now;
  const snapshot = engine.snapshot();
  if (snapshot) {
    rendering = true;
    const generation = engine.generation;
    const isCurrent = () => generation === engine.generation && !!engine.target;
    preview
      .render(snapshot, width, height, isCurrent)
      .then((rendered) => {
        if (rendered) renderSizeLabel.textContent = `${width} × ${height}`;
      })
      .catch(showError)
      .finally(() => {
        rendering = false;
      });
  }
  const { step, loss, averageMs } = engine;
  stepText.textContent = step;
  lossText.textContent = loss == null ? "–" : loss.toExponential(3);
  psnrText.textContent = loss > 0 ? `${(-10 * Math.log10(loss)).toFixed(2)} dB` : "–";
  rateText.textContent = averageMs ? `${averageMs.toFixed(1)} ms` : "–";
  updateControls();
}

async function downloadScene(format) {
  const snapshot = engine.snapshot(format === "svg");
  if (!snapshot || downloading) return;
  downloading = true;
  updateControls();
  try {
    const blob =
      format === "png"
        ? await pngBlob(engine.device, snapshot)
        : svgBlob(snapshot);
    download(blob, `windfoil.${format}`);
  } finally {
    downloading = false;
    updateControls();
  }
}

function updateControls() {
  const ready = !!engine.model && !!engine.target;
  const normalBlend = blendSelect.value === "src-over";
  const learnBlur = learnBlurCheckbox.checked;
  modelLabel.textContent = modeSelect.selectedOptions[0].text;
  playButton.textContent = engine.playing ? "pause" : "play";
  playButton.disabled = !ready;
  resetButton.disabled = !ready;
  cameraButton.textContent = live ? "freeze camera" : "webcam";
  cameraButton.disabled = false;
  closeCameraButton.hidden = !cameraSource;
  pngButton.disabled = !ready || downloading;
  svgButton.disabled = !ready || downloading || !normalBlend || learnBlur;
  svgButton.title = learnBlur
    ? "SVG cannot represent Windfoil per-shape blur"
    : normalBlend
      ? ""
      : "SVG download requires normal blending";
}

function showError(error) {
  errorText.textContent = error?.message ?? String(error);
  console.error(error);
}

function bindUi() {
  // Scrolling over the focused count field should scroll the page, not
  // change the count.
  countInput.addEventListener(
    "wheel",
    (event) => {
      if (document.activeElement === countInput) event.preventDefault();
    },
    { passive: false },
  );
  playButton.onclick = () => {
    engine.playing = !engine.playing;
    updateControls();
  };
  resetButton.onclick = () => engine.reset(true).catch(showError);
  cameraButton.onclick = () => {
    if (live) freezeCamera();
    else startCamera().catch(showError);
  };
  closeCameraButton.onclick = () => closeCamera().catch(showError);
  pngButton.onclick = () => downloadScene("png").catch(showError);
  svgButton.onclick = () => downloadScene("svg").catch(showError);
  countInput.onchange = () => {
    countInput.value = shapeCount();
    restart();
  };
  optSizeSelect.onchange = () => resizeInput().catch(showError);
  for (const control of [
    modeSelect,
    colorCountSelect,
    blendSelect,
    backgroundSelect,
    alphaCheckbox,
    learnBlurCheckbox,
  ]) {
    control.onchange = restart;
  }
  fileInput.onchange = () => {
    const [file] = fileInput.files;
    if (file) loadFile(file);
    fileInput.value = "";
  };
  dropZone.onclick = () => fileInput.click();
  window.ondragover = (event) => event.preventDefault();
  window.ondrop = (event) => {
    event.preventDefault();
    const file = [...event.dataTransfer.files].find((item) =>
      item.type.startsWith("image/"),
    );
    if (file) loadFile(file);
  };
}

async function main() {
  presetControls();
  resizeTarget();
  const device = await requestDevice();
  preview = new Preview(device, resultCanvas);
  engine = new Engine(device, {
    width: sourceCanvas.width,
    height: sourceCanvas.height,
    blend: blendSelect.value,
    build: () =>
      buildFitModel({
        mode: modeSelect.value,
        n: shapeCount(),
        width: engine.width,
        height: engine.height,
        seed: seed++,
        target: engine.target,
        background: engine.background,
        colorCount: Number(colorCountSelect.value),
        opaque: !alphaCheckbox.checked,
        learnBlur: learnBlurCheckbox.checked,
        lrScale: blendSelect.value === "src-over" ? 1 : 0.25,
      }),
    settings: () => ({ s: [1, 1], bg: engine.background }),
    onFrame: () => {
      dirty = true;
    },
    onError: showError,
  });
  bindUi();
  updateControls();
  if (defaultImage) await loadImage(defaultImage);
  else sourceLabel.textContent = "choose an image";
  frameLoop();
  engine.run();
}

if (!navigator.gpu)
  showError("WebGPU is unavailable. Use current Chrome on localhost or HTTPS.");
else main().catch(showError);
