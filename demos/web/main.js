import { requestDevice } from "../../js/renderer.js";
import { buildFitModel, Engine } from "../util/engine.js";
import {
  $,
  boundedInteger,
  imageSize,
  integerParam,
  previewSize,
} from "../util/dom.js";

import { Preview } from "./preview.js";
import { download, pngBlob, svgBlob } from "./render.js";

const params = new URLSearchParams(location.search);
const MAX_COUNT = 100_000;
const imageSizes = ["256", "512", "1080", "2048", "4096"];
let size = imageSizes.includes(params.get("opt-size"))
  ? params.get("opt-size")
  : "512";
let count = integerParam(params, "n", 256, MAX_COUNT);
let seed = integerParam(params, "seed", 7, 0xffff_ffff, 0);
const modelLabels = { shape: "shapes", line: "lines", capsule: "capsules" };
let mode = params.get("mode") in modelLabels ? params.get("mode") : "shape";
let opaque = params.get("opacity") === "opaque";
let learnBlur = params.get("learn-blur") === "true";
const paletteCounts = [1, 2, 4, 8, 16, 32];
const requestedColors = Number(params.get("color-count") ?? 1);
let colorCount = paletteCounts.includes(requestedColors) ? requestedColors : 1;
const blends = ["src-over", "add", "multiply", "screen"];
let blend = blends.includes(params.get("blend"))
  ? params.get("blend")
  : "src-over";
let backgroundMode = ["white", "black"].includes(params.get("bg"))
  ? params.get("bg")
  : "auto";
// The fixture is optional; builds also work before it is downloaded.
const [defaultImage] = Object.values(
  import.meta.glob("../../fixtures/wikimedia/farlev-dip-in-road.jpg", {
    eager: true,
    query: "?url",
    import: "default",
  }),
);
let targetMean = [1, 1, 1];
let background = [1, 1, 1];
let engine;
let target;
let dirty = false;
let rendering = false;
let downloading = false;
let stream = null;
let live = false;
let cameraSource = false;
let videoTime = -1;
let sourceGeneration = 0;
let inputImage = null;
let lastPreview = -Infinity;

const source = $("source");
const result = $("result");
const sourceContext = source.getContext("2d");
let preview;
const work = document.createElement("canvas");
const workContext = work.getContext("2d", { willReadFrequently: true });
const video = $("camera");

function resizeCanvases(width = 512, height = 288) {
  const dimensions = imageSize(width, height, size);
  source.width = work.width = dimensions.width;
  source.height = work.height = dimensions.height;
  document.documentElement.style.setProperty(
    "--image-aspect",
    `${dimensions.width} / ${dimensions.height}`,
  );
  target = new Float32Array(work.width * work.height * 4);
  dirty = true;
  if (engine) {
    engine.width = work.width;
    engine.height = work.height;
  }
}

function drawSource(input, width, height, mirror = false) {
  sourceContext.save();
  sourceContext.fillStyle = "#fff";
  sourceContext.fillRect(0, 0, source.width, source.height);
  if (mirror) {
    sourceContext.translate(source.width, 0);
    sourceContext.scale(-1, 1);
  }
  sourceContext.drawImage(input, 0, 0, source.width, source.height);
  sourceContext.restore();
}

function captureTarget(input, width, height, mirror = false) {
  workContext.save();
  workContext.fillStyle = "#fff";
  workContext.fillRect(0, 0, work.width, work.height);
  if (mirror) {
    workContext.translate(work.width, 0);
    workContext.scale(-1, 1);
  }
  workContext.drawImage(input, 0, 0, work.width, work.height);
  workContext.restore();
  const targetBytes = workContext.getImageData(
    0,
    0,
    work.width,
    work.height,
  ).data;
  const pixels = work.width * work.height;
  targetMean = [0, 0, 0];
  for (let i = 0; i < pixels; i++) {
    for (let c = 0; c < 3; c++) {
      const value = targetBytes[4 * i + c] / 255;
      target[4 * i + c] = value;
      targetMean[c] += value;
    }
    target[4 * i + 3] = 1;
  }
  for (let c = 0; c < 3; c++) targetMean[c] /= pixels;
  updateBackground();
}

function updateBackground() {
  const chosen =
    backgroundMode === "auto"
      ? blend === "src-over"
        ? "mean"
        : blend === "multiply"
          ? "white"
          : "black"
      : backgroundMode;
  background =
    chosen === "mean" ? targetMean : chosen === "white" ? [1, 1, 1] : [0, 0, 0];
}

function resetOptions() {
  if (!engine) return;
  updateBackground();
  engine.blend = blend;
  engine.setTarget(target, background);
  $("error").textContent = "";
  engine.reset(true).catch(showError);
}

async function loadImage(url) {
  const ticket = ++sourceGeneration;
  const image = new Image();
  image.decoding = "async";
  image.src = url;
  await image.decode();
  if (ticket !== sourceGeneration) return;
  stopCamera();
  cameraSource = false;
  inputImage = image;
  resizeCanvases(image.width, image.height);
  captureTarget(image, image.width, image.height);
  drawSource(image, image.width, image.height);
  updateSourceLabel(image.width, image.height);
  engine.setTarget(target, background);
  await engine.reset(true);
}

function updateSourceLabel(width, height) {
  $("source-label").textContent = cameraSource
    ? `webcam · ${live ? "live" : "frozen"} · ${work.width} × ${work.height}`
    : `${width} × ${height} → ${work.width} × ${work.height}`;
}

async function resizeInput() {
  const input = live ? video : inputImage;
  if (!input) return;
  const width = live ? video.videoWidth : input.width;
  const height = live ? video.videoHeight : input.height;
  resizeCanvases(width, height);
  captureTarget(input, width, height, cameraSource);
  drawSource(input, width, height, cameraSource);
  updateSourceLabel(width, height);
  engine.setTarget(target, background);
  await engine.reset(true);
}

function frameLoop(now = 0) {
  requestAnimationFrame(frameLoop);
  if (live) refreshCamera();
  const box = result.getBoundingClientRect();
  const dimensions = previewSize(
    box.width,
    box.height,
    window.devicePixelRatio || 1,
  );
  if (dimensions.width !== result.width || dimensions.height !== result.height)
    dirty = true;
  if (!dirty || rendering || now - lastPreview < 1000 / 30) return;
  dirty = false;
  lastPreview = now;
  if (live && video.videoWidth)
    drawSource(video, video.videoWidth, video.videoHeight, true);
  const snapshot = engine.snapshot();
  if (snapshot) {
    rendering = true;
    const generation = engine.generation;
    preview
      .render(
        snapshot,
        dimensions.width,
        dimensions.height,
        () => generation === engine.generation && !!engine.target,
      )
      .then((rendered) => {
        if (rendered)
          $("render-size").textContent =
            `${dimensions.width} × ${dimensions.height}`;
      })
      .catch(showError)
      .finally(() => {
        rendering = false;
      });
  }
  $("step").textContent = String(engine.step);
  $("loss").textContent =
    engine.loss == null ? "–" : engine.loss.toExponential(3);
  $("psnr").textContent =
    engine.loss > 0 ? `${(-10 * Math.log10(engine.loss)).toFixed(2)} dB` : "–";
  $("rate").textContent = engine.averageMs
    ? `${engine.averageMs.toFixed(1)} ms`
    : "–";
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

async function startCamera() {
  if (live) {
    // Keep the original frozen frame so changing fit size can resample it.
    inputImage = document.createElement("canvas");
    inputImage.width = video.videoWidth;
    inputImage.height = video.videoHeight;
    inputImage.getContext("2d").drawImage(video, 0, 0);
    captureTarget(video, video.videoWidth, video.videoHeight, true);
    drawSource(video, video.videoWidth, video.videoHeight, true);
    stopCamera();
    engine.updateTarget(target, background);
    updateSourceLabel(inputImage.width, inputImage.height);
    return;
  }
  sourceGeneration++;
  stream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: false,
  });
  video.srcObject = stream;
  try {
    await video.play();
    live = true;
    cameraSource = true;
    videoTime = -1;
    resizeCanvases(video.videoWidth, video.videoHeight);
    captureTarget(video, video.videoWidth, video.videoHeight, true);
    drawSource(video, video.videoWidth, video.videoHeight, true);
    updateSourceLabel(video.videoWidth, video.videoHeight);
    engine.setTarget(target, background);
    await engine.reset(true);
    updateControls();
  } catch (error) {
    stopCamera();
    throw error;
  }
}

async function closeCamera() {
  sourceGeneration++;
  stopCamera();
  cameraSource = false;
  updateControls();
  if (defaultImage) {
    await loadImage(defaultImage);
  } else {
    engine.playing = false;
    engine.generation++;
    if (engine.activeStep) await engine.activeStep.catch(() => {});
    engine.setTarget(null);
    engine.session = null;
    engine.model = null;
    engine.step = 0;
    engine.loss = null;
    engine.averageMs = 0;
    inputImage = null;
    resizeCanvases();
    sourceContext.clearRect(0, 0, source.width, source.height);
    preview.clear();
    $("source-label").textContent = "choose an image";
    dirty = true;
  }
}

function refreshCamera() {
  if (!video.videoWidth || video.currentTime === videoTime) return;
  videoTime = video.currentTime;
  captureTarget(video, video.videoWidth, video.videoHeight, true);
  engine.updateTarget(target, background);
  dirty = true;
}

function stopCamera() {
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  live = false;
  updateControls();
}

function updateControls() {
  $("play").textContent = engine.playing ? "pause" : "play";
  $("camera-button").textContent = live ? "freeze camera" : "webcam";
  $("close-camera").hidden = !cameraSource;
  const ready = !!engine.model && !!engine.target;
  $("play").disabled = !ready;
  $("reset").disabled = !ready;
  $("camera-button").disabled = false;
  $("download-png").disabled = !ready || downloading;
  $("download-svg").disabled =
    !ready || downloading || blend !== "src-over" || learnBlur;
  $("download-svg").title = learnBlur
    ? "SVG cannot represent Windfoil per-shape blur"
    : blend === "src-over"
      ? ""
      : "SVG download requires normal blending";
}

function showError(error) {
  $("error").textContent = error?.message ?? String(error);
  console.error(error);
}

function bindUi() {
  for (const input of document.querySelectorAll('input[type="number"]')) {
    input.addEventListener(
      "wheel",
      (event) => {
        if (document.activeElement === input) event.preventDefault();
      },
      { passive: false },
    );
  }
  $("play").onclick = () => {
    engine.playing = !engine.playing;
    updateControls();
  };
  $("reset").onclick = () => engine.reset(true).catch(showError);
  $("camera-button").onclick = () => startCamera().catch(showError);
  $("close-camera").onclick = () => closeCamera().catch(showError);
  $("download-png").onclick = () => downloadScene("png").catch(showError);
  $("download-svg").onclick = () => downloadScene("svg").catch(showError);
  $("count").onchange = () => {
    count = boundedInteger($("count").value, count, 1, MAX_COUNT);
    $("count").value = count;
    resetOptions();
  };
  $("opt-size").onchange = () => {
    size = $("opt-size").value;
    resizeInput().catch(showError);
  };
  $("mode").onchange = () => {
    mode = $("mode").value;
    $("model-label").textContent = modelLabels[mode];
    resetOptions();
  };
  $("color-count").onchange = () => {
    colorCount = Number($("color-count").value);
    resetOptions();
  };
  $("blend").onchange = () => {
    blend = $("blend").value;
    resetOptions();
  };
  $("background").onchange = () => {
    backgroundMode = $("background").value;
    resetOptions();
  };
  $("optimise-alpha").onchange = () => {
    opaque = !$("optimise-alpha").checked;
    resetOptions();
  };
  $("learn-blur").onchange = () => {
    learnBlur = $("learn-blur").checked;
    resetOptions();
  };
  $("file").onchange = ({ target: input }) => {
    const file = input.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    loadImage(url)
      .catch(showError)
      .finally(() => URL.revokeObjectURL(url));
    input.value = "";
  };
  const drop = $("drop");
  drop.onclick = () => $("file").click();
  window.ondragover = (event) => event.preventDefault();
  window.ondrop = (event) => {
    event.preventDefault();
    const file = [...event.dataTransfer.files].find((item) =>
      item.type.startsWith("image/"),
    );
    if (!file) return;
    const url = URL.createObjectURL(file);
    loadImage(url)
      .catch(showError)
      .finally(() => URL.revokeObjectURL(url));
  };
}

async function main() {
  resizeCanvases();
  $("count").value = count;
  $("opt-size").value = size;
  $("mode").value = mode;
  $("color-count").value = colorCount;
  $("blend").value = blend;
  $("background").value = backgroundMode;
  $("optimise-alpha").checked = !opaque;
  $("learn-blur").checked = learnBlur;
  $("model-label").textContent = modelLabels[mode];
  bindUi();
  const device = await requestDevice();
  preview = new Preview(device, result);
  engine = new Engine(device, {
    width: work.width,
    height: work.height,
    blend,
    build: () =>
      buildFitModel({
        mode,
        n: count,
        width: engine.width,
        height: engine.height,
        seed: seed++,
        target: engine.target,
        background: engine.background,
        colorCount,
        opaque,
        learnBlur,
        lrScale: blend === "src-over" ? 1 : 0.25,
      }),
    settings: () => ({ s: [1, 1], bg: engine.background }),
    onFrame: () => {
      dirty = true;
    },
    onError: showError,
  });
  updateControls();
  if (defaultImage) await loadImage(defaultImage);
  else $("source-label").textContent = "choose an image";
  frameLoop();
  engine.run();
}

if (!navigator.gpu)
  showError("WebGPU is unavailable. Use current Chrome on localhost or HTTPS.");
else main().catch(showError);
