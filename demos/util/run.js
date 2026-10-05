import {
  getWebGPUHostInfo,
  Renderer,
  requestDevice,
} from "../../js/renderer.js";
import {
  blurredTargetProvider,
  imageToPng,
  l2Quality,
  loadImageSource,
  padImage,
} from "./image.js";
import { optimize } from "../../js/optimize.js";
import { resolveKernel } from "../../js/filter-kernels.js";
import { TONEMAP_HAS_WHITE, tonemapImage } from "../../js/tonemap.js";
import { withBackgroundColor } from "../../js/background-model.js";
import {
  arg,
  argList,
  booleanArg,
  formatLoss,
  integerArg,
  mkdir,
  nonNegativeIntegerArg,
  numberArg,
  parseArgs,
  positiveArg,
  slug,
  writeBytes,
  writeText,
} from "./runtime.js";

// --pad is a fraction of the longest side; clamp it so the canvas can neither
// shrink nor blow up (at 0.5 the optimization canvas roughly doubles).
const MAX_PAD = 0.5;
import { annealedBlur, shapeCli } from "./model.js";
import { lineCli } from "../lines/model.js";
import { plotCli } from "../plot/model.js";
import { LOSSES } from "./losses.js";
import { optSizeLabel, parseOptSizes, resolveOptSize } from "./opt-size.js";
import { runPath, writeCase } from "./output.js";

// A model descriptor owns everything that varies with the geometry: its
// defaults, flags, initialization, background, target preprocessing, SVG, and
// output fields. The runner below dispatches to these and to the loss
// descriptors, so it never branches on the model or loss kind.
const MODELS = { shape: shapeCli, line: lineCli, plot: plotCli };

export async function runCli(
  lossKind,
  argv,
  { modelKind = "shape", invocation = null } = {},
) {
  const options = parseArgs(argv);
  const cases = assignCaseIds(
    await resolveCases(lossKind, expandCases(lossKind, options, modelKind)),
  );
  const processStart = performance.now();
  const device = await requestDevice();
  const host = getWebGPUHostInfo();
  const environment = host.environment;
  const root = runPath(environment, lossKind, arg(options, "out", null));
  await mkdir(root);
  const results = [];

  try {
    for (let i = 0; i < cases.length; i++) {
      const config = cases[i];
      const label = caseLabel(config);
      const path = cases.length === 1 ? root : `${root}/cells/${label}`;
      const run = await runCase({
        ...config,
        lossKind,
        device,
        host,
        path,
        options,
        invocation: invocation ?? argv,
      });
      results.push(run.result);
    }
    await writeText(
      `${root}/manifest.json`,
      JSON.stringify(
        {
          loss: lossKind,
          environment,
          cases: results.map((result) => ({
            runId: result.runId,
            path: cases.length === 1 ? "." : `cells/${result.runId}`,
          })),
        },
        null,
        2,
      ) + "\n",
    );
  } finally {
    device.destroy();
  }
  return { root, results, processMs: performance.now() - processStart };
}

export async function runCase({
  lossKind,
  device,
  host,
  path,
  options,
  invocation = null,
  ...config
}) {
  const setupStart = performance.now();
  const width = config.width ?? config.size;
  const height = config.height ?? config.size;
  config = { ...config, width, height };
  const modelCli = MODELS[config.modelKind];
  const lossCli = LOSSES[lossKind];
  const visibleTarget = lossCli.loadsTarget
    ? await lossCli.loadTarget({ config, width, height })
    : null;
  if (visibleTarget) modelCli.prepareTarget(visibleTarget, config);
  const background = modelCli.background({ options, target: visibleTarget });
  if (config.initPath && !modelCli.supportsInit)
    throw new Error("--init is reserved for the shared loop benchmark model");
  // --pad expands the optimization canvas by a fraction of the longest side,
  // filling the border with edge-clamped target so marks at the visible edge
  // are fit like interior ones. The raster is cropped back to the visible frame
  // and the SVG clips to it; a pad of 0 leaves everything unchanged.
  const pad = Math.round((config.pad ?? 0) * Math.max(width, height));
  const optWidth = width + 2 * pad;
  const optHeight = height + 2 * pad;
  const target =
    visibleTarget && pad > 0
      ? { ...visibleTarget, rgba: padImage(visibleTarget.rgba, width, height, pad) }
      : visibleTarget;
  let { model, lrs, built } = await modelCli.build({
    config,
    width: optWidth,
    height: optHeight,
    pad,
    target,
    background,
  });
  // --optimize-bg trains the background color too, by prepending a full-canvas
  // opaque shape whose color gradient is exactly dL/d(bg). Off by default; the
  // background otherwise stays the fixed --bg / model default.
  if (config.optimizeBg) {
    ({ model, lrs } = withBackgroundColor(model, lrs, {
      width: optWidth,
      height: optHeight,
      background,
      lr: config.bgLr,
    }));
  }
  if (config.lrScale !== 1) {
    lrs = Object.fromEntries(Object.entries(lrs).map(([group, entry]) =>
      [group, { ...entry, lr: entry.lr * config.lrScale }]));
  }
  // The color codec declares what the scene's colors may contain; the
  // renderer validates that range against the blend mode and tonemap domain
  // at construction. The --optimize-bg wrapper does not re-expose the inner
  // style, so reach through it. A codec with a fixed alpha (raw style's
  // default) freezes the renderer's alpha-gradient pipeline entirely.
  const style = model.style ?? model.inner?.style ?? null;
  const styleRange = style?.range ?? 'unit';
  const trainsAlpha = style?.trainsAlpha !== false;
  const renderer = await Renderer.create(device, {
    width: optWidth,
    height: optHeight,
    maxShapes: model.maxShapes,
    maxPieces: model.maxPieces,
    maxCurves: model.maxCurves ?? config.n * config.k,
    blend: config.blend,
    tonemap: config.tonemap,
    colorRange: styleRange,
    // Only the optimizing renderer trains; blur widens its gradient stride
    // and a codec-fixed alpha drops that gradient pipeline.
    train: { blur: !!config.learnBlur, alpha: trainsAlpha },
  });
  // Auxiliary renderers composite linear light too; their readbacks map to
  // display space on the host with the learned exposure and white point, so
  // every output matches what the loss saw.
  const toDisplay = (image, k, w) =>
    config.tonemap === "none" ? image : tonemapImage(config.tonemap, image, k, w);
  let client = null;
  let frameRenderer = null;
  try {
    const scheduleSteps = integerArg(
      options,
      "schedule-steps",
      config.budget.mode === "steps"
        ? config.budget.value
        : modelCli.defaults.steps,
    );
    const blurAt = (step) =>
      annealedBlur(step, scheduleSteps, config.blurStart, config.blur);
    const settings = (step) => {
      const s = blurAt(step);
      return { s: [s, s], bg: background, kernel: config.kernel };
    };
    // Band-limited tone matching (plot): the render is compared against a target
    // box-filtered like the renderer's current filter, so mark density can
    // express gray levels instead of chasing crisp pixels.
    let stepTarget = target?.rgba ?? null;
    if (modelCli.bandLimited && target) {
      const blurred = blurredTargetProvider(
        target.rgba, optWidth, optHeight, 0.5, config.kernel);
      stepTarget = (step) => blurred(blurAt(step));
    }
    if (config.saveEvery) {
      await mkdir(`${path}/frames`);
      frameRenderer = await Renderer.create(device, {
        width: config.saveWidth,
        height: config.saveHeight,
        maxShapes: model.maxShapes,
        maxPieces: model.maxPieces,
        maxCurves: model.maxCurves ?? config.n * config.k,
        blend: config.blend,
      });
    }

    let loss = null;
    const lossSetup = await lossCli.setup({ config, width, height, background });
    if (lossSetup) ({ loss, client } = lossSetup);
    const setupMs = performance.now() - setupStart;

    const warmupStart = performance.now();
    if (config.warmup) {
      await warmupRenderer({
        renderer,
        model,
        settings,
        target: target?.rgba ?? null,
      });
    }
    const warmupMs = performance.now() - warmupStart;
    const logEvery = Math.max(1, Math.floor(scheduleSteps / 10));
    const onStep =
      config.quiet && !config.saveEvery
        ? null
        : async ({ step, loss: value }) => {
            if (!config.quiet && (step === 1 || step % logEvery === 0)) {
              console.log(
                `${caseLabel(config)} step ${step}: ${formatLoss(value)}`,
              );
            }
            if (frameRenderer && step % config.saveEvery === 0) {
              const { scene } = model.decode();
              frameRenderer.uploadScene(
                scene,
                saveFrameSettings({
                  width,
                  height,
                  saveWidth: config.saveWidth,
                  saveHeight: config.saveHeight,
                  blur: config.saveBlur,
                  current: settings(step - 1),
                  background,
                  pad,
                }),
              );
              // renderer.exposure/white track the session's current learned scalars.
              const image = toDisplay(
                await frameRenderer.forward(), renderer.exposure, renderer.white);
              const name = `step-${String(step).padStart(8, "0")}.png`;
              await writeBytes(
                `${path}/frames/${name}`,
                imageToPng(image, config.saveWidth, config.saveHeight),
              );
            }
          };
    const run = await optimize({
      renderer,
      model,
      lrs,
      settings,
      steps:
        config.budget.mode === "steps" ? config.budget.value : config.maxSteps,
      seconds: config.budget.mode === "seconds" ? config.budget.value : null,
      target: stepTarget,
      loss,
      views: (step) => [{ ...settings(step), weight: 1 }],
      onStep,
      exposure:
        config.tonemap === "none"
          ? null
          : {
              k: config.exposure,
              lr: config.exposureLr,
              w: config.white,
              wLr: config.whiteLr,
            },
    });
    const finalExposure = run.exposureK ?? config.exposure;
    const finalWhite = run.exposureW ?? config.white;
    const finalStart = performance.now();
    // A learned palette snaps each shape to one discovered color for the output.
    model.harden?.();
    // The trained background (when --optimize-bg is on), else the fixed one.
    const finalBackground = model.background ? model.background() : background;
    const decoded = model.decode();
    // Crop the padded scene back to the visible frame (origin at the pad); with
    // no pad this is the optimization renderer at full size.
    const cropRenderer =
      pad > 0
        ? await Renderer.create(device, {
            width,
            height,
            maxShapes: model.maxShapes,
            maxPieces: model.maxPieces,
            maxCurves: model.maxCurves ?? config.n * config.k,
            blend: config.blend,
          })
        : renderer;
    cropRenderer.uploadScene(decoded.scene, {
      s: [1, 1],
      scale: 1,
      origin: [pad, pad],
      bg: finalBackground,
    });
    cropRenderer.forwardNoRead();
    await device.queue.onSubmittedWorkDone();
    const finalRenderMs = performance.now() - finalStart;
    const readbackStart = performance.now();
    const finalImage = toDisplay(await cropRenderer.readImage(), finalExposure, finalWhite);
    const finalReadbackMs = performance.now() - readbackStart;
    if (cropRenderer !== renderer) cropRenderer.destroy();
    const quality = visibleTarget
      ? l2Quality(finalImage, visibleTarget.rgba)
      : {};
    const result = makeResult({
      config,
      lossKind,
      primitive: modelCli.primitive(built),
      host,
      setupMs,
      warmupMs,
      finalRenderMs,
      finalReadbackMs,
      run,
      quality,
      culling: renderer.getCullingInfo(),
    });
    const outputConfig = {
      // The invocation as typed (command + user flags; no interpreter or
      // script paths), so any run's config.json shows how to reproduce it.
      args: invocation ?? undefined,
      loss: lossKind,
      ...lossCli.outputFields(config),
      n: config.n,
      k: config.k,
      model: config.modelKind,
      tonemap: config.tonemap === "none" ? undefined : config.tonemap,
      exposure: config.tonemap === "none" ? undefined : finalExposure,
      white: TONEMAP_HAS_WHITE[config.tonemap] ? finalWhite : undefined,
      // The raw codec's knobs, recorded for reproducibility (absent for the
      // anchor/palette styles).
      style: config.raw ? "raw" : undefined,
      ...(config.raw
        ? {
            channels: config.raw.channels,
            transfer: config.raw.transfer,
            alpha: config.raw.alpha,
          }
        : {}),
      ...modelCli.outputFields(built),
      optSize: [width, height],
      pad: config.pad || undefined,
      seed: config.seed,
      blend: config.blend === "src-over" ? undefined : config.blend,
      kernel: config.kernel === "box" ? undefined : config.kernel,
      blur: config.blur,
      blurStart: config.blurStart,
      // --learn-blur trains a per-shape filter size instead of following the
      // global anneal; it changes the model, so record it for reproducibility
      // (omitted when off, like the other falsy-defaulted flags here).
      learnBlur: config.learnBlur || undefined,
      background: finalBackground,
      optimizeBg: config.optimizeBg || undefined,
      budget: config.budget,
      lrScale: config.lrScale === 1 ? undefined : config.lrScale,
      scheduleSteps,
      maxSteps: config.maxSteps,
      warmup: config.warmup,
      save: config.saveEvery
        ? {
            every: config.saveEvery,
            size: [config.saveWidth, config.saveHeight],
            blur: config.saveBlur,
          }
        : null,
      init: config.initPath ?? undefined,
      saveSize: config.exportWidth
        ? [config.exportWidth, config.exportHeight]
        : undefined,
    };
    // --optimize-bg prepends a full-canvas background shape so the bg color can
    // train as an ordinary shape color. It is redundant in vector output, where
    // the background <rect> already carries the (optimized) bg color, so drop it
    // from the exported shapes; the raster render keeps it via decoded.scene.
    const svgShapes = decoded.shapes.slice(model.prependedShapes ?? 0);
    await writeCase(path, {
      finalImage,
      targetImage: visibleTarget?.rgba ?? null,
      width,
      height,
      shapes: svgShapes,
      background: finalBackground,
      result,
      config: outputConfig,
      timeline: run.timeline,
      svg: modelCli.toSVG(built, {
        shapes: svgShapes,
        width,
        height,
        pad,
        background: finalBackground,
      }),
    });
    // High-resolution hero raster: the hardened final scene re-rendered by the
    // same Windfoil shader at --save-size (longest side), aspect preserved and
    // the padded border cropped exactly like final.png. Written after the core
    // artifacts and guarded so a capacity/OOM failure here never loses the run.
    if (config.exportWidth && config.exportHeight) {
      try {
        const ew = config.exportWidth;
        const eh = config.exportHeight;
        const f = Math.max(ew, eh) / Math.max(width, height);
        const exportRenderer = await Renderer.create(device, {
          width: ew,
          height: eh,
          maxShapes: model.maxShapes,
          maxPieces: model.maxPieces,
          maxCurves: model.maxCurves ?? config.n * config.k,
          blend: config.blend,
        });
        try {
          // The filter width `s` is in scene (world) units, so a crisp ~1px of
          // antialiasing at the export resolution needs s = scale (= 1/f); a
          // constant s would blur by f× and soften the whole raster.
          exportRenderer.uploadScene(decoded.scene, {
            s: [1 / f, 1 / f],
            scale: 1 / f,
            origin: [pad, pad],
            bg: finalBackground,
          });
          exportRenderer.forwardNoRead();
          await device.queue.onSubmittedWorkDone();
          const exportImage = toDisplay(
            await exportRenderer.readImage(), finalExposure, finalWhite);
          await writeBytes(
            `${path}/final-${Math.max(ew, eh)}px.png`,
            imageToPng(exportImage, ew, eh),
          );
        } finally {
          exportRenderer.destroy();
        }
      } catch (error) {
        console.warn(
          `save-size ${config.exportWidth}x${config.exportHeight} failed for ${path}: ${error.message}`,
        );
      }
    }
    if (!config.quiet) {
      console.log(
        `saved ${path} (${run.steps} steps, ${run.msPerStep?.toFixed(2)} ms/step)`,
      );
    }
    return { result, finalImage, target: target?.rgba ?? null };
  } finally {
    try {
      client?.close();
    } finally {
      try {
        frameRenderer?.destroy();
      } finally {
        renderer.destroy();
      }
    }
  }
}

// Box-filter width in pixels: --blur for the whole run (default 1, crisp), or
// a wider --blur-start that eases down to it (see annealedBlur). With
// --learn-blur, each shape starts at --blur-start and never goes below --blur.
export function parseBlurOptions(options) {
  const blur = "blur" in options ? positiveArg(options, "blur") : 1;
  const blurStart = "blur-start" in options ? positiveArg(options, "blur-start") : blur;
  if (blurStart < blur) throw new Error("--blur-start must be at least --blur");
  return { blur, blurStart };
}

export function saveFrameSettings({
  width,
  height,
  saveWidth,
  saveHeight,
  blur,
  current,
  background,
  pad = 0,
}) {
  const scale = Math.max(width / saveWidth, height / saveHeight);
  return {
    scale,
    origin: [
      pad + (width - saveWidth * scale) * 0.5,
      pad + (height - saveHeight * scale) * 0.5,
    ],
    s: blur ? current.s : [scale, scale],
    kernel: blur ? (current.kernel ?? "box") : "box",
    bg: background,
  };
}

function expandCases(lossKind, options, modelKind) {
  const modelCli = MODELS[modelKind];
  const lossCli = LOSSES[lossKind];
  const defaults = modelCli.defaults;
  const subjectKey = lossCli.subjectKey;
  const subjects = nonEmpty(
    argList(options, subjectKey, [lossCli.subjectDefault], {
      comma: lossCli.splitSubjects,
    }),
    subjectKey,
  ).map(String);
  const counts = integers(
    nonEmpty(argList(options, "n", [defaults.n]), "n"),
    "n",
  );
  const sizes = parseOptSizes(
    nonEmpty(
      argList(options, "opt-size", [lossCli.defaultSize ?? defaults.size]),
      "opt-size",
    ),
  );
  const explicitSteps = "steps" in options;
  const explicitSeconds = "seconds" in options;
  const stepBudgets = explicitSteps
    ? integers(nonEmpty(argList(options, "steps"), "steps"), "steps")
    : [];
  const secondBudgets = explicitSeconds
    ? numbers(nonEmpty(argList(options, "seconds"), "seconds"), "seconds")
    : [];
  const budgets = [
    ...stepBudgets.map((value) => ({ mode: "steps", value })),
    ...secondBudgets.map((value) => ({ mode: "seconds", value })),
  ];
  if (!budgets.length) budgets.push({ mode: "steps", value: defaults.steps });
  const save = parseSaveOptions(options);
  const common = {
    modelKind,
    // The model and loss descriptors own their own flags.
    ...modelCli.parse(options),
    ...lossCli.parse(options),
    seed: nonNegativeIntegerArg(options, "seed", 7),
    // --blend selects the scene-wide compositing mode (src-over, add,
    // multiply, screen); the Renderer validates it. Every renderer in the run
    // shares it, so saved frames and exports composite exactly like the
    // optimizer.
    blend: arg(options, "blend", "src-over"),
    // --tonemap fits and displays in tonemapped space while the scene
    // composites unbounded linear light (meant for --blend=add): reinhard,
    // reinhard-white (learnable white point, display reaches 1), or smooth
    // (signed-safe, the operator for --transfer=identity scenes).
    // --exposure seeds the learnable k; --exposure-lr sets its step size;
    // --white / --white-lr do the same for the white-point operators.
    tonemap: arg(options, "tonemap", "none"),
    exposure: "exposure" in options ? positiveArg(options, "exposure") : 1,
    exposureLr: "exposure-lr" in options
      ? positiveArg(options, "exposure-lr")
      : 0.05,
    white: "white" in options ? positiveArg(options, "white") : 1,
    whiteLr: "white-lr" in options ? positiveArg(options, "white-lr") : 0.05,
    ...parseBlurOptions(options),
    // --kernel: box, tent, or cubic; final outputs stay on box.
    kernel: kernelArg(options, "kernel", "box"),
    // --lr-scale multiplies every parameter group's learning rate. The
    // models' defaults are tuned against src-over; order-independent blends
    // deliver unattenuated gradients whose effective step grows with overlap
    // density, so dense add/screen scenes want 0.25-0.5.
    lrScale: "lr-scale" in options ? positiveArg(options, "lr-scale") : 1,
    // Fraction of the longest side to expand the optimization canvas by,
    // clamped to [0, MAX_PAD].
    pad:
      "pad" in options
        ? Math.min(MAX_PAD, Math.max(0, numberArg(options, "pad")))
        : 0,
    maxSteps: integerArg(options, "max-steps", 1_000_000),
    warmup: nonNegativeIntegerArg(options, "warmup", 1) > 0,
    // --save-size=N|max is the one output-resolution knob: it sets the longest
    // side of every raster the run writes except the canonical final.png, which
    // always stays at the optimization size because that is what the benchmark
    // scores against. So it governs the progress frames and an extra
    // `final-<N>px.png` of the hardened scene, both drawn by the Windfoil shader
    // itself (never an SVG re-draw). Independent of the optimization size, so a
    // 128px fit exports a crisp 2048px hero, and it never downsamples below the
    // optimization size.
    saveSize:
      "save-size" in options
        ? parseOptSizes([arg(options, "save-size", null)])[0]
        : null,
    // Optimize the background color alongside the scene (off by default).
    optimizeBg: "optimize-bg" in options,
    bgLr: "bg-lr" in options ? positiveArg(options, "bg-lr") : 0.01,
    quiet: "quiet" in options || "benchmark" in options,
    initPath: arg(options, "init", null),
    ...save,
  };
  const out = [];
  for (const subject of subjects) {
    for (const n of counts) {
      for (const optSize of sizes) {
        for (const budget of budgets)
          out.push({ subject, n, optSize, budget, ...common });
      }
    }
  }
  return out;
}

async function resolveCases(lossKind, cases) {
  if (!LOSSES[lossKind].loadsTarget) {
    return cases.map((config) => resolveCase(config, lossKind));
  }
  const sources = new Map();
  const resolved = [];
  for (const config of cases) {
    let source = sources.get(config.subject);
    if (!source) {
      source = await loadImageSource(config.subject).catch((error) => {
        throw describeMissingSubject(error, config.subject, LOSSES[lossKind]);
      });
      sources.set(config.subject, source);
    }
    resolved.push(resolveCase(config, lossKind, source));
  }
  return resolved;
}

// A loss's default subject may be a fetched fixture rather than a tracked file;
// on a fresh clone, say how to get it instead of reporting a bare ENOENT.
function describeMissingSubject(error, subject, lossCli) {
  const missing = error?.code === "ENOENT" || error?.name === "NotFound";
  if (!missing || subject !== lossCli.subjectDefault || !lossCli.subjectDefaultHint) {
    return error;
  }
  return new Error(`${subject} is missing: ${lossCli.subjectDefaultHint}`, {
    cause: error,
  });
}

function resolveCase(config, lossKind, source = null) {
  const size = resolveOptSize(config.optSize, lossKind, source);
  // One resolution for every extra raster: frames and final-<N>px.png alike.
  const exportSize = config.saveSize == null
    ? null
    : resolveSaveSize(config.saveSize, lossKind, source, size);
  const frameSize = config.saveEvery ? exportSize ?? size : null;
  return {
    ...config,
    ...size,
    exportWidth: exportSize?.width ?? null,
    exportHeight: exportSize?.height ?? null,
    saveWidth: frameSize?.width ?? null,
    saveHeight: frameSize?.height ?? null,
    targetSource: source ?? undefined,
  };
}

export function resolveSaveSize(value, lossKind, source, optimizationSize) {
  if (value == null) return optimizationSize;
  const requested = resolveOptSize(value, lossKind, source);
  const requestedLongest = Math.max(requested.width, requested.height);
  const optimizationLongest = Math.max(
    optimizationSize.width,
    optimizationSize.height,
  );
  return requestedLongest >= optimizationLongest
    ? requested
    : resolveOptSize(optimizationLongest, lossKind, source);
}

export function parseSaveOptions(options) {
  if (!("save-every" in options)) {
    if ("save-blur" in options) throw new Error("--save-blur requires --save-every");
    return { saveEvery: null, saveBlur: false };
  }
  return {
    saveEvery: integerArg(options, "save-every", null),
    saveBlur: booleanArg(options, "save-blur", true),
  };
}

export function caseLabel(config) {
  if (config.runId) return config.runId;
  const budget =
    config.budget.mode === "steps"
      ? `s${config.budget.value}`
      : `t${config.budget.value}`;
  return `${slug(config.subject)}-n${config.n}-o${optSizeLabel(config)}-${budget}`;
}

export function assignCaseIds(cases) {
  const seen = new Map();
  return cases.map((config) => {
    const base = caseLabel(config);
    const count = (seen.get(base) ?? 0) + 1;
    seen.set(base, count);
    return { ...config, runId: count === 1 ? base : `${base}-c${count}` };
  });
}

async function warmupRenderer({ renderer, model, settings, target }) {
  const { scene } = model.decode();
  renderer.uploadScene(scene, settings(0));
  if (target) {
    renderer.uploadTarget(target);
    const result = await renderer.stepGpuLoss();
    model.pullback(result);
  } else {
    await renderer.forward();
    await renderer.backward(new Float32Array(renderer.pixels * 4));
  }
}

function makeResult({
  config,
  lossKind,
  primitive,
  host,
  setupMs,
  warmupMs,
  finalRenderMs,
  finalReadbackMs,
  run,
  quality,
  culling,
}) {
  const losses = run.losses;
  const best = losses.reduce(
    (value, current) => Math.min(value, current),
    Infinity,
  );
  const times = run.timeline.map(
    (item, i, all) => item.elapsedMs - (i ? all[i - 1].elapsedMs : 0),
  );
  const sorted = [...times].sort((a, b) => a - b);
  const deadlineMs =
    config.budget.mode === "seconds" ? config.budget.value * 1000 : Infinity;
  const stopReason =
    config.budget.mode === "steps"
      ? "steps"
      : run.optimizeMs >= deadlineMs
        ? "seconds"
        : "maxSteps";
  return {
    schemaVersion: 1,
    runId: caseLabel(config),
    status: "ok",
    engine: {
      name: "windfoil",
      environment: host.environment,
      backend: host.backend,
    },
    workload: {
      loss: lossKind,
      primitive,
      n: config.n,
      k: config.k,
      optSize: [config.width, config.height],
      seed: config.seed,
      subject: config.subject,
    },
    budget: config.budget,
    progress: {
      stepsCompleted: run.steps,
      stopReason,
      deadlineOvershootMs:
        config.budget.mode === "seconds"
          ? Math.max(0, run.optimizeMs - config.budget.value * 1000)
          : 0,
    },
    timing: {
      setupMs,
      warmupMs,
      optimizeMs: run.optimizeMs,
      optimizeStartEpochMs: run.optimizeStartEpochMs,
      callbackMs: run.callbackMs,
      finalRenderMs,
      finalReadbackMs,
      stepMs: {
        mean: run.msPerStep,
        median: percentile(sorted, 0.5),
        p95: percentile(sorted, 0.95),
      },
    },
    loss: {
      start: losses[0] ?? null,
      end: losses.at(-1) ?? null,
      min: Number.isFinite(best) ? best : null,
      bestStep: Number.isFinite(best) ? losses.indexOf(best) + 1 : null,
    },
    quality,
    system: {
      adapter: host.adapterInfo ?? null,
      limits: host.limits ?? null,
      culling,
    },
    artifacts: {
      finalPng: "final.png",
      finalSvg: "final.svg",
      trace: "trace.jsonl",
      frames: config.saveEvery ? "frames/" : null,
    },
  };
}

function kernelArg(options, key, fallback) {
  const value = String(arg(options, key, fallback));
  resolveKernel(value); // throws with the accepted kernel names
  return value;
}

function numbers(values, key) {
  return values.map((value) => {
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0)
      throw new Error(`--${key} values must be positive numbers`);
    return number;
  });
}

function integers(values, key) {
  const out = numbers(values, key);
  if (out.some((value) => !Number.isInteger(value))) {
    throw new Error(`--${key} values must be positive integers`);
  }
  return out;
}

function nonEmpty(values, key) {
  if (!values.length) throw new Error(`--${key} must not be empty`);
  return values;
}

export function percentile(sorted, p) {
  if (!sorted.length) return null;
  const position = (sorted.length - 1) * Math.max(0, Math.min(1, p));
  const lo = Math.floor(position);
  const hi = Math.ceil(position);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (position - lo);
}
