import { resolveKernel } from "../../js/filter-kernels.js";
import { packScene } from "../../js/prep.js";
import { getWebGPUHostInfo, requestDevice } from "../../js/renderer.js";
import { rgbaToPng } from "../util/image.js";
import { runPath } from "../util/output.js";
import { mkdir, runMain, writeBytes, writeText } from "../util/runtime.js";
import { sceneToSVG } from "../util/svg.js";
import { parseFillSvg, rasterSize } from "../render/svg.js";
import { renderChunked } from "../render/tiled.js";

export const MOIRE_DEFAULTS = Object.freeze({
  viewSize: 512,
  sizes: Object.freeze([256, 512, 1024]),
  kernels: Object.freeze(["box", "tent", "cubic"]),
  scenes: Object.freeze(["sunburst", "floor"]),
  sunburst: Object.freeze({
    spokes: 256,
    lineWidth: 3,
    radius: 235,
  }),
  floor: Object.freeze({
    horizon: 80,
    near: 5,
    rows: 96,
    cellWidth: 40,
  }),
});

const OPTION_NAMES = new Set([
  "scene",
  "size",
  "kernel",
  "spokes",
  "line-width",
  "floor-horizon",
  "floor-near",
  "floor-rows",
  "floor-cell-width",
  "chrome",
  "out",
  "quiet",
]);

const SCENE_NAMES = new Set(MOIRE_DEFAULTS.scenes);

function requiredString(value, name) {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`--${name} requires a value`);
  }
  return value;
}

function positiveInteger(value, name) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < 1) {
    throw new Error(`--${name} must be a positive integer`);
  }
  return number;
}

function positiveNumber(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) {
    throw new Error(`--${name} must be a positive number`);
  }
  return number;
}

function commaList(value, name) {
  const values = requiredString(value, name)
    .split(",")
    .map((item) => item.trim());
  if (values.some((item) => item.length === 0)) {
    throw new Error(
      `--${name} must be a comma-separated list without empty values`,
    );
  }
  if (new Set(values).size !== values.length) {
    throw new Error(`--${name} must not contain duplicate values`);
  }
  return values;
}

/** Strictly parse the renderer-only moire CLI. */
export function parseMoireArgs(argv) {
  const raw = new Map();
  for (let at = 0; at < argv.length; at++) {
    const token = argv[at];
    if (
      typeof token !== "string" ||
      !token.startsWith("--") ||
      token === "--"
    ) {
      throw new Error(`unexpected argument ${JSON.stringify(token)}`);
    }
    const equals = token.indexOf("=");
    const name = token.slice(2, equals < 0 ? undefined : equals);
    if (!OPTION_NAMES.has(name)) throw new Error(`unknown option --${name}`);
    if (raw.has(name)) throw new Error(`duplicate option --${name}`);
    if (name === "quiet" || name === "chrome") {
      if (equals >= 0) throw new Error(`--${name} does not take a value`);
      raw.set(name, true);
      continue;
    }
    const value = equals >= 0 ? token.slice(equals + 1) : argv[++at];
    if (value === undefined || (equals < 0 && value.startsWith("--"))) {
      throw new Error(`--${name} requires a value`);
    }
    raw.set(name, requiredString(value, name));
  }

  const scenes = raw.has("scene")
    ? commaList(raw.get("scene"), "scene")
    : Array.from(MOIRE_DEFAULTS.scenes);
  for (const scene of scenes) {
    if (!SCENE_NAMES.has(scene)) {
      throw new Error(
        `unknown moire scene ${JSON.stringify(scene)} (expected ${Array.from(SCENE_NAMES).join(" or ")})`,
      );
    }
  }

  const sizes = raw.has("size")
    ? commaList(raw.get("size"), "size").map((value) =>
        positiveInteger(value, "size"),
      )
    : Array.from(MOIRE_DEFAULTS.sizes);
  const kernels = raw.has("kernel")
    ? commaList(raw.get("kernel"), "kernel")
    : Array.from(MOIRE_DEFAULTS.kernels);
  for (const kernel of kernels) resolveKernel(kernel);

  return {
    scenes,
    sizes,
    kernels,
    spokes: raw.has("spokes")
      ? positiveInteger(raw.get("spokes"), "spokes")
      : MOIRE_DEFAULTS.sunburst.spokes,
    lineWidth: raw.has("line-width")
      ? positiveNumber(raw.get("line-width"), "line-width")
      : MOIRE_DEFAULTS.sunburst.lineWidth,
    floorHorizon: raw.has("floor-horizon")
      ? positiveNumber(raw.get("floor-horizon"), "floor-horizon")
      : MOIRE_DEFAULTS.floor.horizon,
    floorNear: raw.has("floor-near")
      ? positiveInteger(raw.get("floor-near"), "floor-near")
      : MOIRE_DEFAULTS.floor.near,
    floorRows: raw.has("floor-rows")
      ? positiveInteger(raw.get("floor-rows"), "floor-rows")
      : MOIRE_DEFAULTS.floor.rows,
    floorCellWidth: raw.has("floor-cell-width")
      ? positiveNumber(raw.get("floor-cell-width"), "floor-cell-width")
      : MOIRE_DEFAULTS.floor.cellWidth,
    chrome: raw.has("chrome"),
    out: raw.has("out") ? raw.get("out") : null,
    quiet: raw.has("quiet"),
  };
}

const lineCurve = (x0, y0, x1, y1) => [
  x0,
  y0,
  0.5 * (x0 + x1),
  0.5 * (y0 + y1),
  x1,
  y1,
];

function polygonCurves(points) {
  return Float64Array.from(
    points.flatMap((point, index) => {
      const next = points[(index + 1) % points.length];
      return lineCurve(point[0], point[1], next[0], next[1]);
    }),
  );
}

function filledShape(points) {
  return {
    curves: polygonCurves(points),
    color: [0, 0, 0],
    alpha: 1,
    fillRule: "nonzero",
  };
}

/** Build tapered filled spokes whose outer chord has exactly lineWidth units. */
export function buildSunburst({
  viewSize = MOIRE_DEFAULTS.viewSize,
  spokes = MOIRE_DEFAULTS.sunburst.spokes,
  lineWidth = MOIRE_DEFAULTS.sunburst.lineWidth,
  radius = MOIRE_DEFAULTS.sunburst.radius,
} = {}) {
  positiveNumber(viewSize, "view-size");
  positiveInteger(spokes, "spokes");
  positiveNumber(lineWidth, "line-width");
  positiveNumber(radius, "radius");
  if (radius >= viewSize / 2) {
    throw new Error("sunburst radius must leave a canvas margin");
  }
  if (lineWidth >= 2 * radius) {
    throw new Error("sunburst line width must be smaller than its diameter");
  }

  const halfAngle = Math.asin(lineWidth / (2 * radius));
  if (halfAngle >= Math.PI / spokes) {
    throw new Error(
      "sunburst line width must leave a gap between adjacent spokes",
    );
  }

  const center = viewSize / 2;
  const shapes = [];
  for (let index = 0; index < spokes; index++) {
    const angle = -Math.PI / 2 + (index * 2 * Math.PI) / spokes;
    const ax = center + radius * Math.cos(angle - halfAngle);
    const ay = center + radius * Math.sin(angle - halfAngle);
    const bx = center + radius * Math.cos(angle + halfAngle);
    const by = center + radius * Math.sin(angle + halfAngle);
    shapes.push(
      filledShape([
        [center, center],
        [ax, ay],
        [bx, by],
      ]),
    );
  }

  return {
    name: "sunburst",
    viewBox: { x: 0, y: 0, width: viewSize, height: viewSize },
    background: [1, 1, 1],
    parameters: {
      spokes,
      lineWidth,
      radius,
    },
    shapes,
  };
}

/** Build a perspective checkerboard floor that contracts toward the horizon. */
export function buildFloor({
  viewSize = MOIRE_DEFAULTS.viewSize,
  horizon = MOIRE_DEFAULTS.floor.horizon,
  near = MOIRE_DEFAULTS.floor.near,
  rows = MOIRE_DEFAULTS.floor.rows,
  cellWidth = MOIRE_DEFAULTS.floor.cellWidth,
} = {}) {
  positiveNumber(viewSize, "view-size");
  positiveNumber(horizon, "floor-horizon");
  positiveInteger(near, "floor-near");
  positiveInteger(rows, "floor-rows");
  positiveNumber(cellWidth, "floor-cell-width");
  if (horizon >= viewSize) {
    throw new Error("floor horizon must be inside the reference canvas");
  }

  const center = viewSize / 2;
  const verticalScale = (viewSize - horizon) * near;
  const horizontalScale = cellWidth * near;
  const projectX = (worldX, depth) =>
    center + (horizontalScale * worldX) / depth;
  const projectY = (depth) => horizon + verticalScale / depth;
  const worldX = (screenX, depth) =>
    ((screenX - center) * depth) / horizontalScale;
  const shapes = [];
  let checks = 0;

  for (let row = 0; row < rows; row++) {
    const nearDepth = near + row;
    const farDepth = nearDepth + 1;
    const nearY = projectY(nearDepth);
    const farY = projectY(farDepth);
    const firstColumn = Math.floor(worldX(0, farDepth)) - 1;
    const lastColumn = Math.ceil(worldX(viewSize, farDepth)) + 1;

    for (let column = firstColumn; column < lastColumn; column++) {
      if ((column + nearDepth) % 2 !== 0) continue;
      const nearLeft = projectX(column, nearDepth);
      const nearRight = projectX(column + 1, nearDepth);
      const farLeft = projectX(column, farDepth);
      const farRight = projectX(column + 1, farDepth);
      if (
        Math.max(nearLeft, nearRight, farLeft, farRight) <= 0 ||
        Math.min(nearLeft, nearRight, farLeft, farRight) >= viewSize
      ) {
        continue;
      }
      shapes.push(
        filledShape([
          [nearLeft, nearY],
          [nearRight, nearY],
          [farRight, farY],
          [farLeft, farY],
        ]),
      );
      checks++;
    }
  }

  return {
    name: "floor",
    viewBox: { x: 0, y: 0, width: viewSize, height: viewSize },
    background: [1, 1, 1],
    parameters: {
      horizon,
      near,
      rows,
      cellWidth,
      farDepth: near + rows,
      topY: projectY(near + rows),
      checks,
    },
    shapes,
  };
}

// `renderer` is a Windfoil kernel name, or "chrome" for the browser control.
export function moireArtifactName(scene, size, renderer) {
  if (!SCENE_NAMES.has(scene)) {
    throw new Error(`unknown moire scene ${JSON.stringify(scene)}`);
  }
  positiveInteger(size, "size");
  if (renderer !== CHROME_ARTIFACT) resolveKernel(renderer);
  return `${scene}-${size}-${renderer}.png`;
}

const CHROME_ARTIFACT = "chrome";

// Same steps as rasterize.html, without the UI.
export function chromeRasterPage(svgText, sizes) {
  return `<!doctype html><meta charset="utf-8"><body><script>
const sizes = ${JSON.stringify(sizes)};
const source = new DOMParser().parseFromString(${JSON.stringify(svgText)}, "image/svg+xml").documentElement;
(async () => {
  for (const size of sizes) {
    const svg = source.cloneNode(true);
    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    svg.setAttribute("width", String(size));
    svg.setAttribute("height", String(size));
    const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], { type: "image/svg+xml" }));
    const image = new Image();
    image.decoding = "sync";
    await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = url; });
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    canvas.getContext("2d").drawImage(image, 0, 0);
    URL.revokeObjectURL(url);
    const out = document.createElement("pre");
    out.id = "chrome-" + size;
    out.textContent = canvas.toDataURL("image/png");
    document.body.append(out);
  }
})();
</script>`;
}

function chromeBinary() {
  const configured = globalThis.process?.env?.CHROME_PATH ?? null;
  if (configured) return configured;
  const platform = globalThis.process?.platform;
  if (platform === "darwin") {
    return "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  }
  return "google-chrome";
}

/** Rasterize an SVG in headless Chrome; returns PNG bytes per size. */
export async function rasterizeInChrome(svgText, sizes, { timeoutMs = 120000 } = {}) {
  const [{ spawn }, fs, os, path, url] = await Promise.all([
    import("node:child_process"),
    import("node:fs/promises"),
    import("node:os"),
    import("node:path"),
    import("node:url"),
  ]);
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "windfoil-chrome-"));
  try {
    const page = path.join(dir, "raster.html");
    await fs.writeFile(page, chromeRasterPage(svgText, sizes));
    const binary = chromeBinary();
    const dom = await new Promise((resolve, reject) => {
      // Chrome can linger after printing the DOM; finish on </html>.
      const child = spawn(binary, [
        "--headless=new",
        "--disable-extensions",
        "--no-first-run",
        "--no-default-browser-check",
        `--user-data-dir=${path.join(dir, "profile")}`,
        "--virtual-time-budget=30000",
        "--dump-dom",
        url.pathToFileURL(page).href,
      ], { stdio: ["ignore", "pipe", "ignore"] });
      let out = "";
      let settled = false;
      const settle = (error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        child.kill();
        if (error) reject(error);
        else resolve(out);
      };
      const timer = setTimeout(
        () => settle(new Error(`headless Chrome did not finish within ${timeoutMs} ms`)),
        timeoutMs,
      );
      child.stdout.setEncoding("utf8");
      child.stdout.on("data", (chunk) => {
        out += chunk;
        if (out.includes("</html>")) settle();
      });
      child.on("error", (error) =>
        settle(new Error(`could not start Chrome at ${binary} (${error.message}); set CHROME_PATH`)));
      child.on("exit", () =>
        settle(out.includes("</html>") ? null : new Error("headless Chrome exited without a DOM")));
    });
    return sizes.map((size) => {
      const match = dom.match(new RegExp(`id="chrome-${size}">data:image/png;base64,([A-Za-z0-9+/=]+)<`));
      if (!match) throw new Error(`Chrome did not rasterize the ${size}px SVG`);
      return Uint8Array.from(atob(match[1]), (char) => char.charCodeAt(0));
    });
  } finally {
    await fs.rm(dir, { recursive: true, force: true }).catch(() => {});
  }
}

function buildScene(name, options) {
  if (name === "sunburst") {
    return buildSunburst({
      spokes: options.spokes,
      lineWidth: options.lineWidth,
    });
  }
  return buildFloor({
    horizon: options.floorHorizon,
    near: options.floorNear,
    rows: options.floorRows,
    cellWidth: options.floorCellWidth,
  });
}

function prepareScene(name, options) {
  const geometry = buildScene(name, options);
  const sourceSvg = sceneToSVG(
    geometry.shapes,
    geometry.viewBox.width,
    geometry.viewBox.height,
    { background: "#ffffff", viewBox: geometry.viewBox },
  );
  // Render the serialized SVG itself so it is exact provenance.
  const source = parseFillSvg(sourceSvg);
  return {
    geometry,
    sourceSvg,
    source,
    scene: packScene(source.shapes),
    sourceName: `${name}.svg`,
  };
}

export async function runMoire(argv) {
  const options = parseMoireArgs(argv);
  const prepared = options.scenes.map((name) => prepareScene(name, options));
  const device = await requestDevice();
  const host = getWebGPUHostInfo();
  const root = runPath(host.environment, "moire", options.out);
  await mkdir(root);

  const renders = [];
  try {
    for (const item of prepared) {
      for (const size of options.sizes) {
        const raster = rasterSize(item.source.viewBox, size);
        for (const kernel of options.kernels) {
          const started = performance.now();
          const { rgba, plan } = await renderChunked(
            device,
            item.scene,
            raster,
            {
              s: [raster.scale, raster.scale],
              kernel,
              bg: item.source.background,
            },
          );
          const renderMs = performance.now() - started;
          const artifact = moireArtifactName(item.geometry.name, size, kernel);
          await writeBytes(
            `${root}/${artifact}`,
            rgbaToPng(rgba, raster.width, raster.height),
          );
          renders.push({
            scene: item.geometry.name,
            size,
            width: raster.width,
            height: raster.height,
            scale: raster.scale,
            renderer: "windfoil",
            kernel,
            filterSizePx: 1,
            renderMs,
            chunkSize: [plan.tileWidth, plan.tileHeight],
            chunkGrid: [plan.columns, plan.rows],
            artifact,
          });
          if (!options.quiet) {
            console.log(
              `rendered ${item.geometry.name} ${size}px ${kernel} to ${root}/${artifact}`,
            );
          }
        }
      }
    }

    if (options.chrome) {
      for (const item of prepared) {
        const started = performance.now();
        const pngs = await rasterizeInChrome(item.sourceSvg, options.sizes);
        const renderMs = performance.now() - started;
        for (const [index, size] of options.sizes.entries()) {
          const artifact = moireArtifactName(item.geometry.name, size, CHROME_ARTIFACT);
          await writeBytes(`${root}/${artifact}`, pngs[index]);
          renders.push({
            scene: item.geometry.name,
            size,
            width: size,
            height: size,
            renderer: "chrome",
            sessionMs: renderMs,
            artifact,
          });
          if (!options.quiet) {
            console.log(
              `rendered ${item.geometry.name} ${size}px chrome to ${root}/${artifact}`,
            );
          }
        }
      }
    }

    const sceneEntries = prepared.map((item) => [
      item.geometry.name,
      {
        viewBox: item.source.viewBox,
        ...item.geometry.parameters,
        shapes: item.source.shapes.length,
        curves: item.scene.curveCount,
        sourceSvg: item.sourceName,
      },
    ]);
    const manifest = {
      schemaVersion: 1,
      status: "ok",
      demo: "moire",
      engine: {
        name: "windfoil",
        environment: host.environment,
        backend: host.backend,
      },
      scenes: Object.fromEntries(sceneEntries),
      system: {
        adapter: host.adapterInfo ?? null,
        limits: host.limits ?? null,
      },
      renders,
      artifacts: {
        sourceSvgs: Object.fromEntries(
          prepared.map((item) => [item.geometry.name, item.sourceName]),
        ),
      },
    };
    await Promise.all([
      ...prepared.map((item) =>
        writeText(`${root}/${item.sourceName}`, item.sourceSvg),
      ),
      writeText(
        `${root}/manifest.json`,
        JSON.stringify(manifest, null, 2) + "\n",
      ),
    ]);
    return { root, manifest };
  } finally {
    device.destroy();
  }
}

async function isMainModule() {
  if (typeof Deno !== "undefined" && Deno.version?.deno) {
    return import.meta.main;
  }
  if (
    typeof process !== "undefined" &&
    process.versions?.node &&
    process.argv[1]
  ) {
    const { pathToFileURL } = await import("node:url");
    return pathToFileURL(process.argv[1]).href === import.meta.url;
  }
  return false;
}

if (await isMainModule()) await runMain(runMoire);
