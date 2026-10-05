import assert from "node:assert/strict";
import test from "node:test";

import { packScene } from "../../js/prep.js";
import { parseFillSvg } from "../../demos/render/svg.js";
import {
  buildFloor,
  buildSunburst,
  chromeRasterPage,
  MOIRE_DEFAULTS,
  moireArtifactName,
  parseMoireArgs,
} from "../../demos/moire/cli.js";
import { sceneToSVG } from "../../demos/util/svg.js";

test("moire defaults render both scenes and paper sizes through every filter", () => {
  assert.deepEqual(parseMoireArgs([]), {
    scenes: ["sunburst", "floor"],
    sizes: [256, 512, 1024],
    kernels: ["box", "tent", "cubic"],
    spokes: 256,
    lineWidth: 3,
    floorHorizon: 80,
    floorNear: 5,
    floorRows: 96,
    floorCellWidth: 40,
    chrome: false,
    out: null,
    quiet: false,
  });
  assert.deepEqual(
    parseMoireArgs([
      "--scene=floor",
      "--size=256,512",
      "--kernel",
      "tent,cubic",
      "--spokes=90",
      "--line-width",
      "1.25",
      "--floor-horizon=72",
      "--floor-near=4",
      "--floor-rows=64",
      "--floor-cell-width=48",
      "--chrome",
      "--out=output/moire",
      "--quiet",
    ]),
    {
      scenes: ["floor"],
      sizes: [256, 512],
      kernels: ["tent", "cubic"],
      spokes: 90,
      lineWidth: 1.25,
      floorHorizon: 72,
      floorNear: 4,
      floorRows: 64,
      floorCellWidth: 48,
      chrome: true,
      out: "output/moire",
      quiet: true,
    },
  );
});

test("moire CLI rejects ambiguous or invalid comparisons", () => {
  assert.throws(() => parseMoireArgs(["plain"]), /unexpected argument/);
  assert.throws(() => parseMoireArgs(["--dimension=512"]), /unknown option/);
  assert.throws(() => parseMoireArgs(["--scene=grid"]), /unknown moire scene/);
  assert.throws(() => parseMoireArgs(["--scene=floor,floor"]), /duplicate values/);
  assert.throws(() => parseMoireArgs(["--kernel=gauss"]), /unknown filter kernel/);
  assert.throws(() => parseMoireArgs(["--kernel=box,box"]), /duplicate values/);
  assert.throws(() => parseMoireArgs(["--size=256,"]), /without empty values/);
  assert.throws(() => parseMoireArgs(["--size=0"]), /positive integer/);
  assert.throws(() => parseMoireArgs(["--floor-near=1.5"]), /positive integer/);
  assert.throws(() => parseMoireArgs(["--line-width=0"]), /positive number/);
  assert.throws(() => parseMoireArgs(["--quiet=true"]), /does not take a value/);
  assert.throws(() => parseMoireArgs(["--chrome=true"]), /does not take a value/);
});

test("sunburst geometry is a centered set of tapered triangular spokes", () => {
  const geometry = buildSunburst();
  assert.deepEqual(geometry.viewBox, { x: 0, y: 0, width: 512, height: 512 });
  assert.deepEqual(geometry.background, [1, 1, 1]);
  assert.deepEqual(geometry.parameters, {
    spokes: 256,
    lineWidth: 3,
    radius: 235,
  });
  assert.equal(geometry.shapes.length, MOIRE_DEFAULTS.sunburst.spokes);

  const first = geometry.shapes[0];
  assert.equal(first.curves.length, 18);
  assert.deepEqual(Array.from(first.curves.slice(0, 2)), [256, 256]);
  assert.deepEqual(Array.from(first.curves.slice(-2)), [256, 256]);
  const ax = first.curves[4], ay = first.curves[5];
  const bx = first.curves[10], by = first.curves[11];
  assert.ok(Math.abs(Math.hypot(ax - bx, ay - by) - 3) < 1e-10);
  assert.ok(Math.abs(Math.hypot(ax - 256, ay - 256) - 235) < 1e-10);
  const packed = packScene(geometry.shapes);
  assert.equal(packed.shapeData.length, 256 * 16);
  assert.equal(packed.curveCount, 256 * 3);
  assert.ok(packed.pieceCount >= packed.curveCount);
});

test("sunburst geometry keeps a white gap between adjacent outer spokes", () => {
  assert.throws(
    () => buildSunburst({ spokes: 256, lineWidth: 6, radius: 235 }),
    /leave a gap/,
  );
});

test("floor geometry is a projected square checkerboard that contracts to the horizon", () => {
  const geometry = buildFloor();
  assert.deepEqual(geometry.viewBox, { x: 0, y: 0, width: 512, height: 512 });
  assert.deepEqual(geometry.background, [1, 1, 1]);
  assert.equal(geometry.parameters.horizon, 80);
  assert.equal(geometry.parameters.near, 5);
  assert.equal(geometry.parameters.rows, 96);
  assert.equal(geometry.parameters.cellWidth, 40);
  assert.equal(geometry.parameters.farDepth, 101);
  assert.ok(Math.abs(geometry.parameters.topY - 101.38613861386139) < 1e-10);
  assert.equal(geometry.parameters.checks, 6620);
  assert.equal(geometry.shapes.length, geometry.parameters.checks);

  const first = geometry.shapes[0];
  assert.equal(first.curves.length, 24);
  assert.deepEqual(first.color, [0, 0, 0]);
  assert.equal(first.alpha, 1);
  assert.equal(first.fillRule, "nonzero");
  assert.equal(first.curves[1], 512);
  assert.equal(first.curves[5], 512);
  assert.equal(first.curves[11], 440);
  assert.equal(first.curves[17], 440);
  assert.equal(first.curves[23], 512);

  const packed = packScene(geometry.shapes);
  assert.equal(packed.curveCount, geometry.parameters.checks * 4);
  assert.ok(packed.pieceCount >= packed.curveCount);
});

test("floor and artifact parameters are strictly validated", () => {
  assert.throws(() => buildFloor({ horizon: 512 }), /horizon.*inside/);
  assert.throws(() => buildFloor({ near: 0 }), /positive integer/);
  assert.throws(() => buildFloor({ rows: 1.5 }), /positive integer/);
  assert.throws(() => buildFloor({ cellWidth: 0 }), /positive number/);
  assert.equal(moireArtifactName("sunburst", 256, "box"), "sunburst-256-box.png");
  assert.equal(moireArtifactName("floor", 1024, "cubic"), "floor-1024-cubic.png");
  assert.equal(moireArtifactName("sunburst", 256, "chrome"), "sunburst-256-chrome.png");
  assert.throws(() => moireArtifactName("grid", 256, "box"), /unknown moire scene/);
  assert.throws(() => moireArtifactName("floor", 256, "gauss"), /unknown filter kernel/);
});

test("both default scenes survive the exact SVG provenance round trip", () => {
  for (const geometry of [buildSunburst(), buildFloor()]) {
    const svg = sceneToSVG(
      geometry.shapes,
      geometry.viewBox.width,
      geometry.viewBox.height,
      { background: "#ffffff", viewBox: geometry.viewBox },
    );
    const source = parseFillSvg(svg);
    assert.deepEqual(source.viewBox, geometry.viewBox);
    assert.deepEqual(source.background, [1, 1, 1]);
    assert.equal(source.shapes.length, geometry.shapes.length);
    assert.equal(packScene(source.shapes).curveCount, packScene(geometry.shapes).curveCount);
  }
});

test("the Chrome control page decodes each size separately and draws it 1:1", () => {
  const page = chromeRasterPage('<svg viewBox="0 0 4 4"></svg>', [256, 512]);
  assert.match(page, /const sizes = \[256,512\];/);
  assert.match(page, /svg\.setAttribute\("width", String\(size\)\)/);
  assert.match(page, /drawImage\(image, 0, 0\)/);
  assert.match(page, /"chrome-" \+ size/);
  // The SVG travels as a JSON string literal, so markup cannot break out.
  assert.ok(page.includes(JSON.stringify('<svg viewBox="0 0 4 4"></svg>')));
});
