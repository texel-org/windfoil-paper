import assert from 'node:assert/strict';
import test from 'node:test';
import { exportSize, imageSize, previewSize } from '../../demos/util/dom.js';
import { buildFitModel } from '../../demos/util/engine.js';

// Full-image sizing must preserve portraits and small inputs as well as landscapes.
test('web image sizing caps at 1080p without upscaling or cropping', () => {
  assert.deepEqual(imageSize(3840, 2160), { width: 1920, height: 1080 });
  assert.deepEqual(imageSize(4000, 2000), { width: 1920, height: 960 });
  assert.deepEqual(imageSize(2000, 4000), { width: 540, height: 1080 });
  assert.deepEqual(imageSize(320, 200), { width: 320, height: 200 });
  assert.deepEqual(imageSize(100, 200), { width: 100, height: 200 });
  assert.deepEqual(imageSize(640, 480, 512), { width: 512, height: 384 });
  assert.deepEqual(imageSize(320, 200, 512), { width: 320, height: 200 });
});

test('web optimization size presets preserve aspect ratio without upscaling', () => {
  for (const size of [256, 512, 1080, 2048, 4096]) {
    assert.deepEqual(imageSize(8192, 4096, String(size)), { width: size, height: size / 2 });
    assert.deepEqual(imageSize(4096, 8192, String(size)), { width: size / 2, height: size });
    assert.deepEqual(imageSize(100, 200, String(size)), { width: 100, height: 200 });
  }
  assert.deepEqual(imageSize(3840, 2160, '1080p'), { width: 1920, height: 1080 });
  assert.deepEqual(imageSize(3840, 2160, '4096'), { width: 3840, height: 2160 });
});

test('web shape and line models initialize rectangular targets', () => {
  const width = 64, height = 32;
  const target = new Float32Array(width * height * 4).fill(1);
  for (const mode of ['shape', 'line', 'capsule']) {
    const { model } = buildFitModel({ mode, n: 8, width, height, seed: 7, target });
    const decoded = model.decode();
    assert.equal(decoded.shapes.length, 8);
    assert.ok(decoded.scene.pieceCount > 0);
    assert.ok(decoded.scene.shapeData.every(Number.isFinite));
  }
});

test('web engine rebuilds its renderer when image dimensions change', async () => {
  const { Renderer } = await import('../../js/renderer.js');
  const { Engine } = await import('../../demos/util/engine.js');
  const originalCreate = Renderer.create;
  const created = [];
  Renderer.create = async (_device, options) => {
    const renderer = {
      ...options, destroyed: false, frames: 0, targets: 0,
      uploadTarget(target) {
        assert.equal(target.length, this.width * this.height * 4);
        this.targets++;
      },
      uploadScene(_scene, settings) { assert.deepEqual(settings.s, [1, 1]); },
      forwardNoRead() { this.frames++; },
      destroy() { this.destroyed = true; },
    };
    created.push(renderer);
    return renderer;
  };
  try {
    const engine = new Engine({}, {
      width: 64, height: 32,
      build: () => buildFitModel({ mode: 'shape', n: 8, width: engine.width, height: engine.height, seed: 7 }),
      settings: () => ({ s: [1, 1] }),
    });
    engine.setTarget(new Float32Array(64 * 32 * 4));
    await engine.reset();
    assert.equal(engine.renderer.width, 64);
    assert.equal(engine.renderer.train.blur, false, 'standard L2 does not train blur');
    assert.equal(engine.snapshot().width, 64);
    engine.width = 32;
    engine.height = 64;
    // A live camera frame must not be uploaded to the old-size renderer.
    engine.updateTarget(new Float32Array(32 * 64 * 4));
    assert.equal(created[0].targets, 1);
    await engine.reset(false);
    assert.equal(created.length, 2);
    assert.equal(engine.renderer.height, 64);
    assert.equal(created[0].destroyed, true);
    assert.equal(engine.snapshot().height, 64);
    await engine.reset(false);
    assert.equal(created.length, 2, 'unchanged dimensions reuse the renderer');
  } finally {
    Renderer.create = originalCreate;
  }
});

test('web palette controls configure both shape and line models', () => {
  for (const mode of ['shape', 'line', 'capsule']) {
    const built = buildFitModel({ mode, n: 8, width: 64, height: 32, seed: 7, colorCount: 4 });
    assert.equal(built.model.style.paletteColors(built.model.params).length, 4);
    built.model.harden();
    const colors = built.model.decode().shapes.map((shape) => shape.color.join(','));
    assert.ok(new Set(colors).size <= 4);
  }
});

test('web engine replaces the renderer when blend mode changes', async () => {
  const { Renderer } = await import('../../js/renderer.js');
  const { Engine } = await import('../../demos/util/engine.js');
  const originalCreate = Renderer.create;
  const created = [];
  Renderer.create = async (_device, options) => {
    const renderer = {
      ...options, destroyed: false,
      uploadTarget() {}, uploadScene() {}, forwardNoRead() {},
      destroy() { this.destroyed = true; },
    };
    created.push(renderer);
    return renderer;
  };
  try {
    const engine = new Engine({}, {
      width: 64, height: 32,
      build: () => buildFitModel({ mode: 'shape', n: 8, width: 64, height: 32, seed: 7 }),
      settings: () => ({ s: [1, 1] }),
    });
    engine.setTarget(new Float32Array(64 * 32 * 4));
    await engine.reset();
    engine.blend = 'multiply';
    await engine.reset();
    assert.equal(engine.renderer.blend, 'multiply');
    assert.equal(created.length, 2);
    assert.equal(created[0].destroyed, true);
  } finally {
    Renderer.create = originalCreate;
  }
});

test('preview resolution follows display size and DPR, while PNG export fills 1080p bounds', () => {
  assert.deepEqual(previewSize(500, 281.25, 2), { width: 1000, height: 563 });
  assert.deepEqual(previewSize(400, 250, 1), { width: 400, height: 250 });
  assert.deepEqual(exportSize(320, 180), { width: 1920, height: 1080 });
  assert.deepEqual(exportSize(100, 100), { width: 1080, height: 1080 });
  assert.deepEqual(exportSize(200, 400), { width: 540, height: 1080 });
});

test('opaque mode fixes alpha for unrestricted and learned colors on every web model', () => {
  for (const mode of ['shape', 'line', 'capsule']) {
    for (const colorCount of [1, 4]) {
      const options = { mode, n: 8, width: 64, height: 32, seed: 7, colorCount };
      const opaque = buildFitModel({ ...options, opaque: true }).model;
      assert.equal(opaque.style.trainsAlpha, false);
      assert.ok(opaque.decode().shapes.every((shape) => shape.alpha === 1));
      const learned = buildFitModel(options).model;
      assert.equal(learned.style.trainsAlpha, true);
      assert.ok(learned.decode().shapes.some((shape) => shape.alpha < 1));
    }
  }
});

test('hard lines and round capsules use their respective existing outlines', () => {
  const options = { n: 1, width: 64, height: 32, seed: 7 };
  const line = buildFitModel({ ...options, mode: 'line' }).model.decode().shapes[0];
  const capsule = buildFitModel({ ...options, mode: 'capsule' }).model.decode().shapes[0];
  assert.equal(line.curves.length, 4 * 6);
  assert.equal(capsule.curves.length, 6 * 6);
});

test('SVG download retains scene coordinates and opacity and rejects other blend modes', async () => {
  const { svgBlob } = await import('../../demos/web/render.js');
  const { model } = buildFitModel({ mode: 'capsule', n: 2, width: 64, height: 32, seed: 7, opaque: true });
  const snapshot = { shapes: model.decode().shapes, width: 64, height: 32,
    background: [1, 1, 1], blend: 'src-over' };
  const svg = await svgBlob(snapshot).text();
  assert.match(svg, /viewBox="0 0 64 32"/);
  assert.equal((svg.match(/<path /g) ?? []).length, 2);
  assert.match(svg, /fill-opacity="1.0000"/);
  for (const blend of ['add', 'multiply', 'screen']) {
    assert.throws(() => svgBlob({ ...snapshot, blend }), /normal blending/);
  }
});

test('PNG download rerenders a small scene with Windfoil at export resolution', async () => {
  const { Renderer } = await import('../../js/renderer.js');
  const { pngBlob } = await import('../../demos/web/render.js');
  const { decode } = await import('fast-png');
  const originalCreate = Renderer.create;
  let settings, destroyed = false, rendered = false;
  Renderer.create = async (_device, options) => {
    assert.equal(options.width, 1920);
    assert.equal(options.height, 1080);
    assert.equal(options.blend, 'multiply');
    return {
      uploadScene(_scene, value) { settings = value; },
      async forward() {
        rendered = true;
        return new Float32Array(options.width * options.height * 4);
      },
      destroy() { destroyed = true; },
    };
  };
  try {
    const blob = await pngBlob({}, { width: 320, height: 180, blend: 'multiply',
      scene: {}, background: [1, 1, 1], maxShapes: 2, maxPieces: 48, maxCurves: 16 });
    const image = decode(new Uint8Array(await blob.arrayBuffer()));
    assert.equal(image.width, 1920);
    assert.equal(image.height, 1080);
    assert.deepEqual(settings.s, [1 / 6, 1 / 6]);
    assert.equal(rendered, true);
    assert.equal(destroyed, true);
  } finally {
    Renderer.create = originalCreate;
  }
});

test('export snapshots remain unchanged as the live model advances', async () => {
  const { Engine } = await import('../../demos/util/engine.js');
  const built = buildFitModel({ mode: 'shape', n: 2, width: 64, height: 32, seed: 7 });
  const engine = new Engine({}, { width: 64, height: 32, build: () => built, settings: () => ({ s: [1, 1] }) });
  engine.model = built.model;
  engine.setTarget(new Float32Array(64 * 32 * 4), [0.1, 0.2, 0.3]);
  const saved = engine.snapshot(true);
  const scene = saved.scene.pieceData.slice();
  const x = saved.shapes[0].curves[0];
  engine.model.params.ax[0] += 10;
  engine.background[0] = 0.9;
  const next = engine.snapshot(true);
  assert.deepEqual(saved.scene.pieceData, scene);
  assert.equal(saved.shapes[0].curves[0], x);
  assert.equal(saved.background[0], 0.1);
  assert.notEqual(next.shapes[0].curves[0], x);
});

test('per-shape blur creates independent trainable native filters for every web model', () => {
  for (const mode of ['shape', 'line', 'capsule']) {
    const options = { mode, n: 2, width: 64, height: 32, seed: 7 };
    const sharp = buildFitModel(options).model;
    assert.equal(sharp.params.blur, undefined);
    const learned = buildFitModel({ ...options, learnBlur: true }).model;
    assert.equal(learned.params.blur.length, 2);
    assert.ok(learned.decode().shapes.every((shape) => Math.abs(shape.s - 7) < 1e-5));
    learned.params.blur[0] += Math.log(2);
    const shapes = learned.decode().shapes;
    assert.ok(shapes[0].s > 7 && shapes[0].s < 32);
    assert.ok(Math.abs(shapes[1].s - 7) < 1e-5);
  }
});

test('web learned blur stays within 1–32 pixels and has the correct bounded gradient', () => {
  for (const mode of ['shape', 'line', 'capsule']) {
    const model = buildFitModel({ mode, n: 1, width: 64, height: 32, seed: 7, learnBlur: true }).model;
    for (const parameter of [-1000, -1, 0, 1, 1000]) {
      model.params.blur[0] = parameter;
      const width = model.decode().shapes[0].s;
      assert.ok(width >= 1 && width <= 32);
      const analytic = model.pullback({ blurGrads: Float32Array.of(1, 2) }).blur[0];
      const h = 0.001;
      model.params.blur[0] = parameter + h;
      const plus = model.decode().shapes[0].s;
      model.params.blur[0] = parameter - h;
      const minus = model.decode().shapes[0].s;
      assert.ok(Math.abs(analytic - 3 * (plus - minus) / (2 * h)) < 0.002);
    }
  }
});
