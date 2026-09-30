import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { parseFillSvg, rasterDimensions, rasterSize } from '../../demos/render/svg.js';
import { sceneToSVG } from '../../demos/util/svg.js';

const loop = {
  curves: Float64Array.of(
    2, 3, 8, 1, 12, 3,
    12, 3, 8, 9, 2, 3,
  ),
  color: [0.2, 0.4, 0.6],
  alpha: 0.75,
  fillRule: 'evenodd',
};

test('parses SVG exported by sceneToSVG', () => {
  const parsed = parseFillSvg(sceneToSVG([loop], 16, 12));
  assert.deepEqual(parsed.viewBox, { x: 0, y: 0, width: 16, height: 12 });
  assert.deepEqual(parsed.background, [1, 1, 1]);
  assert.equal(parsed.shapes.length, 1);
  assert.deepEqual([...parsed.shapes[0].curves], [...loop.curves]);
  assert.deepEqual(parsed.shapes[0].color, [0.2, 0.4, 0.6]);
  assert.equal(parsed.shapes[0].alpha, 0.75);
  assert.equal(parsed.shapes[0].fillRule, 'evenodd');
});

test('parses the bundled spoke fixture', async () => {
  const source = await readFile(new URL('../../demos/render/spokes.svg', import.meta.url), 'utf8');
  const parsed = parseFillSvg(source);
  assert.equal(parsed.shapes.length, 180);
  assert.deepEqual(parsed.background, [1, 1, 1]);
});

test('bundled roundtrip stars share geometry and differ only by fill rule', async () => {
  const [nonzeroSource, evenoddSource] = await Promise.all([
    readFile(new URL('../../demos/roundtrip/star-nonzero.svg', import.meta.url), 'utf8'),
    readFile(new URL('../../demos/roundtrip/star-evenodd.svg', import.meta.url), 'utf8'),
  ]);
  const nonzero = parseFillSvg(nonzeroSource);
  const evenodd = parseFillSvg(evenoddSource);
  assert.deepEqual([...nonzero.shapes[0].curves], [...evenodd.shapes[0].curves]);
  assert.equal(nonzero.shapes[0].fillRule, 'nonzero');
  assert.equal(evenodd.shapes[0].fillRule, 'evenodd');
});

test('accepts an omitted background and preserves the viewBox origin', () => {
  const source = sceneToSVG([loop], 16, 12, { background: null })
    .replace('viewBox="0 0 16 12"', 'viewBox="-2 -3 16 12"');
  const parsed = parseFillSvg(source);
  assert.deepEqual(parsed.viewBox, { x: -2, y: -3, width: 16, height: 12 });
  assert.equal(parsed.background, null);
});

test('accepts matching physical root dimensions', () => {
  const source = '<svg xmlns="http://www.w3.org/2000/svg" width="424mm" height="600mm" ' +
    'viewBox="0 0 424 600"><path d="M 0 0 L 1 0 L 0 1 Z" fill="#000000"/></svg>';
  const parsed = parseFillSvg(source);
  assert.deepEqual(parsed.viewBox, { x: 0, y: 0, width: 424, height: 600 });
  assert.equal(parsed.shapes.length, 1);
});

const validPath = 'd="M 0 0 Q 5 0 10 0 Q 5 10 0 0 Z" fill="#336699" ' +
  'fill-opacity="1.0000" fill-rule="nonzero"';
const svg = (child, root = '') =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10" width="10" height="10"${root}>${child}</svg>`;

test('parses simple line paths, relative commands, implicit close, and SVG fill defaults', () => {
  const parsed = parseFillSvg(svg(
    '<path d="M 1,1 L 9 1 9 9 h -8 v -8 z" fill="#336699"/>',
  ));
  assert.equal(parsed.shapes[0].fillRule, 'nonzero');
  assert.equal(parsed.shapes[0].alpha, 1);
  assert.equal(parsed.shapes[0].curves.length, 4 * 6);
  assert.deepEqual([...parsed.shapes[0].curves.slice(-2)], [1, 1]);
});

test('preserves an evenodd rule and a nonzero viewBox background origin', () => {
  const source = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-5 -7 10 10" width="10" height="10">' +
    '<rect x="-5" y="-7" width="10" height="10" fill="#ffffff"/>' +
    '<path d="M -4 -6 Q 0 -7 4 -6 L -4 -6 Z" fill="#336699" fill-rule="evenodd"/>' +
    '</svg>';
  const parsed = parseFillSvg(source);
  assert.deepEqual(parsed.viewBox, { x: -5, y: -7, width: 10, height: 10 });
  assert.equal(parsed.shapes[0].fillRule, 'evenodd');
});

test('preserves compound-path contour boundaries through SVG export', () => {
  const source = svg(
    '<path d="M 1 1 L 4 1 L 2 4 Z M 6 6 L 9 6 L 8 9 Z" ' +
      'fill="#336699" fill-rule="evenodd"/>',
  );
  const parsed = parseFillSvg(source);
  assert.deepEqual(parsed.shapes[0].contours, [3, 3]);
  const exported = sceneToSVG(parsed.shapes, 10, 10, { background: null });
  const roundtripped = parseFillSvg(exported);
  assert.deepEqual(roundtripped.shapes[0].contours, [3, 3]);
  assert.deepEqual([...roundtripped.shapes[0].curves], [...parsed.shapes[0].curves]);
});

test('ignores data-* metadata and accepts rgb() fills', () => {
  const source = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10" width="10" height="10" ' +
    'data-prompt="salt &amp; pepper, &quot;quoted&quot; &#233; > 1" data-seed="14">' +
    '<rect width="10" height="10" fill="rgb(0, 128,255)" data-role="background"/>' +
    `<path ${validPath} data-alpha="0.340169439" data-color="0.8 0.8 0.8"/>` +
    '</svg>';
  const parsed = parseFillSvg(source);
  assert.deepEqual(parsed.background, [0, 128 / 255, 1]);
  assert.equal(parsed.shapes.length, 1);
  assert.deepEqual(parsed.shapes[0].color, [0x33 / 255, 0x66 / 255, 0x99 / 255]);
  assert.equal(parsed.shapes[0].alpha, 1);
});

const rejected = [
  ['declarations', `<?xml version="1.0"?>${svg('')}`],
  ['doctype', `<!DOCTYPE svg>${svg('')}`],
  ['comments', svg('<!-- no -->')],
  ['unknown root attribute', svg('', ' preserveAspectRatio="none"')],
  ['duplicate attribute', svg(`<path ${validPath} fill="#000000"/>`)],
  ['unsupported units', '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10" width="10px" height="10"></svg>'],
  ['mismatched root dimensions', '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10" width="20" height="10"></svg>'],
  ['nested elements', svg(`<path ${validPath}></path>`)],
  ['groups', svg('<g/>')],
  ['scripts', svg('<script/>')],
  ['styles', svg('<style/>')],
  ['images', svg('<image/>')],
  ['uses', svg('<use href="https://example.com/x"/>')],
  ['defs', svg('<defs/>')],
  ['filters', svg('<filter/>')],
  ['transform', svg(`<path ${validPath} transform="scale(2)"/>`)],
  ['stroke', svg(`<path ${validPath} stroke="#000000"/>`)],
  ['style attribute', svg(`<path ${validPath} style="opacity: 1"/>`)],
  ['filter attribute', svg(`<path ${validPath} filter="url(#x)"/>`)],
  ['external reference', svg(`<path ${validPath} href="https://example.com/x"/>`)],
  ['unsupported path command', svg(`<path ${validPath.replace('Q 5 0 10 0', 'C 1 1 2 2 10 0')}/>`)],
  ['missing close command', svg(`<path ${validPath.replace(' Z"', '"')}/>`)],
  ['non-finite coordinate', svg(`<path ${validPath.replace('Q 5 0 10 0', 'Q 1e999 0 10 0')}/>`)],
  ['short color', svg(`<path ${validPath.replace('#336699', '#369')}/>`)],
  ['named color', svg(`<path ${validPath.replace('#336699', 'red')}/>`)],
  ['wrong fill rule', svg(`<path ${validPath.replace('nonzero', 'inherit')}/>`)],
  ['opacity above one', svg(`<path ${validPath.replace('1.0000', '1.1')}/>`)],
  ['non-numeric opacity', svg(`<path ${validPath.replace('1.0000', 'NaN')}/>`)],
  ['rect after path', svg(`<path ${validPath}/><rect width="10" height="10" fill="#ffffff"/>`)],
  ['partial background', svg('<rect width="9" height="10" fill="#ffffff"/>')],
  ['background opacity', svg('<rect width="10" height="10" fill="#ffffff" fill-opacity="0.5"/>')],
  ['second background', svg('<rect width="10" height="10" fill="#ffffff"/><rect width="10" height="10" fill="#ffffff"/>')],
  ['entity', svg(`<path ${validPath.replace('#336699', '&quot;')}/>`)],
  ['bare ampersand in data attribute', svg(`<path ${validPath} data-note="a & b"/>`)],
  ['markup in data attribute', svg(`<path ${validPath} data-note="<b>"/>`)],
  ['empty data attribute name', svg(`<path ${validPath} data-="x"/>`)],
  ['rgb channel above 255', svg('<rect width="10" height="10" fill="rgb(256,0,0)"/>')],
  ['fractional rgb channel', svg('<rect width="10" height="10" fill="rgb(0.5,0,0)"/>')],
  ['trailing content', `${svg('')}<svg/>`],
  ['missing root close', svg('').replace('</svg>', '')],
];

for (const [name, source] of rejected) {
  test(`rejects ${name}`, () => {
    assert.throws(() => parseFillSvg(source), /Invalid SVG:/);
  });
}

test('requires string input', () => {
  assert.throws(() => parseFillSvg(new Uint8Array()), /must be a string/);
});

test('fits the longest SVG dimension and preserves its aspect ratio', () => {
  assert.deepEqual(rasterSize({ x: 0, y: 0, width: 200, height: 100 }, 4096), {
    width: 4096,
    height: 2048,
    scale: 200 / 4096,
    origin: [0, 0],
    content: { x: 0, y: 0, width: 4096, height: 2048 },
  });
  const rounded = rasterSize({ x: 10, y: 20, width: 3, height: 2 }, 4);
  assert.equal(rounded.width, 4);
  assert.equal(rounded.height, 3);
  assert.ok(rounded.width * rounded.scale >= 3);
  assert.ok(rounded.height * rounded.scale >= 2);
});

test('accepts an exact raster resolution and fits the viewBox without distortion', () => {
  const raster = rasterDimensions({ x: 0, y: 0, width: 424, height: 600 }, 5008, 7087);
  assert.deepEqual([raster.width, raster.height], [5008, 7087]);
  assert.equal(raster.scale, 424 / 5008);
  assert.equal(raster.origin[0], 0);
  assert.ok(raster.origin[1] < 0);
  assert.throws(
    () => rasterDimensions({ x: 0, y: 0, width: 10, height: 10 }, 0, 10),
    /positive integers/,
  );
});

test('expand mode centers a square viewBox exactly in a wider raster', () => {
  const raster = rasterDimensions({ x: 0, y: 0, width: 1000, height: 1000 }, 1240, 1125);
  assert.equal(raster.scale, 1000 / 1125);
  assert.deepEqual(raster.content, { x: 57.5, y: 0, width: 1125, height: 1125 });
  assert.equal(raster.origin[0], -57.5 * raster.scale);
  assert.equal(raster.origin[1], 0);
});

test('letterbox mode snaps the content offset to whole pixels on either axis', () => {
  const wide = rasterDimensions({ x: 5, y: 5, width: 1000, height: 1000 }, 1240, 1125, { mode: 'letterbox' });
  assert.deepEqual(wide.content, { x: 57, y: 0, width: 1125, height: 1125 });
  assert.equal(wide.origin[0], 5 - 57 * wide.scale);
  assert.equal(wide.origin[1], 5);
  const tall = rasterDimensions({ x: 0, y: 0, width: 1000, height: 1000 }, 1125, 1240, { mode: 'letterbox' });
  assert.deepEqual(tall.content, { x: 0, y: 57, width: 1125, height: 1125 });
  const fractional = rasterDimensions({ x: 0, y: 0, width: 3, height: 2 }, 1000, 1000, { mode: 'letterbox' });
  assert.equal(fractional.content.y, 166);
  assert.ok(Math.abs(fractional.content.height - 2000 / 3) < 1e-9);
  assert.throws(
    () => rasterDimensions({ x: 0, y: 0, width: 1, height: 1 }, 1, 1, { mode: 'crop' }),
    /scale mode/,
  );
});
