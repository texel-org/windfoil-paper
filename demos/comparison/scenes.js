// Scene specs (glyph:<char>, svg:<file>) to flat quads in pixel coordinates.

import opentype from 'opentype.js';

import { readBytes, readText } from '../util/runtime.js';
import { parseSVG } from './svg-parse.js';

export const LATO_PATH = 'fixtures/lato/Lato-Regular.ttf';
export const LATO_HINT = 'run `npm run fixtures:lato` to download it';

/** Parse a TrueType font file. Outlines come out in font units, Y-down, baseline at 0. */
export async function loadFont(path = LATO_PATH) {
  let bytes;
  try {
    bytes = await readBytes(path);
  } catch (error) {
    if (error?.code !== 'ENOENT' && error?.name !== 'NotFound') throw error;
    throw new Error(`${path} is missing: ${LATO_HINT}`, { cause: error });
  }
  return opentype.parse(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

// A cubic as two quadratics; only CFF outlines reach it, TrueType is quadratic.
function cubicToQuads(x0, y0, c1x, c1y, c2x, c2y, x1, y1, out) {
  const m = (a, b) => (a + b) / 2;
  const ax = m(x0, c1x), ay = m(y0, c1y), bx = m(c1x, c2x), by = m(c1y, c2y);
  const cx = m(c2x, x1), cy = m(c2y, y1), dx = m(ax, bx), dy = m(ay, by), ex = m(bx, cx), ey = m(by, cy);
  const mx = m(dx, ex), my = m(dy, ey);
  out.push(x0, y0, 1.5 * dx - 0.25 * (x0 + mx), 1.5 * dy - 0.25 * (y0 + my), mx, my);
  out.push(mx, my, 1.5 * ex - 0.25 * (mx + x1), 1.5 * ey - 0.25 * (my + y1), x1, y1);
}

/** A glyph's outline as flat quads in font units with its ink bbox, or null when blank. */
export function glyphQuads(font, ch) {
  const path = font.charToGlyph(ch).getPath(0, 0, font.unitsPerEm);
  const quads = [];
  const line = (x0, y0, x1, y1) => quads.push(x0, y0, (x0 + x1) / 2, (y0 + y1) / 2, x1, y1);
  let cx = 0, cy = 0, sx = 0, sy = 0;
  for (const c of path.commands) {
    if (c.type === 'M') {
      cx = sx = c.x;
      cy = sy = c.y;
    } else if (c.type === 'L') {
      line(cx, cy, c.x, c.y);
      cx = c.x;
      cy = c.y;
    } else if (c.type === 'Q') {
      quads.push(cx, cy, c.x1, c.y1, c.x, c.y);
      cx = c.x;
      cy = c.y;
    } else if (c.type === 'C') {
      cubicToQuads(cx, cy, c.x1, c.y1, c.x2, c.y2, c.x, c.y, quads);
      cx = c.x;
      cy = c.y;
    } else if (c.type === 'Z') {
      if (cx !== sx || cy !== sy) line(cx, cy, sx, sy);
      cx = sx;
      cy = sy;
    }
  }
  if (quads.length === 0) return null;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < quads.length; i += 2) {
    x0 = Math.min(x0, quads[i]);
    x1 = Math.max(x1, quads[i]);
    y0 = Math.min(y0, quads[i + 1]);
    y1 = Math.max(y1, quads[i + 1]);
  }
  return { quads, bbox: [x0, y0, x1, y1] };
}

/** One glyph fitted centred in a cell x cell box with `pad` px of margin. */
export function glyphShape(font, ch, cell, pad = cell * (14 / 128)) {
  const g = glyphQuads(font, ch);
  if (!g) throw new Error(`glyph '${ch}' has no outline in this font (blank or missing)`);
  const [x0, y0, x1, y1] = g.bbox;
  const gw = x1 - x0, gh = y1 - y0, box = cell - 2 * pad;
  const k = Math.min(box / gw, box / gh);
  const ox = pad + (box - gw * k) / 2 - x0 * k, oy = pad + (box - gh * k) / 2 - y0 * k;
  return g.quads.map((v, i) => (i % 2 === 0 ? ox + v * k : oy + v * k));
}

/** Translate flat quads; sets a scene's sub-pixel phase against the pixel grid. */
export const translateQuads = (quads, dx, dy = dx) =>
  dx === 0 && dy === 0 ? quads.slice() : quads.map((v, i) => v + (i % 2 === 0 ? dx : dy));

/** 'star {5/2}' -> 'star_5_2'. */
export const slugify = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');

/** Split a spec into its kind and argument; a bare *.svg path or single character is shorthand. */
export function sceneKind(spec) {
  const colon = spec.indexOf(':');
  const kind = colon < 0 ? null : spec.slice(0, colon);
  if (kind === 'glyph' || kind === 'svg') return { kind, rest: spec.slice(colon + 1) };
  if (spec.toLowerCase().endsWith('.svg')) return { kind: 'svg', rest: spec };
  if ([...spec].length === 1) return { kind: 'glyph', rest: spec };
  throw new Error(`unknown scene ${JSON.stringify(spec)} (expected glyph:<char> or svg:<file>)`);
}

/** Resolve a spec to { spec, label, slug, key, quads, evenodd, warnings }. */
export async function resolveScene(spec, { font, size, offset = 0, fit = 'viewbox' }) {
  const { kind, rest } = sceneKind(spec);
  const place = (quads) => translateQuads(quads, offset);

  if (kind === 'svg') {
    const parsed = parseSVG(await readText(rest), { size, fit, pad: 0 });
    const base = rest.split(/[\\/]/).at(-1).replace(/\.svg$/i, '');
    const name = slugify(base) || 'scene';
    return {
      spec,
      label: `svg '${base}' (${parsed.elements} element${parsed.elements === 1 ? '' : 's'}, ` +
        `viewBox ${parsed.viewBox.join(' ')})`,
      slug: `svg_${name}`,
      key: name.replace(/_/g, '-'),
      quads: place(parsed.quads),
      evenodd: parsed.evenodd,
      warnings: parsed.warnings,
    };
  }

  if ([...rest].length !== 1) throw new Error(`glyph scene needs exactly one character, got "${rest}"`);
  if (!font) throw new Error('glyph scenes need a font');
  // The codepoint keeps the slug unique.
  const hex = rest.codePointAt(0).toString(16).padStart(4, '0');
  const name = slugify(rest);
  return {
    spec,
    label: `glyph '${rest}' U+${hex.toUpperCase()}`,
    slug: `glyph_${name ? `${name}_` : ''}u${hex}`,
    key: 'glyph',
    quads: place(glyphShape(font, rest, size)),
    evenodd: false,
    warnings: [],
  };
}
