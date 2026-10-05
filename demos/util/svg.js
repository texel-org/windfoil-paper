import { rgbToHex } from './color.js';

const number = (value) => Number(value.toFixed(3));
const cm = (value) => Number(value.toFixed(4));

// Pen-plotter markers emitted in centimeter units, in creation (paint) order so
// the SVG occludes exactly like the optimized render — overlapping opaque marks
// depend on draw order, and grouping by pen color would resurface marks that the
// fit had covered. Line markers become round-capped straight strokes of the pen
// width; point markers become filled discs of the pen radius. Each mark carries
// its own color (and opacity, for translucent fits).
export function plotMarkersToSVG(markers, {
  widthCm,
  heightCm,
  penWidthCm,
  background = '#ffffff',
}) {
  const lines = [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${cm(widthCm)} ${cm(heightCm)}"` +
      ` width="${cm(widthCm)}cm" height="${cm(heightCm)}cm">`,
  ];
  if (background) {
    lines.push(`  <rect width="${cm(widthCm)}" height="${cm(heightCm)}" fill="${background}"/>`);
  }
  const radius = cm(0.5 * penWidthCm);
  // Per-mark opacity only when a mark is actually translucent (opaque stays bare).
  const fade = (m) =>
    m.opacity != null && m.opacity < 0.999 ? ` opacity="${cm(m.opacity)}"` : '';
  const point = markers.length > 0 && markers[0].mode === 'point';
  lines.push(
    point
      ? '  <g>'
      : `  <g fill="none" stroke-width="${cm(penWidthCm)}" stroke-linecap="round">`,
  );
  for (const m of markers) {
    const color = rgbToHex(m.color);
    lines.push(
      point
        ? `    <circle cx="${cm(m.x)}" cy="${cm(m.y)}" r="${radius}" fill="${color}"${fade(m)}/>`
        : `    <line x1="${cm(m.x0)}" y1="${cm(m.y0)}" x2="${cm(m.x1)}" y2="${cm(m.y1)}" stroke="${color}"${fade(m)}/>`,
    );
  }
  lines.push('  </g>', '</svg>');
  return lines.join('\n') + '\n';
}

export function sceneToSVG(shapes, width, height, { background = '#ffffff', viewBox = null } = {}) {
  viewBox ??= { x: 0, y: 0, width, height };
  const lines = [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}" width="${width}" height="${height}">`,
  ];
  if (background) {
    const origin = viewBox.x === 0 && viewBox.y === 0 ? '' : ` x="${viewBox.x}" y="${viewBox.y}"`;
    lines.push(
      `  <rect${origin} width="${viewBox.width}" height="${viewBox.height}" fill="${background}"/>`,
    );
  }
  for (const shape of shapes) {
    const curves = shape.curves;
    const curveCount = curves.length / 6;
    const contours = shape.contours ?? [curveCount];
    if (!Array.isArray(contours) || contours.some((count) => !Number.isInteger(count) || count < 1) ||
        contours.reduce((sum, count) => sum + count, 0) !== curveCount) {
      throw new Error('shape contours must partition its curves');
    }
    const paths = [];
    let curve = 0;
    for (const count of contours) {
      let path = `M ${number(curves[6 * curve])} ${number(curves[6 * curve + 1])}`;
      for (let i = 0; i < count; i++, curve++) {
        path += ` Q ${number(curves[6 * curve + 2])} ${number(curves[6 * curve + 3])}` +
          ` ${number(curves[6 * curve + 4])} ${number(curves[6 * curve + 5])}`;
      }
      paths.push(`${path} Z`);
    }
    const fillRule = shape.fillRule ?? 'nonzero';
    if (fillRule !== 'nonzero' && fillRule !== 'evenodd') {
      throw new Error(`fillRule must be "nonzero" or "evenodd"`);
    }
    lines.push(
      `  <path d="${paths.join(' ')}" fill="${rgbToHex(shape.color)}" fill-opacity="${shape.alpha.toFixed(4)}" fill-rule="${fillRule}"/>`,
    );
  }
  lines.push('</svg>');
  return lines.join('\n') + '\n';
}
