// The only names parseColor accepts; anything else must be hex or r,g,b.
const NAMED_COLORS = new Map([
  ['black', [0, 0, 0]],
  ['white', [1, 1, 1]],
  ['red', [1, 0, 0]],
  ['green', [0, 0.5, 0]],
  ['blue', [0, 0, 1]],
  ['gray', [0.5, 0.5, 0.5]],
  ['grey', [0.5, 0.5, 0.5]],
]);

export function parseColor(spec) {
  if (Array.isArray(spec)) return spec.slice(0, 3);
  const text = String(spec).trim().toLowerCase();
  if (NAMED_COLORS.has(text)) return NAMED_COLORS.get(text).slice();
  if (text.startsWith('#')) {
    let hex = text.slice(1);
    if (hex.length === 3) hex = [...hex].map((value) => value + value).join('');
    if (!/^[0-9a-f]{6}$/.test(hex)) throw new Error(`invalid color: ${spec}`);
    return [0, 2, 4].map((at) => parseInt(hex.slice(at, at + 2), 16) / 255);
  }
  const values = text.split(',').map(Number);
  if (values.length === 3 && values.every(Number.isFinite)) {
    return Math.max(...values) > 1 ? values.map((value) => value / 255) : values;
  }
  throw new Error(`invalid color: ${spec}`);
}

export function rgbToHex(rgb) {
  const byte = (value) => Math.max(0, Math.min(255, Math.round(value * 255)))
    .toString(16).padStart(2, '0');
  return `#${byte(rgb[0])}${byte(rgb[1])}${byte(rgb[2])}`;
}
