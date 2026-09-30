// Small browser helpers for the web demo.

export const $ = (id) => document.getElementById(id);

export function boundedInteger(value, fallback, min = 1, max = Infinity) {
  const number = Number(value);
  return Number.isInteger(number) && number >= min ? Math.min(number, max) : fallback;
}

export function integerParam(params, name, fallback, max = Infinity, min = 1) {
  const value = params.get(name);
  return boundedInteger(value == null ? fallback : value, fallback, min, max);
}

// Cover-fit box centering source WxH inside a square of the given side.
export function cover(width, height, side) {
  const scale = Math.max(side / width, side / height);
  const w = width * scale, h = height * scale;
  return { x: (side - w) / 2, y: (side - h) / 2, w, h };
}
