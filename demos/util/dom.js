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

// 1080p fits a 1920 × 1080 box; numeric sizes cap the longest side.
// Neither setting enlarges the source.
export function imageSize(width, height, size = '1080p') {
  const scale = size === '1080p'
    ? Math.min(1, 1920 / width, 1080 / height)
    : Math.min(1, Number(size) / Math.max(width, height));
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

// Export always fills the 1080p bounds, including for smaller input images.
export function exportSize(width, height) {
  const scale = Math.min(1920 / width, 1080 / height);
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

export function previewSize(width, height, dpr = 1) {
  return { width: Math.max(1, Math.round(width * dpr)), height: Math.max(1, Math.round(height * dpr)) };
}
