export function parseOptSizes(values) {
  return values.map((value) => {
    if (String(value).trim().toLowerCase() === 'max') return 'max';
    const number = Number(value);
    if (!Number.isInteger(number) || number < 1) {
      throw new Error('--opt-size values must be positive integers or max');
    }
    return number;
  });
}

export function resolveOptSize(value, lossKind, source = null) {
  if (lossKind === 'clip') {
    const size = value === 'max' ? 224 : value;
    return { width: size, height: size };
  }
  if (lossKind !== 'l2') throw new Error(`unsupported loss: ${lossKind}`);
  if (!source) throw new Error('--opt-size requires an L2 source image');
  if (value === 'max') return { width: source.width, height: source.height };
  const scale = value / Math.max(source.width, source.height);
  return {
    width: Math.max(1, Math.round(source.width * scale)),
    height: Math.max(1, Math.round(source.height * scale)),
  };
}

export function optSizeLabel({ width, height }) {
  return width === height ? String(width) : `${width}x${height}`;
}
