// Native filter parameters: exponential above a floor, or sigmoid within bounds.
export function blurParameter(width, floor, ceiling = null) {
  if (ceiling === null) return Math.log(Math.max(width - floor, 1e-3));
  if (!Number.isFinite(ceiling) || ceiling <= floor) throw new Error('blur ceiling must exceed its floor');
  const fraction = Math.max(1e-3, Math.min(1 - 1e-3, (width - floor) / (ceiling - floor)));
  return Math.log(fraction / (1 - fraction));
}

export function blurWidth(parameter, floor, ceiling = null) {
  if (ceiling === null) return floor + Math.exp(parameter);
  const e = Math.exp(parameter >= 0 ? -parameter : parameter);
  const fraction = parameter >= 0 ? 1 / (1 + e) : e / (1 + e);
  return floor + (ceiling - floor) * fraction;
}

export function blurSlope(width, floor, ceiling = null) {
  const slope = width - floor;
  return ceiling === null ? slope : slope * (ceiling - width) / (ceiling - floor);
}
