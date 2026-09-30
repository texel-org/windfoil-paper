const NUMBER = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;
const NAME_START = /[A-Za-z_:]/;
const NAME_PART = /[A-Za-z0-9_.:-]/;
const SPACE = /[\t\n\r ]/;
const MARKUP = /[<>&\u0000-\u0008\u000b\u000c\u000e-\u001f]/;
const DATA_TEXT_FORBIDDEN = /[<\u0000-\u0008\u000b\u000c\u000e-\u001f]/;
const DATA_ATTRIBUTE = /^data-[A-Za-z0-9_.-]+$/;

function invalid(message, offset) {
  const suffix = offset == null ? '' : ` at byte ${offset}`;
  throw new Error(`Invalid SVG: ${message}${suffix}`);
}

class XmlReader {
  constructor(text) {
    if (typeof text !== 'string') throw new TypeError('SVG source must be a string');
    if (text.includes('\0')) invalid('NUL bytes are not allowed');
    this.text = text;
    this.at = 0;
  }

  whitespace() {
    const start = this.at;
    while (SPACE.test(this.text[this.at] ?? '')) this.at++;
    return this.at - start;
  }

  name() {
    const start = this.at;
    if (!NAME_START.test(this.text[this.at] ?? '')) invalid('expected a tag or attribute name', this.at);
    this.at++;
    while (NAME_PART.test(this.text[this.at] ?? '')) this.at++;
    return this.text.slice(start, this.at);
  }

  tag() {
    if (this.text[this.at] !== '<') invalid('expected a tag', this.at);
    const start = this.at++;
    const closing = this.text[this.at] === '/';
    if (closing) this.at++;
    const name = this.name();
    const attributes = new Map();

    for (;;) {
      const spaced = this.whitespace();
      if (closing) {
        if (this.text[this.at] !== '>') invalid('closing tags cannot have attributes', this.at);
        this.at++;
        return { name, attributes, closing: true, selfClosing: false, start };
      }
      if (this.text.startsWith('/>', this.at)) {
        this.at += 2;
        return { name, attributes, closing: false, selfClosing: true, start };
      }
      if (this.text[this.at] === '>') {
        this.at++;
        return { name, attributes, closing: false, selfClosing: false, start };
      }
      if (!spaced) invalid('attributes must be separated by whitespace', this.at);

      const attribute = this.name();
      if (attributes.has(attribute)) invalid(`duplicate ${attribute} attribute`, this.at);
      this.whitespace();
      if (this.text[this.at++] !== '=') invalid(`expected = after ${attribute}`, this.at - 1);
      this.whitespace();
      const quote = this.text[this.at++];
      if (quote !== '"' && quote !== "'") invalid(`${attribute} must be quoted`, this.at - 1);
      const valueStart = this.at;
      while (this.at < this.text.length && this.text[this.at] !== quote) this.at++;
      if (this.at === this.text.length) invalid(`unterminated ${attribute} attribute`, valueStart);
      const value = this.text.slice(valueStart, this.at++);
      // data-* values are ignored metadata, so they may hold any well-formed
      // XML text; attributes the renderer reads stay free of markup.
      const forbidden = DATA_ATTRIBUTE.test(attribute) ? DATA_TEXT_FORBIDDEN : MARKUP;
      if (forbidden.test(value) || /&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)/.test(value)) {
        invalid(`invalid character in ${attribute}`, valueStart);
      }
      attributes.set(attribute, value);
    }
  }
}

// Custom data-* attributes carry generator metadata with no rendering effect,
// so they are accepted on every element and dropped.
function attributes(tag, names, required = names) {
  const allowed = new Set(names);
  const kept = [...tag.attributes].filter(([name]) => !DATA_ATTRIBUTE.test(name));
  for (const [name] of kept) {
    if (!allowed.has(name)) invalid(`<${tag.name}> does not support ${name}`, tag.start);
  }
  for (const name of required) {
    if (!tag.attributes.has(name)) invalid(`<${tag.name}> requires ${name}`, tag.start);
  }
  return Object.fromEntries(kept);
}

function scalar(source, label) {
  if (!NUMBER.test(source)) invalid(`${label} is not a plain finite number`);
  const value = Number(source);
  if (!Number.isFinite(value)) invalid(`${label} is not finite`);
  return value;
}

function physicalLength(source, label) {
  const match = /^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(cm|mm|Q|in|pc|pt)$/.exec(source);
  if (!match) invalid(`${label} must be a plain number or physical length`);
  const value = Number(match[1]);
  if (!(value > 0) || !Number.isFinite(value)) invalid(`${label} must be positive and finite`);
  return { value, unit: match[2] };
}

function numberList(source, count, label) {
  if (source !== source.trim()) invalid(`${label} has surrounding whitespace`);
  const parts = source.split(/[\t\n\r ]+/);
  if (parts.length !== count) invalid(`${label} must contain ${count} numbers`);
  return parts.map((part, index) => scalar(part, `${label}[${index}]`));
}

function color(source, label) {
  if (/^#[0-9a-fA-F]{6}$/.test(source)) {
    return [1, 3, 5].map((offset) => Number.parseInt(source.slice(offset, offset + 2), 16) / 255);
  }
  const rgb = /^rgb\( *(\d{1,3}) *, *(\d{1,3}) *, *(\d{1,3}) *\)$/.exec(source);
  if (rgb && rgb.slice(1).every((channel) => Number(channel) <= 255)) {
    return rgb.slice(1).map((channel) => Number(channel) / 255);
  }
  invalid(`${label} must be #rrggbb or rgb(r,g,b) with integer channels`);
}

function same(a, b) {
  return Object.is(a, b) || a === b;
}

function pathTokens(source) {
  const tokens = [];
  let at = 0;
  while (at < source.length) {
    const separator = /^[\t\n\r ,]+/.exec(source.slice(at));
    if (separator) at += separator[0].length;
    if (at === source.length) break;
    const command = /^[MmLlHhVvQqZz]/.exec(source.slice(at));
    if (command) {
      tokens.push(command[0]);
      at += 1;
      continue;
    }
    const number = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/.exec(source.slice(at));
    if (!number) invalid('unsupported or malformed path data', at);
    const value = Number(number[0]);
    if (!Number.isFinite(value)) invalid('path coordinate is not finite', at);
    tokens.push(value);
    at += number[0].length;
  }
  return tokens;
}

function parsePath(source) {
  const tokens = pathTokens(source);
  let at = 0;
  let command = null;
  let x = 0;
  let y = 0;
  let startX = 0;
  let startY = 0;
  let open = false;
  let contourCurveStart = 0;
  const curves = [];
  const contours = [];

  const number = (label) => {
    const value = tokens[at++];
    if (typeof value !== 'number') invalid(`path ${label} requires a number`);
    return value;
  };
  const lineTo = (endX, endY) => {
    curves.push(x, y, 0.5 * (x + endX), 0.5 * (y + endY), endX, endY);
    x = endX;
    y = endY;
  };
  const close = () => {
    if (!open) invalid('path Z has no open subpath');
    if (!same(x, startX) || !same(y, startY)) lineTo(startX, startY);
    const count = curves.length / 6 - contourCurveStart;
    if (count < 1) invalid('path subpaths must contain at least one segment');
    contours.push(count);
    open = false;
    command = null;
  };

  while (at < tokens.length) {
    if (typeof tokens[at] === 'string') command = tokens[at++];
    if (!command) invalid('path data must start with M');
    const relative = command === command.toLowerCase();
    const upper = command.toUpperCase();

    if (upper === 'Z') {
      close();
      continue;
    }
    if (upper === 'M') {
      if (open) invalid('each path subpath must be closed with Z');
      const nx = number('M x');
      const ny = number('M y');
      x = relative ? x + nx : nx;
      y = relative ? y + ny : ny;
      startX = x;
      startY = y;
      contourCurveStart = curves.length / 6;
      open = true;
      command = relative ? 'l' : 'L';
      continue;
    }
    if (!open) invalid(`path ${command} is outside a subpath`);
    if (upper === 'L') {
      const nx = number('L x');
      const ny = number('L y');
      lineTo(relative ? x + nx : nx, relative ? y + ny : ny);
    } else if (upper === 'H') {
      const nx = number('H x');
      lineTo(relative ? x + nx : nx, y);
    } else if (upper === 'V') {
      const ny = number('V y');
      lineTo(x, relative ? y + ny : ny);
    } else if (upper === 'Q') {
      const dcx = number('Q cx');
      const dcy = number('Q cy');
      const dex = number('Q x');
      const dey = number('Q y');
      const cx = relative ? x + dcx : dcx;
      const cy = relative ? y + dcy : dcy;
      const endX = relative ? x + dex : dex;
      const endY = relative ? y + dey : dey;
      curves.push(x, y, cx, cy, endX, endY);
      x = endX;
      y = endY;
    } else {
      invalid(`unsupported path command ${command}`);
    }
  }

  if (open) invalid('each path subpath must end with Z');
  if (curves.length === 0) invalid('path must contain at least one segment');
  return { curves: Float64Array.from(curves), contours };
}

function parseRoot(tag) {
  const attrs = attributes(tag, ['xmlns', 'viewBox', 'width', 'height']);
  if (attrs.xmlns !== 'http://www.w3.org/2000/svg') invalid('unsupported SVG namespace', tag.start);
  const [x, y, width, height] = numberList(attrs.viewBox, 4, 'viewBox');
  if (!(width > 0) || !(height > 0)) {
    invalid('SVG dimensions must be positive', tag.start);
  }

  if (NUMBER.test(attrs.width) && NUMBER.test(attrs.height)) {
    const rootWidth = scalar(attrs.width, 'width');
    const rootHeight = scalar(attrs.height, 'height');
    if (!(rootWidth > 0) || !(rootHeight > 0) || !same(rootWidth, width) || !same(rootHeight, height)) {
      invalid('unitless width and height must match the viewBox', tag.start);
    }
  } else {
    const rootWidth = physicalLength(attrs.width, 'width');
    const rootHeight = physicalLength(attrs.height, 'height');
    if (rootWidth.unit !== rootHeight.unit) invalid('width and height must use the same unit', tag.start);
    const rootRatio = rootWidth.value / rootHeight.value;
    const viewBoxRatio = width / height;
    if (Math.abs(rootRatio - viewBoxRatio) > 1e-9 * Math.max(rootRatio, viewBoxRatio)) {
      invalid('physical width and height must match the viewBox aspect ratio', tag.start);
    }
  }
  return { x, y, width, height };
}

function parseRect(tag, viewBox) {
  const attrs = attributes(tag, ['x', 'y', 'width', 'height', 'fill'], ['width', 'height', 'fill']);
  const x = scalar(attrs.x ?? '0', 'rect x');
  const y = scalar(attrs.y ?? '0', 'rect y');
  const width = scalar(attrs.width, 'rect width');
  const height = scalar(attrs.height, 'rect height');
  if (!same(viewBox.x, x) || !same(viewBox.y, y) || !same(width, viewBox.width) ||
    !same(height, viewBox.height)) {
    invalid('background rect must cover the full viewBox', tag.start);
  }
  return color(attrs.fill, 'rect fill');
}

function parseShape(tag) {
  const attrs = attributes(
    tag,
    ['d', 'fill', 'fill-opacity', 'fill-rule'],
    ['d', 'fill'],
  );
  const fillRule = attrs['fill-rule'] ?? 'nonzero';
  if (fillRule !== 'nonzero' && fillRule !== 'evenodd') {
    invalid('fill-rule must be nonzero or evenodd', tag.start);
  }
  const alpha = scalar(attrs['fill-opacity'] ?? '1', 'fill-opacity');
  if (alpha < 0 || alpha > 1) invalid('fill-opacity must be between 0 and 1', tag.start);
  const path = parsePath(attrs.d);
  return {
    curves: path.curves,
    contours: path.contours,
    color: color(attrs.fill, 'path fill'),
    alpha,
    fillRule,
  };
}

export function parseFillSvg(text) {
  const reader = new XmlReader(text);
  reader.whitespace();
  const root = reader.tag();
  if (root.closing || root.selfClosing || root.name !== 'svg') invalid('expected an opening <svg>', root.start);
  const viewBox = parseRoot(root);
  const shapes = [];
  let background = null;
  let sawPath = false;

  for (;;) {
    reader.whitespace();
    if (reader.at === reader.text.length) invalid('missing </svg>', reader.at);
    const tag = reader.tag();
    if (tag.closing) {
      if (tag.name !== 'svg') invalid(`unexpected closing </${tag.name}>`, tag.start);
      break;
    }
    if (!tag.selfClosing) invalid(`<${tag.name}> must be self-closing`, tag.start);
    if (tag.name === 'rect') {
      if (background || sawPath) invalid('the background rect must be the first and only rect', tag.start);
      background = parseRect(tag, viewBox);
    } else if (tag.name === 'path') {
      sawPath = true;
      shapes.push(parseShape(tag));
    } else {
      invalid(`unsupported <${tag.name}> element`, tag.start);
    }
  }

  reader.whitespace();
  if (reader.at !== reader.text.length) invalid('content after </svg>', reader.at);
  return { viewBox, background, shapes };
}

export function rasterSize({ x, y, width, height }, dimension) {
  const ratio = dimension / Math.max(width, height);
  const outputWidth = Math.max(1, Math.round(width * ratio));
  const outputHeight = Math.max(1, Math.round(height * ratio));
  return rasterDimensions({ x, y, width, height }, outputWidth, outputHeight);
}

export const SCALE_MODES = ['expand', 'letterbox'];

// Fit the viewBox inside an exact raster without distortion. `content` is the
// viewBox rect in output pixels. In `expand` mode the spare area simply shows
// more of the scene around the viewBox, centered exactly. In `letterbox` mode
// the spare area is later painted over, so the offset is snapped to whole
// pixels: bars stay crisp and the content pixels match an unpadded render.
export function rasterDimensions({ x, y, width, height }, outputWidth, outputHeight, { mode = 'expand' } = {}) {
  if (!Number.isInteger(outputWidth) || outputWidth < 1 ||
      !Number.isInteger(outputHeight) || outputHeight < 1) {
    throw new Error('raster width and height must be positive integers');
  }
  if (!SCALE_MODES.includes(mode)) throw new Error(`scale mode must be one of ${SCALE_MODES.join(', ')}`);
  const scale = Math.max(width / outputWidth, height / outputHeight);
  const contentWidth = snapInteger(width / scale);
  const contentHeight = snapInteger(height / scale);
  const pad = (outer, inner) => {
    const exact = (outer - inner) / 2;
    return mode === 'letterbox' ? Math.max(0, Math.floor(exact)) : exact;
  };
  const left = pad(outputWidth, contentWidth);
  const top = pad(outputHeight, contentHeight);
  return {
    width: outputWidth,
    height: outputHeight,
    scale,
    origin: [x - left * scale, y - top * scale],
    content: { x: left, y: top, width: contentWidth, height: contentHeight },
  };
}

function snapInteger(value) {
  const rounded = Math.round(value);
  return Math.abs(value - rounded) < 1e-6 ? rounded : value;
}
