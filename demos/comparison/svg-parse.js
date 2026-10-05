// Permissive SVG importer for comparison scenes: filled geometry as quadratics.

const NUM = /[+-]?(?:\d*\.\d+|\d+\.?)(?:[eE][+-]?\d+)?/g;

// ── affine transforms: [a, b, c, d, e, f] ⇒ x' = a·x + c·y + e, y' = b·x + d·y + f ────────────────────────
const IDENTITY = [1, 0, 0, 1, 0, 0];
const mul = (m, n) => [
  m[0] * n[0] + m[2] * n[1],
  m[1] * n[0] + m[3] * n[1],
  m[0] * n[2] + m[2] * n[3],
  m[1] * n[2] + m[3] * n[3],
  m[0] * n[4] + m[2] * n[5] + m[4],
  m[1] * n[4] + m[3] * n[5] + m[5],
];
const apply = (m, x, y) => [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];
/** The transform's mean linear scale — used to convert a pixel tolerance into user units. */
const scaleOf = (m) => Math.sqrt(Math.abs(m[0] * m[3] - m[1] * m[2])) || 1;

function parseTransform(str) {
  let m = IDENTITY;
  for (const [, name, argStr] of str.matchAll(/([a-zA-Z]+)\s*\(([^)]*)\)/g)) {
    const a = (argStr.match(NUM) ?? []).map(Number);
    const rad = (d) => (d * Math.PI) / 180;
    switch (name) {
      case 'translate':
        m = mul(m, [1, 0, 0, 1, a[0] ?? 0, a[1] ?? 0]);
        break;
      case 'scale':
        m = mul(m, [a[0] ?? 1, 0, 0, a[1] ?? a[0] ?? 1, 0, 0]);
        break;
      case 'rotate': {
        const c = Math.cos(rad(a[0] ?? 0)), s = Math.sin(rad(a[0] ?? 0));
        const cx = a[1] ?? 0, cy = a[2] ?? 0;
        m = mul(m, [1, 0, 0, 1, cx, cy]);
        m = mul(m, [c, s, -s, c, 0, 0]);
        m = mul(m, [1, 0, 0, 1, -cx, -cy]);
        break;
      }
      case 'matrix':
        if (a.length >= 6) m = mul(m, a.slice(0, 6));
        break;
      case 'skewX':
        m = mul(m, [1, 0, Math.tan(rad(a[0] ?? 0)), 1, 0, 0]);
        break;
      case 'skewY':
        m = mul(m, [1, Math.tan(rad(a[0] ?? 0)), 0, 1, 0, 0]);
        break;
    }
  }
  return m;
}

// ── curve conversion ────────────────────────────────────────────────────────────────────────────────────
// Cubics to quadratics by halving until within tolerance.
const CUBIC_ERR = Math.sqrt(3) / 36;
function cubicToQuads(x0, y0, x1, y1, x2, y2, x3, y3, tol, out, depth = 0) {
  const dx = x3 - 3 * x2 + 3 * x1 - x0, dy = y3 - 3 * y2 + 3 * y1 - y0;
  if (Math.sqrt(dx * dx + dy * dy) * CUBIC_ERR <= tol || depth >= 12) {
    out.push(x0, y0, (3 * x1 + 3 * x2 - x0 - x3) / 4, (3 * y1 + 3 * y2 - y0 - y3) / 4, x3, y3);
    return;
  }
  const m = (a, b) => (a + b) / 2;
  const ax = m(x0, x1), ay = m(y0, y1), bx = m(x1, x2), by = m(y1, y2), cx = m(x2, x3), cy = m(y2, y3);
  const dx1 = m(ax, bx), dy1 = m(ay, by), ex = m(bx, cx), ey = m(by, cy);
  const mx = m(dx1, ex), my = m(dy1, ey);
  cubicToQuads(x0, y0, ax, ay, dx1, dy1, mx, my, tol, out, depth + 1);
  cubicToQuads(mx, my, ex, ey, cx, cy, x3, y3, tol, out, depth + 1);
}

// Elliptical arc to cubics (SVG spec F.6.5), then quads.
function arcToQuads(x0, y0, rx, ry, rot, large, sweep, x, y, tol, out) {
  if (rx === 0 || ry === 0) {
    out.push(x0, y0, (x0 + x) / 2, (y0 + y) / 2, x, y);
    return;
  }
  rx = Math.abs(rx);
  ry = Math.abs(ry);
  const phi = (rot * Math.PI) / 180, cosP = Math.cos(phi), sinP = Math.sin(phi);
  const dx2 = (x0 - x) / 2, dy2 = (y0 - y) / 2;
  const x1p = cosP * dx2 + sinP * dy2, y1p = -sinP * dx2 + cosP * dy2;
  const lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
  if (lam > 1) {
    const s = Math.sqrt(lam);
    rx *= s;
    ry *= s;
  }
  const num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p;
  const den = rx * rx * y1p * y1p + ry * ry * x1p * x1p;
  const co = (large !== sweep ? 1 : -1) * Math.sqrt(Math.max(0, num / den));
  const cxp = co * (rx * y1p) / ry, cyp = co * -(ry * x1p) / rx;
  const cx = cosP * cxp - sinP * cyp + (x0 + x) / 2;
  const cy = sinP * cxp + cosP * cyp + (y0 + y) / 2;
  const ang = (ux, uy, vx, vy) => {
    const d = Math.sqrt((ux * ux + uy * uy) * (vx * vx + vy * vy));
    const c = Math.min(1, Math.max(-1, (ux * vx + uy * vy) / (d || 1)));
    return (ux * vy - uy * vx < 0 ? -1 : 1) * Math.acos(c);
  };
  const t1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry);
  let dt = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry);
  if (!sweep && dt > 0) dt -= 2 * Math.PI;
  if (sweep && dt < 0) dt += 2 * Math.PI;

  // Segment count from the tolerance.
  const rMax = Math.max(rx, ry);
  const arcErr = (th) => (4 / 27) * Math.sin(th / 4) ** 6 / Math.cos(th / 4) ** 2 * rMax;
  let segs = Math.max(1, Math.ceil(Math.abs(dt) / (Math.PI / 2)));
  while (arcErr(Math.abs(dt) / segs) > tol && segs < 4096) segs *= 2;
  const step = dt / segs;
  const k = (4 / 3) * Math.tan(step / 4);
  let a = t1, px = x0, py = y0;
  for (let i = 0; i < segs; i++) {
    const b = a + step;
    const pt = (t) => {
      const ct = Math.cos(t), st = Math.sin(t);
      return [cx + rx * ct * cosP - ry * st * sinP, cy + rx * ct * sinP + ry * st * cosP];
    };
    const der = (t) => {
      const ct = Math.cos(t), st = Math.sin(t);
      return [-rx * st * cosP - ry * ct * sinP, -rx * st * sinP + ry * ct * cosP];
    };
    const [ex, ey] = pt(b), [d1x, d1y] = der(a), [d2x, d2y] = der(b);
    cubicToQuads(px, py, px + k * d1x, py + k * d1y, ex - k * d2x, ey - k * d2y, ex, ey, tol, out);
    px = ex;
    py = ey;
    a = b;
  }
}

// ── path data ───────────────────────────────────────────────────────────────────────────────────────────
// Tokenise; arc flags are single characters.
function tokenizePath(d) {
  const toks = [];
  const re = /([MmLlHhVvCcSsQqTtAaZz])|([+-]?(?:\d*\.\d+|\d+\.?)(?:[eE][+-]?\d+)?)/g;
  let m;
  while ((m = re.exec(d))) toks.push(m[1] ?? Number(m[2]));
  return toks;
}

/** Flat quads for one path `d`, in the path's own user units. */
export function pathToQuads(d, tol) {
  const t = tokenizePath(d);
  const out = [];
  let i = 0, cmd = null;
  let x = 0, y = 0, sx = 0, sy = 0; // current point, subpath start
  let px = 0, py = 0, qx = 0, qy = 0; // previous cubic / quadratic control, for S and T
  let prev = null;
  const line = (X, Y) => {
    out.push(x, y, (x + X) / 2, (y + Y) / 2, X, Y);
    x = X;
    y = Y;
  };
  const num = () => t[i++];
  while (i < t.length) {
    if (typeof t[i] === 'string') cmd = t[i++];
    else if (cmd === 'M') cmd = 'L'; // implicit: extra pairs after M are lineto
    else if (cmd === 'm') cmd = 'l';
    if (cmd === undefined) break;
    const rel = cmd === cmd.toLowerCase();
    const ox = rel ? x : 0, oy = rel ? y : 0;
    switch (cmd.toUpperCase()) {
      case 'M': {
        x = ox + num();
        y = oy + num();
        sx = x;
        sy = y;
        break;
      }
      case 'L':
        line(ox + num(), oy + num());
        break;
      case 'H':
        line(ox + num(), y);
        break;
      case 'V':
        line(x, oy + num());
        break;
      case 'C': {
        const c1x = ox + num(), c1y = oy + num(), c2x = ox + num(), c2y = oy + num();
        const ex = ox + num(), ey = oy + num();
        cubicToQuads(x, y, c1x, c1y, c2x, c2y, ex, ey, tol, out);
        px = c2x;
        py = c2y;
        x = ex;
        y = ey;
        break;
      }
      case 'S': {
        const refl = 'CS'.includes(prev) ? [2 * x - px, 2 * y - py] : [x, y];
        const c2x = ox + num(), c2y = oy + num(), ex = ox + num(), ey = oy + num();
        cubicToQuads(x, y, refl[0], refl[1], c2x, c2y, ex, ey, tol, out);
        px = c2x;
        py = c2y;
        x = ex;
        y = ey;
        break;
      }
      case 'Q': {
        const cx = ox + num(), cy = oy + num(), ex = ox + num(), ey = oy + num();
        out.push(x, y, cx, cy, ex, ey);
        qx = cx;
        qy = cy;
        x = ex;
        y = ey;
        break;
      }
      case 'T': {
        const cx = 'QT'.includes(prev) ? 2 * x - qx : x;
        const cy = 'QT'.includes(prev) ? 2 * y - qy : y;
        const ex = ox + num(), ey = oy + num();
        out.push(x, y, cx, cy, ex, ey);
        qx = cx;
        qy = cy;
        x = ex;
        y = ey;
        break;
      }
      case 'A': {
        const rx = num(), ry = num(), rot = num();
        const large = !!num(), sweep = !!num();
        const ex = ox + num(), ey = oy + num();
        arcToQuads(x, y, rx, ry, rot, large, sweep, ex, ey, tol, out);
        x = ex;
        y = ey;
        break;
      }
      case 'Z':
        if (x !== sx || y !== sy) line(sx, sy);
        x = sx;
        y = sy;
        break;
    }
    prev = cmd.toUpperCase();
  }
  return out;
}

// ── primitive shapes ────────────────────────────────────────────────────────────────────────────────────
const ellipseQuads = (cx, cy, rx, ry, tol) => {
  // Four ≤90° arcs, each adaptively converted — exact-tangent at the axes.
  const out = [];
  const k = (4 / 3) * (Math.SQRT2 - 1);
  const P = [[cx + rx, cy], [cx, cy + ry], [cx - rx, cy], [cx, cy - ry]];
  const C = [
    [cx + rx, cy + k * ry, cx + k * rx, cy + ry],
    [cx - k * rx, cy + ry, cx - rx, cy + k * ry],
    [cx - rx, cy - k * ry, cx - k * rx, cy - ry],
    [cx + k * rx, cy - ry, cx + rx, cy - k * ry],
  ];
  for (let i = 0; i < 4; i++) {
    const [x0, y0] = P[i], [x3, y3] = P[(i + 1) % 4];
    cubicToQuads(x0, y0, C[i][0], C[i][1], C[i][2], C[i][3], x3, y3, tol, out);
  }
  return out;
};

function rectQuads(a, tol) {
  const x = a.x ?? 0, y = a.y ?? 0, w = a.width ?? 0, h = a.height ?? 0;
  if (w <= 0 || h <= 0) return [];
  let rx = a.rx ?? a.ry ?? 0, ry = a.ry ?? a.rx ?? 0;
  rx = Math.min(rx, w / 2);
  ry = Math.min(ry, h / 2);
  if (!(rx > 0 && ry > 0)) {
    return pathToQuads(`M${x} ${y}H${x + w}V${y + h}H${x}Z`, tol);
  }
  return pathToQuads(
    `M${x + rx} ${y}H${x + w - rx}A${rx} ${ry} 0 0 1 ${x + w} ${y + ry}V${y + h - ry}` +
      `A${rx} ${ry} 0 0 1 ${x + w - rx} ${y + h}H${x + rx}A${rx} ${ry} 0 0 1 ${x} ${y + h - ry}` +
      `V${y + ry}A${rx} ${ry} 0 0 1 ${x + rx} ${y}Z`,
    tol,
  );
}

const pointsToQuads = (str, close) => {
  const n = (str.match(NUM) ?? []).map(Number);
  const pts = [];
  for (let i = 0; i + 1 < n.length; i += 2) pts.push([n[i], n[i + 1]]);
  if (pts.length < 2) return [];
  const out = [];
  const seg = (a, b) => out.push(a[0], a[1], (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, b[0], b[1]);
  for (let i = 0; i + 1 < pts.length; i++) seg(pts[i], pts[i + 1]);
  if (close) seg(pts[pts.length - 1], pts[0]);
  return out;
};

// ── the document walk ───────────────────────────────────────────────────────────────────────────────────
const SKIP_SUBTREE = new Set(['defs', 'clippath', 'mask', 'symbol', 'pattern', 'marker', 'title', 'desc',
  'style', 'script', 'metadata', 'filter', 'lineargradient', 'radialgradient']);
const REPORT_UNSUPPORTED = new Set(['use', 'text', 'tspan', 'textpath', 'image', 'foreignobject', 'switch']);

const attrsOf = (tag) => {
  const a = {};
  for (const [, k, v1, v2] of tag.matchAll(/([a-zA-Z_:][\w:.-]*)\s*=\s*(?:"([^"]*)"|'([^']*)')/g)) {
    a[k.toLowerCase()] = v1 ?? v2 ?? '';
  }
  return a;
};
const numAttr = (a, k) => (a[k] === undefined ? undefined : parseFloat(a[k]));

/** Read an SVG's filled geometry, fitted to a size x size render (fit: 'viewbox' | 'ink'). */
export function parseSVG(source, { size, fit = 'viewbox', pad = 0, tol = 1 / 2000 }) {
  const src = source.replace(/<!--[\s\S]*?-->/g, '').replace(/<!\[CDATA\[[\s\S]*?\]\]>/g, '');
  const root = src.match(/<svg\b[^>]*>/i);
  if (!root) throw new Error('not an SVG document (no <svg> element)');
  const rootAttrs = attrsOf(root[0]);

  // The source coordinate box: viewBox if present, else width/height, else fall back to the render size.
  const vb = (rootAttrs.viewbox?.match(NUM) ?? []).map(Number);
  const viewBox = vb.length === 4 ? vb : [0, 0, numAttr(rootAttrs, 'width') || size, numAttr(rootAttrs, 'height') || size];

  const warnings = [];
  const fillRules = { nonzero: 0, evenodd: 0 };
  let elements = 0, strokeOnly = 0;

  // `ink` fit needs a measuring pass first.
  const collect = (base) => {
    const quads = [];
    const stack = [{ m: base, fill: null, rule: null, display: null, depth: 0 }];
    let skipDepth = -1;
    const tagRe = /<([a-zA-Z][\w:-]*)((?:"[^"]*"|'[^']*'|[^>"'])*?)(\/?)>|<\/([a-zA-Z][\w:-]*)\s*>/g;
    let m;
    while ((m = tagRe.exec(src))) {
      const closing = m[4];
      if (closing) {
        const name = closing.toLowerCase();
        if (skipDepth >= 0 && stack.length - 1 <= skipDepth) skipDepth = -1;
        if (name !== 'svg' && stack.length > 1) stack.pop();
        continue;
      }
      const name = m[1].toLowerCase();
      const selfClosing = m[3] === '/' || name === 'svg' && false;
      const a = attrsOf(m[2]);
      const parent = stack[stack.length - 1];

      if (SKIP_SUBTREE.has(name)) {
        if (!selfClosing) {
          stack.push({ ...parent });
          if (skipDepth < 0) skipDepth = stack.length - 1;
        }
        continue;
      }
      if (skipDepth >= 0) {
        if (!selfClosing && name !== 'svg') stack.push({ ...parent });
        continue;
      }

      const ctx = {
        m: a.transform ? mul(parent.m, parseTransform(a.transform)) : parent.m,
        fill: a.fill ?? parent.fill,
        rule: (a['fill-rule'] ?? parent.rule)?.toLowerCase() ?? null,
        display: a.display ?? parent.display,
      };
      if (name === 'svg' || name === 'g' || name === 'a') {
        if (!selfClosing) stack.push(ctx);
        continue;
      }
      if (REPORT_UNSUPPORTED.has(name)) {
        warnings.push(`<${name}> skipped — geometry this importer cannot resolve`);
        if (!selfClosing) stack.push(ctx);
        continue;
      }

      const hidden = ctx.display?.toLowerCase() === 'none';
      const unfilled = (ctx.fill ?? '').trim().toLowerCase() === 'none';
      const localTol = tol / scaleOf(ctx.m);
      let raw = null;
      switch (name) {
        case 'path':
          raw = a.d ? pathToQuads(a.d, localTol) : [];
          break;
        case 'rect':
          raw = rectQuads(
            { x: numAttr(a, 'x'), y: numAttr(a, 'y'), width: numAttr(a, 'width'), height: numAttr(a, 'height'),
              rx: numAttr(a, 'rx'), ry: numAttr(a, 'ry') },
            localTol,
          );
          break;
        case 'circle':
          raw = ellipseQuads(numAttr(a, 'cx') ?? 0, numAttr(a, 'cy') ?? 0, numAttr(a, 'r') ?? 0, numAttr(a, 'r') ?? 0, localTol);
          break;
        case 'ellipse':
          raw = ellipseQuads(numAttr(a, 'cx') ?? 0, numAttr(a, 'cy') ?? 0, numAttr(a, 'rx') ?? 0, numAttr(a, 'ry') ?? 0, localTol);
          break;
        case 'polygon':
          raw = pointsToQuads(a.points ?? '', true);
          break;
        case 'polyline':
          raw = pointsToQuads(a.points ?? '', true); // filled, so it closes like <polygon>
          break;
        case 'line':
          raw = []; // zero-area when filled
          break;
        default:
          if (!selfClosing) stack.push(ctx);
          continue;
      }
      if (!selfClosing) stack.push(ctx);
      if (!raw.length) continue;
      if (hidden) continue;
      if (unfilled) {
        strokeOnly++;
        continue;
      }
      elements++;
      fillRules[ctx.rule === 'evenodd' ? 'evenodd' : 'nonzero']++;
      for (let k = 0; k < raw.length; k += 2) {
        const [X, Y] = apply(ctx.m, raw[k], raw[k + 1]);
        quads.push(X, Y);
      }
    }
    return quads;
  };

  // viewBox → the render box, uniform scale, centred (SVG's default preserveAspectRatio).
  const fitTo = (bx, by, bw, bh, margin) => {
    const box = size - 2 * margin;
    const k = Math.min(box / (bw || 1), box / (bh || 1));
    return [k, 0, 0, k, margin + (box - bw * k) / 2 - bx * k, margin + (box - bh * k) / 2 - by * k];
  };

  let quads = collect(fitTo(viewBox[0], viewBox[1], viewBox[2], viewBox[3], 0));
  if (fit === 'ink' && quads.length) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < quads.length; i += 2) {
      if (quads[i] < x0) x0 = quads[i];
      if (quads[i] > x1) x1 = quads[i];
      if (quads[i + 1] < y0) y0 = quads[i + 1];
      if (quads[i + 1] > y1) y1 = quads[i + 1];
    }
    // Re-walk with the ink fit baked in, so flattening is measured against the final scale.
    elements = 0;
    strokeOnly = 0;
    fillRules.nonzero = fillRules.evenodd = 0;
    const inkFit = mul(fitTo(x0, y0, x1 - x0, y1 - y0, pad), fitTo(viewBox[0], viewBox[1], viewBox[2], viewBox[3], 0));
    quads = collect(inkFit);
  }

  if (!elements) throw new Error('no filled geometry found in the SVG (all elements hidden, fill="none", or unsupported)');
  if (strokeOnly) {
    warnings.push(
      `${strokeOnly} element(s) with fill="none" ignored — this compares FILL coverage, so strokes ` +
        `contribute nothing. Outline them to a filled path if they matter.`,
    );
  }
  if (elements > 1) {
    warnings.push(
      `merged ${elements} filled elements into one shape. SVG paints each separately, so overlapping ` +
        `elements do NOT interact there, but their windings DO here — identical for same-orientation ` +
        `nonzero art, not otherwise.`,
    );
  }
  if (fillRules.nonzero && fillRules.evenodd) {
    warnings.push(
      `mixed fill rules (${fillRules.nonzero} nonzero, ${fillRules.evenodd} even-odd); the renderers take ` +
        `one rule per scene, so the majority wins and this scene is NOT a faithful reproduction.`,
    );
  }
  return {
    quads,
    evenodd: fillRules.evenodd > fillRules.nonzero,
    warnings,
    viewBox,
    elements,
  };
}
