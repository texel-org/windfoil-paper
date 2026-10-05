// Slug (Lengyel, JCGT 2017), dual-ray coverage, ported for the comparison.

struct Uniforms {
  res : vec2<f32>,    // render-target size in pixels
  style : vec2<f32>,  // (gamma, sharp) — unused by Slug; present to share the windfoil uniform layout
  cam : vec4<f32>,    // camera: device px = worldPx·(scaleX, scaleY) + (transX, transY). (1,1,0,0) = identity.
};

struct Instance {
  place : vec4<f32>, // originX, originY (device px of the glyph's origin), unitsToPx, fillRule (0 = nonzero, 1 = even-odd)
  bbox  : vec4<f32>, // ink box loX, loY, hiX, hiY (font units, Y-down)
  color : vec4<f32>, // straight-alpha RGBA
  hband : vec4<f32>, // horizontal bands (filed by y): rowBase, bandCount, y0, invH
  vband : vec4<f32>, // vertical bands (rotated, filed by x): rowBase, bandCount, rotY0 (= −hiX), invW
};

// Bands longer than SORT_MIN are sorted, so the gather can stop early. Must equal BAND_SORT_MIN in slug.js.
const SORT_MIN : u32 = 4u;

@group(0) @binding(0) var<uniform> U : Uniforms;
@group(0) @binding(1) var<storage, read> instances : array<Instance>;
// Three vec2 per quadratic; vertical-band curves are stored rotated as (y, -x).
@group(0) @binding(2) var<storage, read> curves : array<vec2<f32>>;
// Row-band table: a flat [start, count] pair per band, indexing into `curves` (whole-quad units).
@group(0) @binding(3) var<storage, read> rows : array<u32>;

struct VsOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) rc : vec2<f32>,                       // em-space position of this fragment (Y-down)
  @location(1) @interpolate(flat) inst : u32,
};

// One instanced quad over the ink box plus a 0.625px skirt.
@vertex
fn vs(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VsOut {
  let I = instances[ii];
  let unitsToPx = I.place.z;
  let camScale = U.cam.xy;
  let pad = vec2<f32>(0.625) / (unitsToPx * max(abs(camScale), vec2<f32>(1e-6)));
  let lo = I.bbox.xy - pad;
  let hi = I.bbox.zw + pad;
  let uv = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
  let em = mix(lo, hi, uv);
  let worldPx = I.place.xy + em * unitsToPx;
  let devicePx = worldPx * camScale + U.cam.zw;
  let clip = vec2<f32>(devicePx.x / U.res.x * 2.0 - 1.0, 1.0 - devicePx.y / U.res.y * 2.0);
  var o : VsOut;
  o.pos = vec4<f32>(clip, 0.0, 1.0);
  o.rc = em;
  o.inst = ii;
  return o;
}

// Root eligibility from the control points' y-signs (reference CalcRootCode).
fn calc_root_code(y1 : f32, y2 : f32, y3 : f32) -> u32 {
  let i1 = bitcast<u32>(y1) >> 31u;
  let i2 = bitcast<u32>(y2) >> 30u;
  let i3 = bitcast<u32>(y3) >> 29u;
  var shift = (i2 & 2u) | (i1 & ~2u);
  shift = (i3 & 4u) | (shift & ~4u);
  return (0x2E74u >> shift) & 0x0101u;
}

// One quadratic's (signed coverage, weight) for a +x ray at y = 0.
fn curve_cover(p1 : vec2<f32>, p2 : vec2<f32>, p3 : vec2<f32>, invDiam : f32) -> vec2<f32> {
  let code = calc_root_code(p1.y, p2.y, p3.y);
  if (code == 0u) { return vec2<f32>(0.0, 0.0); }

  let a = p1 - 2.0 * p2 + p3;
  let b = p1 - p2;
  let c = p1;
  let d = sqrt(max(b.y * b.y - a.y * c.y, 0.0));
  var t1 : f32;
  var t2 : f32;
  if (abs(a.y) < 1e-4 * (abs(b.y) + 1e-6)) {
    let t = c.y / (2.0 * b.y); // near-linear: double root at the linear solution
    t1 = t; t2 = t;
  } else if (d == 0.0) {
    // Grazing curve: equal roots, so the ramps cancel.
    let t = b.y / a.y;
    t1 = t; t2 = t;
  } else {
    let q = b.y + select(-d, d, b.y >= 0.0);
    t1 = select(q / a.y, c.y / q, b.y >= 0.0); // (b.y − d)/a.y
    t2 = select(c.y / q, q / a.y, b.y >= 0.0); // (b.y + d)/a.y
  }

  var cov : f32 = 0.0;
  var wgt : f32 = 0.0;
  if ((code & 1u) != 0u) {
    let r = ((a.x * t1 - 2.0 * b.x) * t1 + c.x) * invDiam;
    cov += clamp(r + 0.5, 0.0, 1.0);
    wgt = max(wgt, clamp(1.0 - abs(r) * 2.0, 0.0, 1.0));
  }
  if (code > 1u) {
    let r = ((a.x * t2 - 2.0 * b.x) * t2 + c.x) * invDiam;
    cov -= clamp(r + 0.5, 0.0, 1.0);
    wgt = max(wgt, clamp(1.0 - abs(r) * 2.0, 0.0, 1.0));
  }
  return vec2<f32>(cov, wgt);
}

// One ray's (coverage sum, max weight) over its band.
fn gather_ray(rc : vec2<f32>, band : vec4<f32>, invDiam : f32, half : f32) -> vec2<f32> {
  let rowBase = u32(band.x);
  let R = u32(band.y);
  let y0 = band.z;
  let invH = band.w;

  var bi : u32 = 0u;
  if (invH > 0.0 && R > 1u) {
    bi = u32(clamp(floor((rc.y - y0) * invH), 0.0, f32(R) - 1.0));
  }
  let rIdx = (rowBase + bi) * 2u;
  let start = rows[rIdx];
  let count = rows[rIdx + 1u];

  var cov : f32 = 0.0;
  var wgt : f32 = 0.0;
  let sorted = count > SORT_MIN;
  for (var i : u32 = 0u; i < count; i = i + 1u) {
    let base = (start + i) * 3u;
    let p1 = curves[base] - rc;
    let p2 = curves[base + 1u] - rc;
    let p3 = curves[base + 2u] - rc;
    // Fully behind the ray's near clamp edge → 0 coverage, 0 weight. Sorted bands can stop.
    if (max(p1.x, max(p2.x, p3.x)) <= -half) {
      if (sorted) { break; }
      continue;
    }
    let cw = curve_cover(p1, p2, p3, invDiam);
    cov += cw.x;
    wgt = max(wgt, cw.y);
  }
  return vec2<f32>(cov, wgt);
}

// Straight-alpha color × coverage → premultiplied RGBA (the pipeline blends premultiplied-over).
fn shade(color : vec4<f32>, cov : f32) -> vec4<f32> {
  let a = color.a * cov;
  return vec4<f32>(color.rgb * a, a);
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4<f32> {
  let I = instances[in.inst];
  let rc = in.rc;
  // units_per_pixel per axis — fwidth, exactly the reference SlugPixelShader's emsPerPixel measure.
  let s = max(fwidth(rc), vec2<f32>(1e-9));

  // Dual ray → dual (coverage, weight): horizontal in the plain frame, vertical in the rotated frame.
  let xc = gather_ray(rc, I.hband, 1.0 / s.x, s.x * 0.5);                     // (xcov, xwgt)
  let yc = gather_ray(vec2<f32>(rc.y, -rc.x), I.vband, 1.0 / s.y, s.y * 0.5); // (ycov, ywgt)

  // Reference CalcCoverage: weight each ray by its reliability.
  var cov = max(
    abs(xc.x * xc.y + yc.x * yc.y) / max(xc.y + yc.y, 1.0 / 65536.0),
    min(abs(xc.x), abs(yc.x)),
  );
  if (I.place.w > 0.5) {
    cov = 1.0 - abs(1.0 - fract(cov * 0.5) * 2.0); // even-odd
  } else {
    cov = clamp(cov, 0.0, 1.0);                    // nonzero
  }
  return shade(I.color, cov);
}
