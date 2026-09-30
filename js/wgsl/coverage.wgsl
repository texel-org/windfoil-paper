// Windfoil box coverage and its analytic VJP: monotone-piece root finding,
// the exact box-filtered area integral, and the shape-level winding fold.

// ---------------------------------------------------------------- mono_root

fn mono_root(A : f32, B : f32, a0 : f32, e1 : f32, v : f32, rising : bool) -> f32 {
  let c = a0 - v;
  let disc = max(B * B - 4.0 * A * c, 0.0);
  let guard = 1e-14 * (B * B + abs(4.0 * A * c)) + 1e-30;
  let sq = sqrt(disc + guard);
  let qq = -0.5 * (B + select(-sq, sq, B >= 0.0));
  let r1 = qq / nonzero(A);
  let r2 = c / nonzero(qq);
  let t = clamp(select(r2, r1, (B < 0.0) == rising), 0.0, 1.0);
  let sat0 = select(a0 <= v, a0 >= v, rising);
  let sat1 = select(e1 >= v, e1 <= v, rising);
  return select(select(t, 1.0, sat1), 0.0, sat0);
}

// Returns (dA, dB, da0), including the guard gradient used by JAX.
fn mono_root_grad(A : f32, B : f32, a0 : f32, e1 : f32, v : f32, rising : bool, dt : f32) -> vec3<f32> {
  let c = a0 - v;
  let disc_raw = B * B - 4.0 * A * c;
  let disc = max(disc_raw, 0.0);
  let g4 = 4.0 * A * c;
  let guard = 1e-14 * (B * B + abs(g4)) + 1e-30;
  let sq = sqrt(disc + guard);
  let qq = -0.5 * (B + select(-sq, sq, B >= 0.0));
  let nzA = nonzero(A);
  let nzq = nonzero(qq);
  let r1 = qq / nzA;
  let r2 = c / nzq;
  let pick_r1 = (B < 0.0) == rising;
  let t_raw = select(r2, r1, pick_r1);
  let sat0 = select(a0 <= v, a0 >= v, rising);
  let sat1 = select(e1 >= v, e1 <= v, rising);
  // JAX assigns a 0.5 subgradient at a clamp boundary.
  if (sat0 || sat1) {
    return vec3<f32>(0.0);
  }
  var wt = 1.0;
  if (t_raw < 0.0 || t_raw > 1.0) { wt = 0.0; }
  if (t_raw == 0.0 || t_raw == 1.0) { wt = 0.5; }
  if (wt == 0.0) {
    return vec3<f32>(0.0);
  }
  let dtw = dt * wt;
  var dA = 0.0;
  var dB = 0.0;
  var dc = 0.0;
  var dqq = 0.0;
  if (pick_r1) {
    dqq = dtw / nzA;
    // The nonzero floor is constant in this branch.
    if (abs(A) >= TINY) { dA += -dtw * qq / (nzA * nzA); }
  } else {
    dc += dtw / nzq;
    if (abs(qq) >= TINY) { dqq += -dtw * c / (nzq * nzq); }
  }
  dB += -0.5 * dqq;
  let dsq = -0.5 * dqq * select(-1.0, 1.0, B >= 0.0);
  let dinner = dsq / (2.0 * sq); // sq > 0 by the guard
  // disc = max(disc_raw, 0): full inside, 0.5 at the exact tie
  var wdisc = 0.0;
  if (disc_raw > 0.0) { wdisc = 1.0; }
  if (disc_raw == 0.0) { wdisc = 0.5; }
  if (wdisc != 0.0) {
    dB += wdisc * dinner * 2.0 * B;
    dA += -wdisc * dinner * 4.0 * c;
    dc += -wdisc * dinner * 4.0 * A;
  }
  // guard = 1e-14 * (B*B + |4Ac|) + 1e-30
  dB += dinner * 1e-14 * 2.0 * B;
  let sgn4 = select(-1.0, 1.0, g4 >= 0.0);
  dA += dinner * 1e-14 * sgn4 * 4.0 * c;
  dc += dinner * 1e-14 * sgn4 * 4.0 * A;
  return vec3<f32>(dA, dB, dc); // da0 = dc (v is constant)
}

// ---------------------------------------------------------- integrate_piece

// Box-filtered area for one pixel-relative, xy-monotone piece.
fn integrate_piece(q1 : vec2<f32>, q2 : vec2<f32>, q3 : vec2<f32>, wlo : f32, whi : f32, hx : f32) -> f32 {
  // Fixed splitting can emit point pieces. Only discard an exact f32 point:
  // a relative cutoff can remove valid short SVG edges and break the
  // telescoping cancellation of a closed contour.
  let span = abs(q2.x - q1.x) + abs(q2.y - q1.y) + abs(q3.x - q2.x) + abs(q3.y - q2.y);
  if (span == 0.0) { return 0.0; }

  // Exact telescoping path; >= preserves boundary routing.
  let hull_min = min(q1.x, min(q2.x, q3.x));
  if (hull_min >= hx) {
    return (2.0 * hx) * (clamp(q3.y, wlo, whi) - clamp(q1.y, wlo, whi));
  }

  // Strict comparison preserves the -hx tie subgradient.
  let hull_max = max(q1.x, max(q2.x, q3.x));
  if (hull_max < -hx) { return 0.0; }

  let a2 = q1 - 2.0 * q2 + q3;
  let a1 = 2.0 * (q2 - q1);
  let y_rising = q3.y >= q1.y;
  let t_lo = mono_root(a2.y, a1.y, q1.y, q3.y, select(whi, wlo, y_rising), y_rising);
  let t_hi = mono_root(a2.y, a1.y, q1.y, q3.y, select(wlo, whi, y_rising), y_rising);
  let x_rising = q3.x >= q1.x;
  let t_left = clamp(mono_root(a2.x, a1.x, q1.x, q3.x, -hx, x_rising), t_lo, t_hi);
  let t_right = clamp(mono_root(a2.x, a1.x, q1.x, q3.x, hx, x_rising), t_lo, t_hi);
  let t1 = select(t_right, t_left, x_rising);
  let t2 = max(select(t_left, t_right, x_rising), t1);
  let tm = 0.5 * (t1 + t2);
  let d = 0.5 * (t2 - t1);
  let x_mid = (a2.x * tm + a1.x) * tm + q1.x + hx;
  let dxm = 2.0 * a2.x * tm + a1.x;
  let dym = 2.0 * a2.y * tm + a1.y;
  let inside = 2.0 * d * x_mid * dym + (2.0 / 3.0) * d * d * d * (a2.x * dym + 2.0 * a2.y * dxm);
  let ra = select(t_lo, t2, x_rising);
  let rb = select(t1, t_hi, x_rising);
  let rm = 0.5 * (ra + rb);
  let right = max(rb - ra, 0.0) * (2.0 * a2.y * rm + a1.y) * (2.0 * hx);
  return select(0.0, inside + right, t_hi > t_lo);
}

// d/dx clamp(x, lo, hi) with JAX tie semantics: 1 inside, 0.5 at either bound.
fn clamp_subgrad(x : f32, lo : f32, hi : f32) -> f32 {
  if (x > lo && x < hi) { return 1.0; }
  if (x == lo || x == hi) { return 0.5; }
  return 0.0;
}

// Adjoint of integrate_piece with JAX-compatible branch subgradients.
struct PieceGrad {
  dq1 : vec2<f32>,
  dq2 : vec2<f32>,
  dq3 : vec2<f32>,
};

fn integrate_piece_grad(q1 : vec2<f32>, q2 : vec2<f32>, q3 : vec2<f32>, wlo : f32, whi : f32, hx : f32, dArea : f32) -> PieceGrad {
  var G : PieceGrad;
  G.dq1 = vec2<f32>(0.0);
  G.dq2 = vec2<f32>(0.0);
  G.dq3 = vec2<f32>(0.0);

  let span = abs(q2.x - q1.x) + abs(q2.y - q1.y) + abs(q3.x - q2.x) + abs(q3.y - q2.y);
  if (span == 0.0) { return G; }
  if (dArea == 0.0) { return G; }

  // Fully-right telescoping branch with JAX's clamp tie gradient.
  let hull_min = min(q1.x, min(q2.x, q3.x));
  if (hull_min >= hx) {
    G.dq3.y += dArea * 2.0 * hx * clamp_subgrad(q3.y, wlo, whi);
    G.dq1.y -= dArea * 2.0 * hx * clamp_subgrad(q1.y, wlo, whi);
    return G;
  }

  // Keep exact-boundary pieces on the general path.
  let hull_max = max(q1.x, max(q2.x, q3.x));
  if (hull_max < -hx) { return G; }

  let a2 = q1 - 2.0 * q2 + q3;
  let a1 = 2.0 * (q2 - q1);

  // forward recompute
  let y_rising = q3.y >= q1.y;
  let v_lo = select(whi, wlo, y_rising);
  let v_hi = select(wlo, whi, y_rising);
  let t_lo = mono_root(a2.y, a1.y, q1.y, q3.y, v_lo, y_rising);
  let t_hi = mono_root(a2.y, a1.y, q1.y, q3.y, v_hi, y_rising);
  if (!(t_hi > t_lo)) {
    return G; // area was hard 0
  }
  let x_rising = q3.x >= q1.x;
  let root_l = mono_root(a2.x, a1.x, q1.x, q3.x, -hx, x_rising);
  let root_r = mono_root(a2.x, a1.x, q1.x, q3.x, hx, x_rising);
  let t_left = clamp(root_l, t_lo, t_hi);
  let t_right = clamp(root_r, t_lo, t_hi);
  let tA = select(t_right, t_left, x_rising);  // t1
  let tB = select(t_left, t_right, x_rising);
  let t1 = tA;
  let t2 = max(tB, tA);
  let tm = 0.5 * (t1 + t2);
  let d = 0.5 * (t2 - t1);
  let x_mid = (a2.x * tm + a1.x) * tm + q1.x + hx;
  let vx = 2.0 * a2.x * tm + a1.x;
  let vy = 2.0 * a2.y * tm + a1.y;
  let ra = select(t_lo, t2, x_rising);
  let rb = select(t1, t_hi, x_rising);
  let rm = 0.5 * (ra + rb);

  // ---- adjoint ----
  var da2 = vec2<f32>(0.0);
  var da1 = vec2<f32>(0.0);
  var dt1 = 0.0;
  var dt2 = 0.0;
  var dt_lo = 0.0;
  var dt_hi = 0.0;

  // right = max(rb - ra, 0) * (2 a2.y rm + a1.y) * 2hx
  // max uses JAX's 0.5 subgradient at a tie.
  {
    let w = max(rb - ra, 0.0);
    let ymid = 2.0 * a2.y * rm + a1.y;
    let dymid = dArea * w * 2.0 * hx;
    da2.y += dymid * 2.0 * rm;
    da1.y += dymid;
    let drm = dymid * 2.0 * a2.y;
    var mw = 0.0;
    if (rb - ra > 0.0) { mw = 1.0; }
    if (rb - ra == 0.0) { mw = 0.5; }
    let dw = dArea * ymid * 2.0 * hx * mw;
    let dra = -dw + 0.5 * drm;
    let drb = dw + 0.5 * drm;
    if (x_rising) {
      dt2 += dra;
      dt_hi += drb;
    } else {
      dt_lo += dra;
      dt1 += drb;
    }
  }

  // inside = 2 d x_mid vy + (2/3) d^3 (a2.x vy + 2 a2.y vx)
  {
    let d3 = d * d * d;
    let dd = dArea * (2.0 * x_mid * vy + 2.0 * d * d * (a2.x * vy + 2.0 * a2.y * vx));
    let dxmid = dArea * 2.0 * d * vy;
    let dvy = dArea * (2.0 * d * x_mid + (2.0 / 3.0) * d3 * a2.x);
    let dvx = dArea * (2.0 / 3.0) * d3 * 2.0 * a2.y;
    da2.x += dArea * (2.0 / 3.0) * d3 * vy;
    da2.y += dArea * (2.0 / 3.0) * d3 * 2.0 * vx;
    // x_mid = a2.x tm^2 + a1.x tm + q1.x + hx
    da2.x += dxmid * tm * tm;
    da1.x += dxmid * tm;
    G.dq1.x += dxmid;
    // vx = 2 a2.x tm + a1.x ; vy = 2 a2.y tm + a1.y
    da2.x += dvx * 2.0 * tm;
    da1.x += dvx;
    da2.y += dvy * 2.0 * tm;
    da1.y += dvy;
    let dtm = dxmid * (2.0 * a2.x * tm + a1.x) + dvx * 2.0 * a2.x + dvy * 2.0 * a2.y;
    dt1 += 0.5 * dtm - 0.5 * dd;
    dt2 += 0.5 * dtm + 0.5 * dd;
  }

  // t2 = max(tB, tA); t1 = tA — 0.5/0.5 at the exact tie (JAX max semantics)
  var dtA = dt1;
  var dtB = 0.0;
  if (tB > tA) {
    dtB += dt2;
  } else if (tB == tA) {
    dtB += 0.5 * dt2;
    dtA += 0.5 * dt2;
  } else {
    dtA += dt2;
  }
  var dt_left = select(dtB, dtA, x_rising);
  var dt_right = select(dtA, dtB, x_rising);

  // Split clamp-boundary gradients evenly to match JAX.
  var droot_l = 0.0;
  var droot_r = 0.0;
  if (root_l < t_lo) {
    dt_lo += dt_left;
  } else if (root_l == t_lo) {
    dt_lo += 0.5 * dt_left;
    droot_l += 0.5 * dt_left;
  } else if (root_l < t_hi) {
    droot_l += dt_left;
  } else if (root_l == t_hi) {
    dt_hi += 0.5 * dt_left;
    droot_l += 0.5 * dt_left;
  } else {
    dt_hi += dt_left;
  }
  if (root_r < t_lo) {
    dt_lo += dt_right;
  } else if (root_r == t_lo) {
    dt_lo += 0.5 * dt_right;
    droot_r += 0.5 * dt_right;
  } else if (root_r < t_hi) {
    droot_r += dt_right;
  } else if (root_r == t_hi) {
    dt_hi += 0.5 * dt_right;
    droot_r += 0.5 * dt_right;
  } else {
    dt_hi += dt_right;
  }

  // x roots: mono_root(a2.x, a1.x, q1.x, q3.x, ±hx, x_rising)
  if (droot_l != 0.0) {
    let g = mono_root_grad(a2.x, a1.x, q1.x, q3.x, -hx, x_rising, droot_l);
    da2.x += g.x;
    da1.x += g.y;
    G.dq1.x += g.z;
  }
  if (droot_r != 0.0) {
    let g = mono_root_grad(a2.x, a1.x, q1.x, q3.x, hx, x_rising, droot_r);
    da2.x += g.x;
    da1.x += g.y;
    G.dq1.x += g.z;
  }
  // y-window roots: mono_root(a2.y, a1.y, q1.y, q3.y, v, y_rising)
  if (dt_lo != 0.0) {
    let g = mono_root_grad(a2.y, a1.y, q1.y, q3.y, v_lo, y_rising, dt_lo);
    da2.y += g.x;
    da1.y += g.y;
    G.dq1.y += g.z;
  }
  if (dt_hi != 0.0) {
    let g = mono_root_grad(a2.y, a1.y, q1.y, q3.y, v_hi, y_rising, dt_hi);
    da2.y += g.x;
    da1.y += g.y;
    G.dq1.y += g.z;
  }

  // a2 = q1 - 2 q2 + q3 ; a1 = 2 (q2 - q1)
  G.dq1 += da2 - 2.0 * da1;
  G.dq2 += -2.0 * da2 + 2.0 * da1;
  G.dq3 += da2;
  return G;
}

// -------------------------------------------------------------- shape cover

// The shape's filter size: its own (sx, sy) when set, else the global U.s.
fn shape_filter(si : u32) -> vec2<f32> {
  let f = shapes[si].filt.xy;
  return select(U.s, f, f.x > 0.0);
}

fn shape_winding(si : u32, cx : f32, cy : f32) -> f32 {
  let sh = shapes[si];
  let sf = shape_filter(si);
  let hx = 0.5 * sf.x;
  let hy = 0.5 * sf.y;
  var F = 0.0;
  let start = sh.info.x;
  let count = sh.info.y;
  for (var p = 0u; p < count; p++) {
    let o = 6u * (start + p);
    // Strict y-band culling retains boundary subgradients.
    let q1y = pieces[o + 1] - cy;
    let q3y = pieces[o + 5] - cy;
    if (max(q1y, q3y) < -hy || min(q1y, q3y) > hy) { continue; }
    let q1x = pieces[o] - cx;
    let q2x = pieces[o + 2] - cx;
    let q3x = pieces[o + 4] - cx;
    // Strictly-left pieces are exactly zero.
    if (max(q1x, max(q2x, q3x)) < -hx) { continue; }
    let q1 = vec2<f32>(q1x, q1y);
    let q2 = vec2<f32>(q2x, pieces[o + 3] - cy);
    let q3 = vec2<f32>(q3x, q3y);
    F += integrate_piece(q1, q2, q3, -hy, hy, hx);
  }
  return F / (sf.x * sf.y);
}

// Fold the box-averaged winding into Windfoil's analytic fill approximation.
// Returns (coverage, dCoverage/dWinding); fold cusps use a zero subgradient.
fn coverage_fold(winding : f32, fill_rule : u32) -> vec2<f32> {
  let magnitude = abs(winding);
  if (fill_rule == FILL_EVENODD) {
    let phase = magnitude - 2.0 * floor(0.5 * magnitude);
    let rising = phase < 1.0;
    let coverage = select(2.0 - phase, phase, rising);
    let live = phase != 0.0 && abs(phase - 1.0) >= FOLD_EPS;
    let slope = select(-sign(winding), sign(winding), rising);
    return vec2<f32>(coverage, select(0.0, slope, live));
  }
  let coverage = clamp(magnitude, 0.0, 1.0);
  let live = magnitude < 1.0 - FOLD_EPS;
  return vec2<f32>(coverage, select(0.0, sign(winding), live));
}

fn in_bbox(si : u32, cx : f32, cy : f32) -> bool {
  let sf = shape_filter(si);
  let hx = 0.5 * sf.x;
  let hy = 0.5 * sf.y;
  let b = shapes[si].bbox;
  return cx >= b.x - hx && cx <= b.z + hx && cy >= b.y - hy && cy <= b.w + hy;
}
