// Forward composite, fused L2 loss, analytic backward, and the gradient
// reductions that contract piece gradients back to curve controls.

// ------------------------------------------------------------------ forward
//
// Loss plug-in contract: `composite` and `backward` know nothing about any loss.
// A loss reads `image` and writes `dLdImage`; `backward` consumes only those
// two. So a new loss (a WebGPU CLIP, or anything else) is a separate kernel
// dispatched between `forward` and `backward` -- no change here, no ubershader.
// `forward_l2` below is only an L2 fast path that skips the image round-trip;
// the generic `forward` + `l2grad` pair stays as the reference two-pass form.

// Scene-wide blend mode, specialized per pipeline so the dead modes cost
// nothing. Codes MUST match BLEND_MODES in renderer.js.
//
// add, multiply, and screen are order-independent: the host skips bin_sort,
// so tile lists arrive in bin_fill's arbitrary order and nothing here may
// depend on position in the list -- no transmittance state, no data-dependent
// early exit. add composites out = bg + sum(color * a); multiply composites
// out = bg * prod(1 - a * (1 - color)); screen is multiply in complement
// space, out = 1 - (1 - bg) * prod(1 - a * color) -- additive light that
// saturates at 1. Layer alpha a = coverage * shape alpha in every mode.
//
// A new mode is a code here, a row in renderer.js's BLEND_MODES, an
// accumulate + finalize branch in composite(), and an adjoint branch in
// backward(). Planned candidates (weighted OIT, log-sum-exp) are also
// reductions of per-layer terms, so they follow the same skeleton -- a
// backward that needs the finished reduction recomputes it the way multiply
// and screen recompute their products below, keeping backward free of the
// forward image.
override BLEND_MODE : u32 = 0u;
override OUTPUT_ALPHA : bool = false;
const BLEND_SRC_OVER : u32 = 0u;
const BLEND_ADD : u32 = 1u;
const BLEND_MULTIPLY : u32 = 2u;
const BLEND_SCREEN : u32 = 3u;

fn composite(gid : vec3<u32>) -> vec4<f32> {
  let cx = U.origin.x + (f32(gid.x) + 0.5) * U.scale;
  let cy = U.origin.y + (f32(gid.y) + 0.5) * U.scale;
  var img = vec3<f32>(0.0);
  var P = 1.0;
  var M = vec3<f32>(1.0);
  let ti = pixel_tile(gid);
  let lo = tileOffset[ti];
  for (var k = tileOffset[ti + 1u]; k > lo; k--) {
    // Match backward's transmittance cutoff; omitted contribution is <=1e-5.
    if (BLEND_MODE == BLEND_SRC_OVER && P < 1e-5) { break; }
    let si = tileShapes[k - 1u];
    if (!in_bbox(si, cx, cy)) { continue; }
    let sh = shapes[si];
    let fold = coverage_fold(shape_winding(si, cx, cy), sh.info.z);
    let cov = fold.x;
    let color = sh.color;
    let a = cov * color.w;
    if (BLEND_MODE == BLEND_ADD) {
      img += color.rgb * a;
    } else if (BLEND_MODE == BLEND_MULTIPLY) {
      M *= 1.0 - a * (1.0 - color.rgb);
    } else if (BLEND_MODE == BLEND_SCREEN) {
      M *= 1.0 - a * color.rgb;
    } else {
      img += color.rgb * (a * P);
      P *= 1.0 - a;
    }
  }
  if (BLEND_MODE == BLEND_ADD) { return vec4<f32>(U.bg.rgb + img, 1.0); }
  if (BLEND_MODE == BLEND_MULTIPLY) { return vec4<f32>(U.bg.rgb * M, 1.0); }
  if (BLEND_MODE == BLEND_SCREEN) { return vec4<f32>(1.0 - (1.0 - U.bg.rgb) * M, 1.0); }
  return vec4<f32>(img + U.bg.rgb * P, select(1.0, 1.0 - P, OUTPUT_ALPHA));
}

fn store_image(idx : u32, img : vec4<f32>) {
  image[idx] = img.r;
  image[idx + 1u] = img.g;
  image[idx + 2u] = img.b;
  image[idx + 3u] = img.a;
}

@compute @workgroup_size(8, 8)
fn forward(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= U.size.x || gid.y >= U.size.y) { return; }
  store_image(4u * (gid.y * U.size.x + gid.x), composite(gid));
}

// ------------------------------------------------------------- L2 GPU loss
// Fused RGB mean-squared loss and image cotangent; alpha is ignored.
var<workgroup> lossPartial : array<f32, 64>;
var<workgroup> kPartial : array<f32, 64>; // referenced only when TONEMAP != 0
var<workgroup> wPartial : array<f32, 64>; // referenced only by white-point modes

// A 2^30 fixed-point sum is deterministic and cannot overflow for loss in [0,1].
const LOSS_SCALE : f32 = 1073741824.0; // 2^30

// The tonemap lives at the loss boundary: the composite and its backward
// stay linear (the image buffer holds unbounded linear light), and only the
// comparison happens in display space, with the chain-rule factor folded
// into dLdImage. TONEMAP is a pipeline constant, so the default path
// compiles exactly as before. The operators (matching js/tonemap.js):
//   1 reinhard        T(x) = k*x / (1 + k*x)
//   2 reinhard-white  T(x) = R(k*x) / R(k*W), R(u) = u/(1+u)
//   3 smooth          T(x) = s(k*x) / s(k*W), s(u) = u*inverseSqrt(1+u*u)
// dL/dk accumulates into lossAccum[1]; the white-point modes also accumulate
// dL/dW into lossAccum[2]. The Reinhard pair has a pole at x = -1/k, so the
// host refuses signed color codecs under it; smooth is odd and C-infinity.
override TONEMAP : u32 = 0u;
const TONEMAP_REINHARD : u32 = 1u;
const TONEMAP_REINHARD_WHITE : u32 = 2u;
const TONEMAP_SMOOTH : u32 = 3u;
const KGRAD_SCALE : f32 = 16777216.0; // 2^24; |dL/dk|, |dL/dW| stay far below 128

fn tonemap_has_white() -> bool {
  return TONEMAP == TONEMAP_REINHARD_WHITE || TONEMAP == TONEMAP_SMOOTH;
}

fn add_loss(v : f32) {
  atomicAdd(&lossAccum[0], bitcast<u32>(i32(round(v * LOSS_SCALE))));
}

fn add_kgrad(v : f32) {
  atomicAdd(&lossAccum[1], bitcast<u32>(i32(round(v * KGRAD_SCALE))));
}

fn add_wgrad(v : f32) {
  atomicAdd(&lossAccum[2], bitcast<u32>(i32(round(v * KGRAD_SCALE))));
}

// Residual, cotangent and squared error for one pixel, shared by the two-pass
// and fused entry points so both produce identical bits. Returns (d2, dk, dw).
fn l2_pixel(idx : u32, img : vec3<f32>) -> vec3<f32> {
  let n = f32(U.size.x * U.size.y) * 3.0;
  var d2 = 0.0;
  var dk = 0.0;
  var dw = 0.0;
  for (var c = 0u; c < 3u; c++) {
    if (TONEMAP == TONEMAP_REINHARD) {
      let x = img[c];
      let denom = 1.0 + U.tonemapK * x;
      let d = U.tonemapK * x / denom - targetImg[idx + c];
      let cot = 2.0 * d / n;
      dLdImage[idx + c] = cot * U.tonemapK / (denom * denom);
      dk += cot * x / (denom * denom);
      d2 += d * d / n;
    } else if (TONEMAP == TONEMAP_REINHARD_WHITE) {
      let x = img[c];
      let kw = U.tonemapK * U.tonemapW;
      let nrm = (1.0 + kw) / kw;
      let denom = 1.0 + U.tonemapK * x;
      let r = U.tonemapK * x / denom;
      let d = nrm * r - targetImg[idx + c];
      let cot = 2.0 * d / n;
      dLdImage[idx + c] = cot * nrm * U.tonemapK / (denom * denom);
      // d(nrm)/dk = -1/(k^2 W), d(nrm)/dW = -1/(k W^2); matches tonemapChain.
      dk += cot * (nrm * x / (denom * denom) - r / (U.tonemapK * kw));
      dw -= cot * r / (kw * U.tonemapW);
      d2 += d * d / n;
    } else if (TONEMAP == TONEMAP_SMOOTH) {
      let x = img[c];
      let u = U.tonemapK * x;
      let v = U.tonemapK * U.tonemapW;
      let iu = inverseSqrt(1.0 + u * u);
      let iv = inverseSqrt(1.0 + v * v);
      let sv = v * iv;
      let spu = iu * iu * iu; // s'(k*x)
      let spv = iv * iv * iv; // s'(k*W)
      let t = u * iu / sv;
      let d = t - targetImg[idx + c];
      let cot = 2.0 * d / n;
      dLdImage[idx + c] = cot * U.tonemapK * spu / sv;
      dk += cot * (x * spu - t * U.tonemapW * spv) / sv;
      dw -= cot * t * U.tonemapK * spv / sv;
      d2 += d * d / n;
    } else {
      let d = img[c] - targetImg[idx + c];
      dLdImage[idx + c] = 2.0 * d / n;
      d2 += d * d / n;
    }
  }
  dLdImage[idx + 3u] = 0.0;
  return vec3<f32>(d2, dk, dw);
}

// One atomic add per workgroup. All lanes must reach the barrier, so callers
// pass zeros for out-of-bounds pixels rather than returning early.
fn reduce_loss(lidx : u32, v : vec3<f32>) {
  lossPartial[lidx] = v.x;
  if (TONEMAP != 0u) { kPartial[lidx] = v.y; }
  if (tonemap_has_white()) { wPartial[lidx] = v.z; }
  workgroupBarrier();
  if (lidx == 0u) {
    var s = 0.0;
    for (var i = 0u; i < 64u; i++) { s += lossPartial[i]; }
    add_loss(s);
    if (TONEMAP != 0u) {
      var k = 0.0;
      for (var i = 0u; i < 64u; i++) { k += kPartial[i]; }
      add_kgrad(k);
    }
    if (tonemap_has_white()) {
      var w = 0.0;
      for (var i = 0u; i < 64u; i++) { w += wPartial[i]; }
      add_wgrad(w);
    }
  }
}

// Generic two-pass form: reads the image a loss-agnostic `forward` produced.
// This is the shape any new loss kernel takes.
@compute @workgroup_size(8, 8)
fn l2grad(@builtin(global_invocation_id) gid : vec3<u32>,
          @builtin(local_invocation_index) lidx : u32) {
  let inb = gid.x < U.size.x && gid.y < U.size.y;
  var v = vec3<f32>(0.0);
  if (inb) {
    let idx = 4u * (gid.y * U.size.x + gid.x);
    v = l2_pixel(idx, vec3<f32>(image[idx], image[idx + 1u], image[idx + 2u]));
  }
  reduce_loss(lidx, v);
}

// Fused fast path: the composited pixel is still in registers, so the image
// round-trip and a whole dispatch disappear. L2 only -- see the contract above.
@compute @workgroup_size(8, 8)
fn forward_l2(@builtin(global_invocation_id) gid : vec3<u32>,
              @builtin(local_invocation_index) lidx : u32) {
  let inb = gid.x < U.size.x && gid.y < U.size.y;
  var v = vec3<f32>(0.0);
  if (inb) {
    let idx = 4u * (gid.y * U.size.x + gid.x);
    let img = composite(gid);
    store_image(idx, img);   // backward still replays against the image
    v = l2_pixel(idx, img.rgb);
  }
  reduce_loss(lidx, v);
}

// ----------------------------------------------------------------- backward
//
// Parameter groups are pipeline-specialized, not branched at runtime: a frozen
// group's gradient code is dead and the compiler removes it, along with its
// buffer bindings. Freezing must not perturb the groups still trained, so the
// surviving code and its fixed-point accumulation are byte-for-byte unchanged.
override TRAIN_GEOMETRY : bool = true;
override TRAIN_COLOR : bool = true;
override TRAIN_ALPHA : bool = true;
// Per-shape filter-size (blur) training is off by default: it widens the
// shape-gradient stride from 4 to 6 (slots 4, 5 hold dL/dsx, dL/dsy), so the
// default pipelines carry no extra memory or atomics whatsoever.
override TRAIN_BLUR : bool = false;
override SHAPE_GRAD_STRIDE : u32 = 4u;

// Deterministic fixed-point atomics replace unavailable f32 atomicAdd.
//
// U.fixedScale is sized so the worst-case *sum* fits i32, which leaves a single
// pixel's contribution only a few integer steps: rounded alone, a small one
// becomes zero and that pixel stops contributing at all. So each accumulator
// is two words. `hi` at [i] keeps that scale and its overflow bound; `lo` at
// [half + i] carries FX_LO_BITS further fractional bits. q splits exactly as
// (q >> B) * 2^B + (q & mask) under the arithmetic shift, and both words are
// plain wrapping adds, so the sum stays exact and order-independent. lo is
// unsigned and at most `mask` per contribution, so a word holds 2^32 / 255 =
// 16.8M contributions; reduce_slots carries each slot's lo into hi before
// summing, and a piece only hears from pixels along its own curve.
const FX_LO_BITS : u32 = 8u;
const FX_LO_MASK : i32 = 255;
const FX_LO_SCALE : f32 = 256.0;

fn add_piece_grad(i : u32, v : f32) {
  let q = i32(round(v * U.fixedScale * FX_LO_SCALE));
  if (q == 0) { return; }
  let half = arrayLength(&pieceGrads) / 2u;
  atomicAdd(&pieceGrads[i], bitcast<u32>(q >> FX_LO_BITS));
  atomicAdd(&pieceGrads[half + i], bitcast<u32>(q & FX_LO_MASK));
}

fn add_shape_grad(i : u32, v : f32) {
  let q = i32(round(v * U.fixedScale * FX_LO_SCALE));
  if (q == 0) { return; }
  let half = arrayLength(&shapeGrads) / 2u;
  atomicAdd(&shapeGrads[i], bitcast<u32>(q >> FX_LO_BITS));
  atomicAdd(&shapeGrads[half + i], bitcast<u32>(q & FX_LO_MASK));
}

// Replay compositing top-down and contract the analytic piece derivatives.
//
// Only src-over replays against the forward image (outC): its adjoint needs
// the canvas below each layer. add's adjoint is a per-shape constant, and
// multiply's and screen's are reconstructed from a product recomputed here,
// so none of them depends on `image` -- their backward runs without a
// forward readback.
@compute @workgroup_size(8, 8)
fn backward(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(workgroup_id) wgid : vec3<u32>) {
  if (gid.x >= U.size.x || gid.y >= U.size.y) { return; }
  let cx = U.origin.x + (f32(gid.x) + 0.5) * U.scale;
  let cy = U.origin.y + (f32(gid.y) + 0.5) * U.scale;
  let idx = 4u * (gid.y * U.size.x + gid.x);
  let dL = vec3<f32>(dLdImage[idx], dLdImage[idx + 1u], dLdImage[idx + 2u]);
  var outC = vec3<f32>(0.0);
  if (BLEND_MODE == BLEND_SRC_OVER) {
    outC = vec3<f32>(image[idx], image[idx + 1u], image[idx + 2u]);
  }
  let slot = (wgid.x + wgid.y * 7919u) % U.slots;

  // Bounded top-down composite adjoint: S_i = out - D.
  var P = 1.0;
  var D = vec3<f32>(0.0);

  let ti = pixel_tile(gid);
  let lo = tileOffset[ti];
  let hi = tileOffset[ti + 1u];

  // multiply/screen: one factor's adjoint is the product of all the others.
  // Recompute the full product first, then divide each factor back out with a
  // guard.
  var M = vec3<f32>(1.0);
  if (BLEND_MODE == BLEND_MULTIPLY || BLEND_MODE == BLEND_SCREEN) {
    for (var k = hi; k > lo; k--) {
      let si = tileShapes[k - 1u];
      if (!in_bbox(si, cx, cy)) { continue; }
      let sh = shapes[si];
      let F = shape_winding(si, cx, cy);
      if (F == 0.0) { continue; }
      let fold = coverage_fold(F, sh.info.z);
      if (BLEND_MODE == BLEND_SCREEN) {
        M *= 1.0 - fold.x * sh.color.w * sh.color.rgb;
      } else {
        M *= 1.0 - fold.x * sh.color.w * (1.0 - sh.color.rgb);
      }
    }
  }

  // top shape first: walk this tile's list in reverse (= painter's order
  // reversed). add/multiply lists are unsorted, but their adjoints are
  // order-independent, so the shared direction is fine.
  for (var k = hi; k > lo; k--) {
    // Lower layers and their gradients are negligible below this cutoff.
    if (BLEND_MODE == BLEND_SRC_OVER && P < 1e-5) { break; }
    let si = tileShapes[k - 1u];
    if (!in_bbox(si, cx, cy)) { continue; }
    let sh = shapes[si];
    let sf = shape_filter(si);
    let hx = 0.5 * sf.x;
    let hy = 0.5 * sf.y;
    let F = shape_winding(si, cx, cy);
    // Zero winding is an identity layer.
    if (F == 0.0) { continue; }
    let fold = coverage_fold(F, sh.info.z);
    let cov = fold.x;
    let a = cov * sh.color.w;
    var dColor = vec3<f32>(0.0); // dL/d(color.rgb)
    var dLda = 0.0;              // dL/d(a), a = coverage * alpha
    if (BLEND_MODE == BLEND_ADD) {
      // out = bg + sum(color * a): exact adjoints, no guard, no cross terms.
      dColor = dL * a;
      dLda = dot(dL, sh.color.rgb);
    } else if (BLEND_MODE == BLEND_MULTIPLY) {
      let m = 1.0 - a * (1.0 - sh.color.rgb);
      // Guarded divide: as a factor saturates toward 0 its reconstructed
      // sibling product fades to 0 smoothly -- the multiply analogue of
      // src-over's max(1 - a, 1e-3) below.
      let others = U.bg.rgb * M / max(m, vec3<f32>(1e-3));
      dColor = dL * others * a;
      dLda = dot(dL * others, sh.color.rgb - vec3<f32>(1.0));
    } else if (BLEND_MODE == BLEND_SCREEN) {
      // multiply's adjoint in complement space: the sibling product scales by
      // 1 - bg, and out = 1 - (1 - bg) * prod, so the two signs cancel.
      let m = 1.0 - a * sh.color.rgb;
      let others = (1.0 - U.bg.rgb) * M / max(m, vec3<f32>(1e-3));
      dColor = dL * others * a;
      dLda = dot(dL * others, sh.color.rgb);
    } else {
      let aP = a * P;
      D += sh.color.rgb * aP;
      let S = outC - D; // canvas below i, pre-scaled by P_i (1 - a_i)
      dColor = dL * aP;
      dLda = dot(dL, sh.color.rgb * P - S / max(1.0 - a, 1e-3));
    }
    // Per-shape color/alpha gradients, strided by slot. The stride stays
    // fixed whatever is frozen: a frozen slot is simply left at zero, so every
    // model's pullback keeps the same layout.
    let so = SHAPE_GRAD_STRIDE * (slot * U.nShapes + si);
    if (TRAIN_COLOR) {
      add_shape_grad(so, dColor.r);
      add_shape_grad(so + 1u, dColor.g);
      add_shape_grad(so + 2u, dColor.b);
    }
    if (TRAIN_ALPHA) {
      add_shape_grad(so + 3u, dLda * cov);
    }
    // coverage -> winding, including the selected fill rule's fold slope.
    let dF = select(0.0, fold.y * dLda * sh.color.w, TRAIN_GEOMETRY || TRAIN_BLUR);
    if ((TRAIN_GEOMETRY || TRAIN_BLUR) && dF != 0.0) {
      let dA = dF / (sf.x * sf.y);
      let start = sh.info.x;
      let count = sh.info.y;
      // Filter-size adjoints come free from the piece gradients: the box
      // integral is homogeneous per axis, A(t*qx, t*hx) = t*A, so Euler's
      // identity gives dL/dsx = -sum(qx * dL/dqx) / sx (same for y). The
      // contraction reuses integrate_piece_grad's exact branch subgradients,
      // so blur gradients match JAX by construction.
      var sxAdj = 0.0;
      var syAdj = 0.0;
      for (var p = 0u; p < count; p++) {
        let o = 6u * (start + p);
        // Mirror forward's strict y-band cull.
        let q1y = pieces[o + 1] - cy;
        let q3y = pieces[o + 5] - cy;
        if (max(q1y, q3y) < -hy || min(q1y, q3y) > hy) { continue; }
        let q1x = pieces[o] - cx;
        let q2x = pieces[o + 2] - cx;
        let q3x = pieces[o + 4] - cx;
        if (max(q1x, max(q2x, q3x)) < -hx) { continue; }
        let q1 = vec2<f32>(q1x, q1y);
        let q2 = vec2<f32>(q2x, pieces[o + 3] - cy);
        let q3 = vec2<f32>(q3x, q3y);
        let g = integrate_piece_grad(q1, q2, q3, -hy, hy, hx, dA);
        if (TRAIN_GEOMETRY) {
          add_piece_grad(o, g.dq1.x);
          add_piece_grad(o + 1u, g.dq1.y);
          add_piece_grad(o + 2u, g.dq2.x);
          add_piece_grad(o + 3u, g.dq2.y);
          add_piece_grad(o + 4u, g.dq3.x);
          add_piece_grad(o + 5u, g.dq3.y);
        }
        if (TRAIN_BLUR) {
          sxAdj += q1.x * g.dq1.x + q2.x * g.dq2.x + q3.x * g.dq3.x;
          syAdj += q1.y * g.dq1.y + q2.y * g.dq2.y + q3.y * g.dq3.y;
        }
      }
      if (TRAIN_BLUR) {
        add_shape_grad(so + 4u, -sxAdj / sf.x);
        add_shape_grad(so + 5u, -syAdj / sf.y);
      }
    }
    P *= (1.0 - a);
  }
}

// Reduce strided shape-gradient slots on-GPU before readback.
@compute @workgroup_size(64)
fn reduce_slots(@builtin(global_invocation_id) gid : vec3<u32>) {
  let n = U.nShapes * SHAPE_GRAD_STRIDE;
  if (gid.x >= n) { return; }
  let half = arrayLength(&shapeGrads) / 2u;
  var hi = 0u;
  var lo = 0u;
  for (var s = 0u; s < U.slots; s++) {
    let l = atomicLoad(&shapeGrads[half + s * n + gid.x]);
    hi += atomicLoad(&shapeGrads[s * n + gid.x]) + (l >> FX_LO_BITS);
    lo += l & u32(FX_LO_MASK);
  }
  // Decode here, so the host reads f32 gradients back directly.
  let total = (f32(bitcast<i32>(hi)) * FX_LO_SCALE + f32(lo)) / (U.fixedScale * FX_LO_SCALE);
  atomicStore(&shapeGrads[gid.x], bitcast<u32>(total));
}

// Contract piece gradients to the original quadratic controls. Split parameters
// are constants because the summed winding integral is subdivision-invariant.
fn load_piece_grad2(o : u32, invScale : f32) -> vec2<f32> {
  let half = arrayLength(&pieceGrads) / 2u;
  let hi = vec2<f32>(
    f32(bitcast<i32>(atomicLoad(&pieceGrads[o]))),
    f32(bitcast<i32>(atomicLoad(&pieceGrads[o + 1u])))
  );
  let lo = vec2<f32>(
    f32(atomicLoad(&pieceGrads[half + o])),
    f32(atomicLoad(&pieceGrads[half + o + 1u]))
  );
  return (hi * FX_LO_SCALE + lo) * (invScale / FX_LO_SCALE);
}

@compute @workgroup_size(64)
fn reduce_curve_grads(@builtin(global_invocation_id) gid : vec3<u32>) {
  let curve = gid.x;
  if (curve >= U.nCurves) { return; }

  let m = curveMeta[curve];
  let t1 = m.splitT.x;
  let t2 = m.splitT.y;
  let u1 = 1.0 - t1;
  let u2 = 1.0 - t2;
  let d1 = vec3<f32>(u1 * u1, 2.0 * t1 * u1, t1 * t1);
  let d2 = vec3<f32>(u2 * u2, 2.0 * t2 * u2, t2 * t2);
  let mid = vec3<f32>(u1 * u2, t1 * u2 + t2 * u1, t1 * t2);
  let invScale = 1.0 / U.fixedScale;

  var db0 = vec2<f32>(0.0);
  var db1 = vec2<f32>(0.0);
  var db2 = vec2<f32>(0.0);
  var p = m.pieceStart;

  if ((m.mask & 1u) != 0u) {
    let o = 6u * p;
    let q0 = load_piece_grad2(o, invScale);
    let q1 = load_piece_grad2(o + 2u, invScale);
    let q2 = load_piece_grad2(o + 4u, invScale);
    db0 += q0 + u1 * q1 + d1.x * q2;
    db1 += t1 * q1 + d1.y * q2;
    db2 += d1.z * q2;
    p += 1u;
  }
  if ((m.mask & 2u) != 0u) {
    let o = 6u * p;
    let q0 = load_piece_grad2(o, invScale);
    let q1 = load_piece_grad2(o + 2u, invScale);
    let q2 = load_piece_grad2(o + 4u, invScale);
    db0 += d1.x * q0 + mid.x * q1 + d2.x * q2;
    db1 += d1.y * q0 + mid.y * q1 + d2.y * q2;
    db2 += d1.z * q0 + mid.z * q1 + d2.z * q2;
    p += 1u;
  }
  if ((m.mask & 4u) != 0u) {
    let o = 6u * p;
    let q0 = load_piece_grad2(o, invScale);
    let q1 = load_piece_grad2(o + 2u, invScale);
    let q2 = load_piece_grad2(o + 4u, invScale);
    db0 += d2.x * q0;
    db1 += d2.y * q0 + u2 * q1;
    db2 += d2.z * q0 + t2 * q1 + q2;
  }

  let out = 6u * curve;
  curveGrads[out] = db0.x;
  curveGrads[out + 1u] = db0.y;
  curveGrads[out + 2u] = db1.x;
  curveGrads[out + 3u] = db1.y;
  curveGrads[out + 4u] = db2.x;
  curveGrads[out + 5u] = db2.y;
}
