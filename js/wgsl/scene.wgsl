// Shared scene state: uniforms, shape table, and the buffers every render
// stage binds. Pipelines use auto layouts, so each entry point only pays for
// the bindings it statically references (backward already uses the WebGPU
// default of eight storage buffers — keep new kernels at or below that).

struct Uniforms {
  size : vec2<u32>,   // W, H
  nShapes : u32,
  slots : u32,        // stride count for shape-gradient accumulation
  origin : vec2<f32>, // curve-space position of the grid's top-left corner
  scale : f32,        // curve units per pixel
  fixedScale : f32,   // backward fixed-point scale of an accumulator's hi word (see add_piece_grad)
  s : vec2<f32>,      // filter size (sx, sy) in curve units
  tilesX : u32,       // tile grid width  = ceil(W / TILE)
  tilesY : u32,       // tile grid height = ceil(H / TILE)
  bg : vec3<f32>,
  nCurves : u32,
  tonemapK : f32,     // exposure for TONEMAP != 0; the composite stays linear
  tonemapW : f32,     // white point for the white-normalized operators (T(W) = 1)
};

struct Shape {
  bbox : vec4<f32>,  // lox, loy, hix, hiy (curve units)
  color : vec4<f32>, // rgb + alpha
  info : vec4<u32>,  // pieceStart, pieceCount, fillRule (0 nonzero, 1 evenodd), 0
  filt : vec4<f32>,  // per-shape filter (sx, sy, 0, 0); sx == 0 -> use global U.s
};

@group(0) @binding(0) var<uniform> U : Uniforms;
@group(0) @binding(1) var<storage, read> shapes : array<Shape>;
@group(0) @binding(2) var<storage, read> pieces : array<f32>;      // 6 f32 per piece (3 x vec2)
@group(0) @binding(3) var<storage, read_write> image : array<f32>; // W*H*4 rgba

// L2 writes dLdImage on-GPU; CLIP uploads it from the host.
@group(0) @binding(4) var<storage, read_write> dLdImage : array<f32>;            // W*H*4
// Gradient accumulators are two words each: hi words first, lo words in the second half.
@group(0) @binding(5) var<storage, read_write> pieceGrads : array<atomic<u32>>; // 2 * P*6
@group(0) @binding(6) var<storage, read_write> shapeGrads : array<atomic<u32>>; // 2 * slots*S*stride

// L2-loss bindings (fused GPU loss path)
@group(0) @binding(7) var<storage, read> targetImg : array<f32>;                // W*H*4
@group(0) @binding(8) var<storage, read_write> lossAccum : array<atomic<u32>>;  // loss, dL/dk, dL/dW (fixed-point i32)

// Sorted per-tile painter lists: counts, prefix offsets, then shape indices.
@group(0) @binding(9)  var<storage, read_write> tileCount : array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> tileOffset : array<u32>;
@group(0) @binding(11) var<storage, read_write> tileShapes : array<u32>;

// Per-curve split parameters and the live compact-piece mask.
struct CurveMeta {
  splitT : vec2<f32>,
  pieceStart : u32,
  mask : u32,
};
@group(0) @binding(12) var<storage, read> curveMeta : array<CurveMeta>;
@group(0) @binding(13) var<storage, read_write> curveGrads : array<f32>; // C*6, plain f32

// Per-block totals for the parallel tile-offset scan: ceil(nTiles / 256).
@group(0) @binding(14) var<storage, read_write> tileBlockSum : array<u32>;

const TILE : u32 = 16u; // pixels per tile side; MUST match TILE in renderer.js

const TINY : f32 = 1e-12;
const FOLD_EPS : f32 = 1e-5;
const FILL_EVENODD : u32 = 1u;

fn nonzero(x : f32) -> f32 {
  return select(x, select(TINY, -TINY, x < 0.0), abs(x) < TINY);
}
