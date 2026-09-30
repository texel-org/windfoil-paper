// Tile binning: conservative per-shape tile ranges, counts, prefix offsets,
// compact fill, and an exact ascending sort restoring painter order. Tile
// lists conservatively cull shapes before the exact per-pixel bbox test.

// Conservative inclusive tile range; .x reports an on-image hit.
fn axis_tiles(lo : f32, hi : f32, org : f32, dim : u32) -> vec3<u32> {
  let plo = (lo - org) / U.scale - 0.5; // real-valued pixel-index bounds
  let phi = (hi - org) / U.scale - 0.5;
  if (phi < 0.0 || plo > f32(dim - 1u)) { return vec3<u32>(0u, 0u, 0u); }
  let i0 = u32(clamp(floor(plo), 0.0, f32(dim - 1u)));
  let i1 = u32(clamp(ceil(phi), 0.0, f32(dim - 1u)));
  return vec3<u32>(1u, i0 / TILE, i1 / TILE);
}

// The tile block (tx0..tx1, ty0..ty1) a shape's filter-expanded bbox covers.
struct TileRange { hit : bool, tx0 : u32, ty0 : u32, tx1 : u32, ty1 : u32 };
fn shape_tile_range(si : u32) -> TileRange {
  let sf = shape_filter(si);
  let b = shapes[si].bbox;
  let ax = axis_tiles(b.x - 0.5 * sf.x, b.z + 0.5 * sf.x, U.origin.x, U.size.x);
  let ay = axis_tiles(b.y - 0.5 * sf.y, b.w + 0.5 * sf.y, U.origin.y, U.size.y);
  return TileRange(ax.x == 1u && ay.x == 1u, ax.y, ay.y, ax.z, ay.z);
}

// Count every tile touched by each shape bbox.
@compute @workgroup_size(64)
fn bin_count(@builtin(global_invocation_id) gid : vec3<u32>) {
  let si = gid.x;
  if (si >= U.nShapes) { return; }
  let r = shape_tile_range(si);
  if (!r.hit) { return; }
  for (var ty = r.ty0; ty <= r.ty1; ty++) {
    for (var tx = r.tx0; tx <= r.tx1; tx++) {
      atomicAdd(&tileCount[ty * U.tilesX + tx], 1u);
    }
  }
}

// Prefix-sum counts and reset them for fill cursors.
//
// Three parallel passes replace a single-lane loop over every tile: that loop
// serialized the whole device between bin_count and bin_fill, and its cost grew
// with tile count, so it hurt most exactly where tiles are most numerous.
// Result is identical: tileOffset holds the exclusive prefix sum, tileOffset[nt]
// the total, and tileCount is left zeroed for bin_fill's cursors.

const SCAN_WG : u32 = 256u;
var<workgroup> scanScratch : array<u32, SCAN_WG>;

// The original single-lane scan, kept only so WF_SCAN=serial can A/B it.
@compute @workgroup_size(1)
fn bin_scan() {
  let nt = U.tilesX * U.tilesY;
  var acc = 0u;
  for (var t = 0u; t < nt; t++) {
    tileOffset[t] = acc;
    acc += atomicExchange(&tileCount[t], 0u);
  }
  tileOffset[nt] = acc;
}

// Exclusive scan within each block; writes the block's total to tileBlockSum.
@compute @workgroup_size(SCAN_WG)
fn bin_scan_block(@builtin(global_invocation_id) gid : vec3<u32>,
                  @builtin(local_invocation_index) lidx : u32,
                  @builtin(workgroup_id) wgid : vec3<u32>) {
  let nt = U.tilesX * U.tilesY;
  let t = gid.x;
  // atomicExchange resets the counter for bin_fill in the same read.
  var v = 0u;
  if (t < nt) { v = atomicExchange(&tileCount[t], 0u); }
  scanScratch[lidx] = v;
  workgroupBarrier();

  // Hillis-Steele inclusive scan: log2(SCAN_WG) steps.
  for (var offset = 1u; offset < SCAN_WG; offset = offset << 1u) {
    var add = 0u;
    if (lidx >= offset) { add = scanScratch[lidx - offset]; }
    workgroupBarrier();
    scanScratch[lidx] += add;
    workgroupBarrier();
  }

  if (t < nt) { tileOffset[t] = scanScratch[lidx] - v; } // inclusive -> exclusive
  if (lidx == SCAN_WG - 1u) { tileBlockSum[wgid.x] = scanScratch[lidx]; }
}

// Exclusive scan of the per-block totals, in one workgroup. Blocks are
// nTiles/256, so a strided loop covers any canvas we can allocate.
@compute @workgroup_size(SCAN_WG)
fn bin_scan_blocks(@builtin(local_invocation_index) lidx : u32) {
  let nt = U.tilesX * U.tilesY;
  let blocks = (nt + SCAN_WG - 1u) / SCAN_WG;
  var carry = 0u;
  for (var base = 0u; base < blocks; base += SCAN_WG) {
    let i = base + lidx;
    var v = 0u;
    if (i < blocks) { v = tileBlockSum[i]; }
    scanScratch[lidx] = v;
    workgroupBarrier();
    for (var offset = 1u; offset < SCAN_WG; offset = offset << 1u) {
      var add = 0u;
      if (lidx >= offset) { add = scanScratch[lidx - offset]; }
      workgroupBarrier();
      scanScratch[lidx] += add;
      workgroupBarrier();
    }
    let total = scanScratch[SCAN_WG - 1u];
    if (i < blocks) { tileBlockSum[i] = carry + scanScratch[lidx] - v; }
    workgroupBarrier();
    carry += total;
  }
  if (lidx == 0u) { tileOffset[nt] = carry; }
}

// Add each block's base to its tiles.
@compute @workgroup_size(SCAN_WG)
fn bin_scan_add(@builtin(global_invocation_id) gid : vec3<u32>,
                @builtin(workgroup_id) wgid : vec3<u32>) {
  let nt = U.tilesX * U.tilesY;
  if (gid.x >= nt) { return; }
  tileOffset[gid.x] += tileBlockSum[wgid.x];
}

// Scatter each shape into its tiles at tileOffset[t] + (running cursor).
@compute @workgroup_size(64)
fn bin_fill(@builtin(global_invocation_id) gid : vec3<u32>) {
  let si = gid.x;
  if (si >= U.nShapes) { return; }
  let r = shape_tile_range(si);
  if (!r.hit) { return; }
  for (var ty = r.ty0; ty <= r.ty1; ty++) {
    for (var tx = r.tx0; tx <= r.tx1; tx++) {
      let t = ty * U.tilesX + tx;
      let pos = tileOffset[t] + atomicAdd(&tileCount[t], 1u);
      tileShapes[pos] = si;
    }
  }
}

// Restore painter order, one workgroup per tile: a 2k-entry workgroup bitonic
// sort for typical tiles, with an exact comb-sort fallback for larger ones.
// Shared-memory capacity for the parallel sort, in entries. The default fits
// WebGPU's guaranteed 16 KiB of workgroup storage; the host raises it to what
// the device actually offers, which moves the point where a tile falls back to
// the serial path. Must be a power of two: the bitonic network sorts a
// power-of-two span.
override SORT_CAPACITY : u32 = 2048u;
const SORT_WG : u32 = 256u;

var<workgroup> sortScratch : array<u32, SORT_CAPACITY>;
var<workgroup> sortInfo : vec2<u32>; // compact-list offset + length

@compute @workgroup_size(SORT_WG)
fn bin_sort(@builtin(workgroup_id) wgid : vec3<u32>,
            @builtin(local_invocation_id) lid : vec3<u32>) {
  let t = wgid.x;
  if (t >= U.tilesX * U.tilesY) { return; }
  if (lid.x == 0u) {
    let lo = tileOffset[t];
    let hi = tileOffset[t + 1u];
    sortInfo = vec2<u32>(lo, hi - lo);
  }
  // Synchronize metadata and establish uniform barrier control flow.
  let info = workgroupUniformLoad(&sortInfo);
  let lo = info.x;
  let n = info.y;
  let hi = lo + n;
  if (n <= 1u) { return; }

  // Exact fallback for tiles larger than workgroup storage. It is serial, so it
  // is a performance cliff -- raising SORT_CAPACITY is what keeps tiles off it.
  if (n > SORT_CAPACITY) {
    if (lid.x == 0u) {
      var gap = n;
      var swapped = true;
      while (gap > 1u || swapped) {
        gap = max((gap * 10u) / 13u, 1u);
        swapped = false;
        for (var i = lo; i + gap < hi; i++) {
          let a = tileShapes[i];
          let b = tileShapes[i + gap];
          if (a > b) {
            tileShapes[i] = b;
            tileShapes[i + gap] = a;
            swapped = true;
          }
        }
      }
    }
    return;
  }

  var span = 1u;
  while (span < n) { span = span << 1u; }
  for (var i = lid.x; i < span; i += SORT_WG) {
    sortScratch[i] = select(0xffffffffu, tileShapes[lo + i], i < n);
  }
  workgroupBarrier();

  var width = 2u;
  while (width <= span) {
    var stride = width >> 1u;
    while (stride > 0u) {
      for (var i = lid.x; i < span; i += SORT_WG) {
        let peer = i ^ stride;
        if (peer > i) {
          let a = sortScratch[i];
          let b = sortScratch[peer];
          let ascending = (i & width) == 0u;
          if ((ascending && a > b) || (!ascending && a < b)) {
            sortScratch[i] = b;
            sortScratch[peer] = a;
          }
        }
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    width = width << 1u;
  }

  for (var i = lid.x; i < n; i += SORT_WG) {
    tileShapes[lo + i] = sortScratch[i];
  }
}

fn pixel_tile(gid : vec3<u32>) -> u32 {
  return (gid.y / TILE) * U.tilesX + (gid.x / TILE);
}
