# Windfoil performance investigation

This records the retained changes and local measurements on `codex/perf` on
30 September 2026. The changes remove a dense-tile sorting cliff, reuse compiled
pipelines, reduce export allocations and uploads, and accelerate target resizing.
A GPU optimizer experiment was removed to keep this reference repository small;
a brief record of its approach and results is retained below. A fresh Node
process starting in 10 ms has not been demonstrated.

Measurements used an Apple M2, macOS 13, Node 24.16.0, and
`webgpu-legacy` 0.3.0 through Dawn/Metal. NVIDIA was not measured during this
investigation. The numerical results are saved in
[performance-results.json](performance-results.json). Detailed runs, images,
and validation logs remain in the ignored `output/perf/` directory.

## Shader findings

The original dense-tile path had a major performance cliff: a tile list that
exceeded workgroup sort storage used a serial comb sort. The replacement assigns
consecutive shape IDs to lanes, counts tile intersections, scans those counts,
then gathers the IDs in ascending order. It preserves painter order and uses the
existing compact list allocation. Sparse tiles retain the existing bitonic sort.

For 8,192 overlapping shapes on a 128×128 canvas, the complete static evaluation
fell from **44.03 ms to 4.48 ms**, a **9.8×** improvement. The loss was identical.
A separate forced-capacity GPU test compares the gather with the bitonic path
and requires identical images, loss, and geometry/color gradients. This win
specifically removes the dense-tile cliff; it is not the speedup for ordinary
sparse scenes.

### How the dense tile gather works

The ordinary binning passes count intersections, allocate compact ranges with a
prefix sum, and scatter shape IDs into each range. Atomic scatter gives arbitrary
order, so `bin_sort` restores ascending IDs. The existing bitonic sort handles
lists that fit in workgroup memory. The old overflow fallback ran comb sort on
one lane, repeatedly comparing and swapping IDs in global memory.

The replacement overflow fallback reconstructs the list in order:

1. Split all shape IDs into 256 consecutive chunks, one per lane.
2. Each lane scans its chunk and counts shapes touching this tile, using the
   same `shape_tile_range` helper as count and fill.
3. Prefix-sum the 256 counts in shared memory to obtain disjoint output ranges.
4. Rescan each chunk, writing matching IDs in ascending order into its range.

For example, four lanes scanning IDs 0–15 might find these matches. Offsets are
relative to the tile's already allocated range:

| Lane | IDs examined | Matching IDs | Output offset |
| --- | --- | --- | ---: |
| 0 | 0–3 | 0, 3 | 0 |
| 1 | 4–7 | 6 | 2 |
| 2 | 8–11 | none | 3 |
| 3 | 12–15 | 12, 15 | 3 |

The result is `[0, 3, 6, 12, 15]`. Consecutive chunks and disjoint output ranges
make the list globally sorted without compare/swap sorting. Work is two scans of
all shapes plus a small parallel scan. The existing count/fill passes still run;
the overflow path overwrites their unsorted IDs using the same compact offsets.

The added shader logic is localized, with a shape-table binding added to the
sort pipeline and about 1 KiB of workgroup scratch. Maintenance obligations are:

- **Membership must agree.** Count, fill, and gather must include exactly the
  same shapes; otherwise offsets and emitted counts disagree. All three currently
  use `shape_tile_range`, keeping the bbox/filter calculation in one place.
- **Barriers must remain valid.** Each prefix-sum round separates reads from
  writes with barriers. The fallback decision uses workgroup-uniform tile
  metadata so every lane takes the same path through those barriers.
- **Storage accounting must agree.** Changing scratch sizes requires updating
  the host's workgroup-memory budget and sort-capacity selection.
- **Both paths need coverage.** The exact comparison test forces capacities
  2 and 2048 so gather and bitonic output, gradients, and loss are compared even
  on hardware whose normal sort capacity would avoid overflow.

This is a moderate, localized maintenance cost. Coverage, its VJP, scene formats,
and optimizer mathematics are unchanged. The performance tradeoff is that an
oversized tile scans all N shapes, including shapes absent from that tile. A tile
just above the threshold in a huge sparse scene may gain little or regress;
9.8× describes the measured crowded case rather than a universal speedup.

Outside that cliff, the expensive work is coverage and its analytic VJP. In a
diagnostic replay at 512×512 with 512 shapes, backward took about 1.38 ms,
forward with L2 about 0.33 ms, and sorting about 0.13 ms. With 4,096 shapes,
backward was about 2.95 ms. These replay timings split dispatches into separate
passes, which changes scheduling. Dawn also quantized timestamps to about
65.5 µs here. Use wall-clock timings to compare implementations.

Several plausible changes failed to improve performance broadly and were
discarded: returning from saturated root solves before computing the quadratic,
smaller sort scratch for sparse tiles, increasing gradient slots from 16 to 64,
and merging the ordinary CPU optimizer's two submissions into one. The analytic
coverage and VJP math remain unchanged. The shader is not proven optimal; the
measurements identify the expensive stage and rule out these particular changes.

## Startup findings

The historical benchmark's `startup_ms` means process spawn to the first
optimizer step: interpreter, imports, input decoding, device initialization,
model setup, shader compilation, and warmup. Saved NVIDIA results include
roughly 650–710 ms for Kodak workloads, consistent with the reported 0.7 s.
They do not isolate shader compilation, and these changes have not been timed
on NVIDIA.

The renderer now caches sources, shader modules, and specialization-specific
pipelines per device. It creates only the selected scan and loss pipelines,
and compiles generic forward when needed. Resolution and scene capacities can
change while compiled code is reused; blend, tonemap, training flags, output
alpha, and sort capacity are part of the pipeline specialization.

The following is initialization, upload, and the first L2 evaluation for the
second renderer on an already-live device, using the original code at
`88d6d7c` versus this branch:

| Canvas | Shapes | Original ready ms | Cached ready ms |
| --- | ---: | ---: | ---: |
| 128×128 | 64 | 20.55 | 1.18 |
| 512×512 | 512 | 22.67 | 3.96 |
| 1024×1024 | 512 | 32.45 | 11.56 |
| 512×512 | 4096 | 27.11 | 6.69 |

This supports a sub-10-ms target for small renderers when a process and device
stay alive. It does not establish 10 ms for a fresh Node command. Local device
initialization alone commonly took 35–65 ms; the first renderer took about
60–70 ms in several runs, with larger compilation spikes when shader code was
new. Before device creation, Node and imports were already about 19–26 ms.
Large new image buffers also retain a noticeable allocation cost.

Target resizing now precomputes repeated coordinates, eliminates per-pixel
arrays, and reuses alpha samples. A 32-cell check spans 64–1024 px, all four
PNG channel counts, and 8/16-bit samples. Every cell improved and every
Float32 output pixel matched the original. RGBA8 resizing at 512×341 fell from
20.08 to 2.80 ms; at 1024×683 it fell from 80.05 to 11.16 ms. Additional exact
checks exercise noninteger scaling, single-pixel output, and clamped borders.

Input decoding is a separate cost: loading the default 14 MB, 4925×2770 JPEG
took about 1.13 s locally using `jpeg-js`. Resizing improvements do not remove
this decode. Reusing decoded inputs, or preparing targets before a timed request,
is necessary for very short request latency. The CLI already reuses a decoded
source across sweep cells within one process.

To target 10 ms request latency, retain the Node worker, GPU device, pipelines,
and common-size renderers/targets. Persisting compiled pipelines across fresh
processes would require investigating the native addon's cache support.
[Dawn has native pipeline cache machinery](https://dawn.googlesource.com/dawn/+/refs/heads/chromium/7424/src/dawn/native/PipelineCache.h);
the application-level cache here lasts only for a live device.
[WebGPU provides asynchronous pipeline creation](https://gpuweb.github.io/types/interfaces/GPUDevice),
which can move compilation off a blocking call; it does not by itself eliminate
the compilation or device-start cost.

## Rendering and exports

`Renderer.create(device, { ..., forwardOnly: true })` now allocates no training
gradients, image cotangents, curve metadata, or training pipelines. Chunked
exports upload immutable scene geometry once and change the raster settings
for subsequent chunks. Repeated forward renders reuse binning until an upload
marks it dirty. A real GPU test requires chunked and whole-image output to
match exactly. Frame, crop, and export renderers use this mode.

## Removed GPU optimizer experiment

The GPU optimizer implementation and its CLI/API integration were removed at
the user's request because they duplicated reference mathematics and increased
maintenance cost. These are historical experiment results, not current features.

It kept loop parameters and Adam moments in GPU buffers. Each step decoded the
closed quadratic loops, split them at x/y extrema into at most three monotone
pieces per curve, ran the existing renderer and analytic VJP, applied the style
pullback, and updated parameters with GPU Adam. Batches of 16 sequential steps
used one submission and read one loss word per step; parameters returned to the
CPU for export or model-dependent callbacks. Default anchor colour/opacity and
raw gray/RGB with fixed alpha were implemented for L2.

On the local Apple M2, all 12 tested combinations of 64–1024 px and 64–4096
shapes had improved aggregate medians: 1.05–3.72× (four curves per shape, 1 px
filter, 96 steps, three repeats). These batch timings excluded final parameter
readback. Four real Kodak image fits improved 1.5–1.7× including that readback
(eight curves, 7-to-1 blur anneal, 64 steps); final PSNR differed by less than
0.03 dB. Tiny shapes, crowded scenes, and wider blur also improved. No NVIDIA
measurement of this experiment was made.

A future implementation must address the costs that prompted removal: duplicated
CPU decode/style/Adam logic, f32 differences from JavaScript arithmetic, callback
synchronization, and unsupported codecs/features. It also reserved the worst-case
list of `ceil(width/16) * ceil(height/16) * nShapes` indices: 64 MiB at 1024²
and 4096 shapes before other buffers. A compact allocation with overflow checking
and retry would be needed for larger scenes. Detailed historical runs remain in
ignored `output/perf/`, including `matrix-anchor-{cpu,gpu}.json` and `cli-{cpu,gpu}/`.

## Remaining opportunities

The larger-canvas bottleneck is coverage and backward. Better curve/edge culling
or selectively retaining intermediates needs a memory and divergence study;
per-pixel/per-shape storage can grow dramatically. Cold NVIDIA startup needs
separate measurements of imports, device setup, pipeline creation, and first
dispatch before investigating native cache support. These are future
investigations, not measured speedups from this branch.

## Reproducing the measurements

The deterministic harness uses synthetic curves and a constant target to isolate
rendering and optimizer costs. `--optimize=1` performs real CPU Adam updates;
without it, geometry is static. Shader compilation and the first evaluation
are reported separately. `--gpu=1` adds
diagnostic dispatch timestamps.

```sh
npm run perf -- --style=anchor --optimize=1 --steps=96 --repeats=3 \
  --cases=64:64:0.04,64:512:0.04,64:4096:0.04,128:64:0.04,128:512:0.04,128:4096:0.04,512:64:0.04,512:512:0.04,512:4096:0.04,1024:64:0.04,1024:512:0.04,1024:4096:0.04 \
  --out=output/perf/cpu.json

# Repeat with --k=8 --blur=7 and with radii 0.005 and 0.3.

npm run demo:l2 -- --target=photo.png --n=64,512 --opt-size=128,512 \
  --steps=64 --quiet --out=output/perf/fit-cpu

# Save original sources for shader/startup and image-resize comparisons.
mkdir -p output/perf/baseline
git archive 88d6d7c js demos/util/image.js | tar -x -C output/perf/baseline
npm run perf -- --source=output/perf/baseline --cases=128:8192:0.4 \
  --steps=10 --repeats=1 --out=output/perf/dense-original.json
npm run perf -- --cases=128:8192:0.4 --steps=10 --repeats=1 \
  --out=output/perf/dense-current.json
npm run perf:image -- --baseline=output/perf/baseline/demos/util/image.js
```

## Correctness and validation

`npm run validate` passes JavaScript tests available in the sandbox, JAX unit
checks, and Deno type checks. The three retained real-GPU tests pass on local Dawn.
The WebGPU/JAX oracle passes all 11 fill/blend/blur/tonemap cases. The browser
production build passes.

The complete local GPU suite has one existing failure in the frozen-colour
bit-identity test. One alpha gradient is `0.0008328245021402836` rather than
`0.0008328244439326227`, a single f32 ULP. Running that test against the original
`88d6d7c` sources produces the same values and failure. Its assertion has not
been relaxed. The gather test requires exact equality for images, loss, and
geometry/shape gradients.
