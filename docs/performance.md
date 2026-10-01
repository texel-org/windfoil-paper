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

## Sort threshold sweep

Lowering `WF_SORT_CAPACITY` reduces the bitonic scratch allocation and sends
tiles with longer lists through ordered gather sooner. The gather scans all N
shape bounds twice per tile; its crossover therefore depends on both N and
the tile's own list length. There is no universal best cutoff at 32 or 256.

The local sweep tested capacities 32–4096 and the default, using 50 static
evaluations per trial. Twelve workloads cover 64–1024 px, N=64–8192, and
small through heavily overlapping loops, with three repeats. Six representative
workloads were repeated four times with K=8 and diagnostic GPU timestamps;
an N-aware early-gather experiment was also tested at K=8 with 1 px and 7 px
filters. Every capacity produced identical images, loss, and geometry/color
gradients. These are local Apple M2 measurements.

The table gives GPU **sort-stage** times for K=8, 1 px filter, using the median
of the mean of 20 timestamp replays per trial. Replay passes change scheduling,
and native timestamps are quantized around 65.5 µs; averaging reduces that
granularity. These values describe the sorting tradeoff, not training speedup.

| Scene | Mean entries/tile | Default sort time | Lower cutoff | Sort time |
| --- | ---: | ---: | ---: | ---: |
| 128², N=512, crowded | 247 | 0.131 ms | 64 | 0.038 ms |
| 128², N=2048, crowded | 1021 | 0.655 ms | 256 | 0.139 ms |
| 512², N=4096, moderate overlap | 318 | 0.628 ms | 256 | 1.034 ms |
| 512², N=8192, sparse | 153 | 0.449 ms | 32 | 2.790 ms |
| 1024², N=4096, sparse | 59 | 0.742 ms | 32 | 4.878 ms |

For the crowded N=2048 case, the complete static-evaluation median improved
from 3.478 to 2.972 ms at 256 (1.17×). In contrast, the larger sparse scenes
took about 17–18% longer per complete evaluation at 32. Even 256 regressed
the moderate-overlap N=4096 case. Capacities at or above a tile's maximum
list length can still affect performance by changing workgroup storage, without
ever activating gather.

Actual Färlev training was compared on the performance commit against itself,
with default versus 32/128/256/512, 50 steps and six paired fresh-process trials
per setting. All loss trajectories and final PNGs were identical:

| Cutoff | N=512 mean speed ratio | N=2048 mean speed ratio |
| --- | ---: | ---: |
| 32 | 0.999× | 0.952× |
| 128 | 1.025× | 1.006× |
| 256 | 1.008× | 0.950× |
| 512 | 1.017× | 0.928× |

Small-workload wall timings varied substantially, so the small ratios should
not be treated as universal gains or regressions. The fits establish no
consistent training benefit from lowering the default. **The renderer retains
its existing adaptive 2048/4096 capacity on this M2.** Lower capacities are
useful for a known crowded workload, where the sweep supports roughly 64–128
at N=512 and 256–512 at N=2048.

The experimental predicate `n > SORT_CAPACITY || n > max(64, N/4)` reduced
dense sort costs and avoided the full-scene scans in the sparse examples,
but did not establish a broad end-to-end improvement. It is retained only in
ignored `output/perf/sort-density-probe/`, with its benchmark copy
`output/perf/perf-sort-density.js`; it adds no runtime flags or shader logic
to the reference implementation.

Raw results and a compact reduction are in
`output/perf/sort-sweep-{screen,validation-k8}.json`,
`output/perf/sort-density-validation{,-blur7}.json`, and
`output/perf/sort-threshold-summary.json`. The comparison figure is
`output/perf/sort-threshold-comparison.{svg,pdf,png}`. Reproduce a fixed-capacity
sweep with:

```sh
node tools/perf.js --sort=default,32,64,128,256,512,1024,2048 \
  --cases=128:2048:0.3,512:8192:0.04 --steps=50 --repeats=4 --gpu=1 --k=8
```

## Färlev training comparison

A follow-up compares `88d6d7c` with `1cc8e36` using the exact prepared target
and shared initialization from the Färlev 300-second suite: 512 × 288, N=512,
K=8, raw sigmoid RGB, learned alpha, color LR 0.1, and a constant 1 px filter.
The machine and runtime are the Apple M2/Dawn configuration above, on AC power.
Each trial starts a fresh Node process; revision order alternates. The requested
short comparison uses **50 steps and ten trials per revision**.

| Arithmetic mean | Before | Performance commit | Speed ratio |
| --- | ---: | ---: | ---: |
| 50 optimization steps | 195.30 ms | 187.90 ms | 1.039× |
| Complete Node process | 420.47 ms | 404.41 ms | 1.040× |
| Launch to optimization start | 153.70 ms | 138.50 ms | 1.110× |

All twenty trials produced identical loss trajectories and final PNGs, with
final floating-point PSNR 20.9371 dB. Mean per-step times were 3.906 and
3.758 ms; the standard deviations across trials were 0.266 and 0.299 ms.
The measured difference is small relative to timing variation; these data
demonstrate no dramatic training speedup for this workload.

The dense-tile gather cannot activate at N=512: no tile can exceed the default
2,048-entry bitonic capacity. This test also uses one renderer and a target
already at optimization size, so it exercises little device-cache reuse or
target-resizing work.

An initial four-pair 800-step comparison measured a 1.056× median optimization
ratio. A single 300-second pair instead completed 35,204 baseline steps versus
31,002 candidate steps (0.881×), with final PSNR 22.0174 versus 22.0185 dB.
These inconsistent sustained timings are not evidence of a reliable speedup.
A reverse-order long comparison was stopped when the requested budget changed
to 50 steps; its partial artifacts are not included in the short-run means.

The reusable command and plot instructions are in
[the benchmark README](../bench/README.md#comparing-windfoil-revisions-on-färlev).
Raw short-run measurements, source snapshots, per-trial artifacts, `summary.json`,
and `comparison.{svg,pdf,png}` are in ignored
`output/farlev-perf-mac-50steps-2026-09-30/`. The initial longer runs are in
`output/farlev-perf-mac-2026-09-30/`.

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
