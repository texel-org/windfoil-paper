# Flag reference

Options for the headless demos (`demo:l2`, `demo:clip`, `demo:lines`,
`demo:plot`), beyond the basics in the [README](../README.md). Pass them after
`--`, as in `npm run demo:l2 -- --n=1024`. Most flags are parsed by the shared
runner in [`demos/util/run.js`](../demos/util/run.js) and apply to every demo;
the exceptions are noted. `demo:roundtrip` and `render` have their own small
option sets, described in the README.

- [Common flags](#common-flags)
- [Constrained palettes](#constrained-palettes)
- [Fixed palettes and primitives](#fixed-palettes-and-primitives)
- [Optimizing the background](#optimizing-the-background)
- [Blend modes](#blend-modes)
- [HDR linear light, learnable exposure and white point](#hdr-linear-light-learnable-exposure-and-white-point)
- [Raw linear color](#raw-linear-color---styleraw)
- [Scale knobs](#scale-knobs)
- [Learnable per-shape blur](#learnable-per-shape-blur)
- [Performance knobs](#performance-knobs)

## Common flags

`--target`, `--prompt`, `--n`, `--opt-size`, `--steps` and `--seconds` accept a
comma-separated list or may be repeated; every combination runs as its own cell
under the run directory.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--target=PATH` | `fixtures/wikimedia/farlev-dip-in-road.jpg` | L2 target image (PNG or JPEG). The default is fetched by `npm run fixtures:farlev`. |
| `--prompt=TEXT` | `a hot air balloon festival` | CLIP prompt. |
| `--n=N` | 512 (`plot`: 4000) | Number of primitives. |
| `--k=N` | 8 | Quadratic curves per loop (shape model only). |
| `--opt-size=N\|max` | 128 (`plot`: 256) | Longest side of the optimization canvas. `max` is the source size for L2 and 224 for CLIP. |
| `--steps=N` | 500 (`plot`: 800) | Step budget. |
| `--seconds=S` | — | Wall-clock budget instead of a step budget. |
| `--max-steps=N` | 1000000 | Step cap for a `--seconds` run. |
| `--schedule-steps=N` | the step budget | Length of the blur anneal. Set it for a `--seconds` run, whose step count is not known in advance. |
| `--seed=N` | 7 | Seed for initialization. Runs are deterministic for a given seed and device. |
| `--blur=PX`, `--blur-floor=PX` | 7, 1 (`plot` floor: 2) | Start and end width of the annealed box filter ([scale knobs](#scale-knobs)). |
| `--bg=COLOR` | mean target color | Background color. White when there is no target image (CLIP) and for `plot`. |
| `--pad=F` | 0 | Expand the optimization canvas by a fraction of the longest side, clamped to `[0, 0.5]`. |
| `--save-size=N\|max` | — | Longest side of `final-<N>px.png` and of progress frames. Never below the optimization size. |
| `--save-every=N` | — | Write a progress frame to `frames/` every N steps. |
| `--save-blur=true\|false` | `true` | Frames keep the current annealed filter, or render crisp. Requires `--save-every`. |
| `--out=PATH` | timestamped | Run directory. |
| `--quiet` | off | Suppress per-step logging. |
| `--benchmark` | off | Same as `--quiet`; passed by the benchmark harness. |
| `--warmup=0\|1` | 1 | Run one untimed step first so pipeline compilation stays out of the timings. |
| `--init=PATH` | — | Load a serialized initial state (shape model only). The benchmark harness uses it to give every engine the same starting scene. |
| `--loss=l2\|clip` | per command | Objective; see [primitives](#fixed-palettes-and-primitives). |
| `--augs=N` | 4 | CLIP: augmented views per step. |
| `--loss-url=URL` | `ws://127.0.0.1:8765` | CLIP: loss server address. |
| `--clipag` | off | CLIP: use the CLIPAG weights (see the README's CLIP section). |
| `--clip-weights=TAG` | `openai` | CLIP: `open_clip` pretrained tag, or `clipag`. |

## Constrained palettes

`--color-count=N` (N ≥ 2) makes the fit discover an `N`-color palette instead of
giving every shape a free color. A shared, trainable set of `N` colors and a
per-shape softmax over them are optimized *jointly* with the geometry, so the
result represents the target in exactly `N` colors that the optimizer chose; no
palette is supplied. Internally it is the default anchor color codec with the
eight fixed sRGB corners replaced by `N` learned, shared colors. The palette
itself is not seeded from the target (only per-shape assignments are, to break
symmetry). At the end each shape snaps to one palette entry, and the learned
palette is written to `config.json`.

```sh
npm run demo:l2   -- --n=2000 --color-count=6 --opaque   # flat 6-color poster
npm run demo:l2   -- --n=2000 --color-count=6            # translucent, blends bg
npm run demo:plot -- --primitive=point --n=15000 --color-count=5   # 5 discovered inks
```

`--opaque` makes every shape fully opaque, so the output is exactly `N` flat
colors; without it (`l2`, `lines`) shapes keep a trained opacity and blend the
background, so the `N` colors are bases rather than the final pixels. `plot` ink
is always opaque (`--translucent` opts into blended marks). `--color-count` and a
fixed `--colors` are mutually exclusive. It applies to `l2`, `lines`, and
`plot`; the default free-color path is unchanged.

Because tone can come from coverage on white, the best-fitting palette drifts
toward saturated primaries that do not look like the image's real colors.
`--palette-fidelity=W` (default 0) counteracts that: it adds a small regularizer
that pulls each learned color toward the perceptual (OKLab) centroid of the
target colors of the marks assigned to it. It compares only the `N` colors
against one pre-sampled color per mark, never the pixels, so its cost is
negligible. `W` is the pull's strength relative to the fit gradient, so it stays
in range across images and mark counts: `0` is the best raster fit, `~0.3` is
subtle, `~1` a balanced blend, `~3` a fully faithful, muted palette. OKLab
conversions come from `@texel/color`.

## Fixed palettes and primitives

`--colors=#0b1e3a,#e8dcc4` pins the palette instead of discovering it: every
element decodes to exactly one of the given colors (opacity still trains unless
`--opaque`), so `--colors=#ffffff --bg=black` is white translucent ink on black.
It applies to `l2`, `clip`, and `lines`; the `plot` model has its own pen
handling of the same flag.

`--primitive=shape|capsule|line|point` is `--loss`'s dual on the geometry axis:
`shape` is the quadratic-loop model, `capsule` a round-capped stroke, `line` a
sharp rectangular stroke, and `point` a disc. The raster commands accept all
four, so `clip --primitive=capsule` draws a prompt with strokes and
`lines --loss=clip --primitive=point` stipples one. `plot` offers `line` and
`point`.

## Optimizing the background

`--optimize-bg` trains the background color along with the scene, instead of
holding it at the fixed `--bg` (which stays the starting value). It needs no
renderer change: the composite is `image = sum(shapes) + bg·P`, where `P` is the
per-pixel transmittance, so `dL/d(bg)` is exactly the color gradient of a
full-canvas opaque shape drawn underneath everything. The fit prepends one and
reads its existing gradient. It works across `l2`, `clip`, `lines`, and `plot`
(for `plot` it trains the paper color); the trained color is written to
`config.json` and used for the final render and SVG. `--bg-lr` sets its learning
rate (default 0.01).

## Blend modes

`--blend` selects how the whole scene composites: `src-over` (default,
painter's order), `add` (`bg + Σ color·a`), `multiply`
(`bg · Π (1 − a·(1 − color))`), or `screen`
(`1 − (1 − bg) · Π (1 − a·color)`, multiply in complement space: additive light
that saturates at 1), with layer alpha `a = coverage · opacity` in every mode.
The same option is available directly on the renderer
(`Renderer.create(device, { …, blend })`), and every renderer in a run (frames,
crop, export) shares it.

The non-default modes are order-independent, which changes the machinery, not
just the math: the tile lists skip the painter-order sort entirely, and the
backward pass no longer reads the forward image. `add`'s adjoints are exact
per-shape constants with no `max(1−a, ε)` guard at all, while `multiply` and
`screen` reconstruct each factor's sibling product with a guarded divide
(`max(m, 1e-3)`). One consequence: src-over's transmittance decay rounds deep
layers' gradients to zero below the fixed-point resolution, whereas `add`
delivers full-strength gradients to every layer, so dense scenes pay real
gradient-write cost for gradients src-over was dropping. `add` starts brighter
than its target (sums are unclamped), so a dark `--bg` is the natural pairing;
`multiply` stays inside `[0, bg]` and pairs with a light one; `screen` pairs
with a dark background like `add` but cannot overshoot 1.

## HDR linear light, learnable exposure and white point

`--tonemap` composites the scene in unbounded linear light (meant for
`--blend=add`, where sums exceed 1 freely) and applies the operator only at the
loss boundary and on output. The image buffer, the composite kernels, and their
backward all stay linear; the chain rule rides in through `dLdImage`. Three
operators share that machinery:

- `reinhard`: `T(x) = k·x / (1 + k·x)`. Saturating but asymptotic: display 1.0
  is unreachable, so highlights compress toward gray.
- `reinhard-white`: `T(x) = R(k·x) / R(k·W)` with `R(u) = u/(1+u)`. The learnable
  white point `W` makes `T(W) = 1` exactly, and as `k → 0` the operator tends to
  the pure linear rescale `x/W`, so "no compression" is inside the family.
- `smooth`: `T(x) = s(k·x) / s(k·W)` with `s(u) = u/√(1+u²)`. Same white-point
  normalization, but odd and C-infinity on all of ℝ. The Reinhard pair has a
  pole at `x = −1/k` (and is refused for signed scenes at construction), while
  `smooth` maps negative light symmetrically; it is the operator for
  `--transfer=identity` scenes.

The exposure `k` (seeded by `--exposure`, stepped at `--exposure-lr`) and white
point `W` (`--white`, `--white-lr`) are learnable scene scalars, trained in log
space. Their gradients come from the fused L2 kernel and from the host chain on
the CLIP path, both checked against JAX by `npm run test:oracle`. Final values
land in `config.json` and every saved raster is mapped with them. Because `T`
saturates, add-mode fits stay stable at learning rates tuned for src-over. The
default `--tonemap=none` leaves the loss kernels unchanged.

## Raw linear color (`--style=raw`)

The anchor softmax exists to keep colors displayable and to fuse alpha for
src-over fits. Both jobs are moot when the scene composites unbounded linear
light. `--style=raw` stores each shape's color as its own parameter through an
elementwise transfer whose codomain *is* the range declaration:

- `--transfer=identity`: signed linear light (ℝ). The composite is exactly
  linear in the parameters; with an L2 loss the color subproblem is convex. In
  add mode a shape can subtract light as easily as add it.
- `--transfer=softplus`: nonnegative HDR ((0, ∞)), for add-mode scenes that
  should only emit light.
- `--transfer=sigmoid`: bounded ((0, 1)), the baseline for A/B runs against the
  anchor codec's dynamics.

`--channels=gray` trains one scalar per shape (decoded neutrally; the pullback
sums the rgb cotangents); `--channels=rgb` trains three. `--alpha` is a fixed
opacity (default 1: in add mode `a·c` is a redundant product, so the raw codec
trains the premultiplied quantity directly and the renderer drops the
alpha-gradient pipeline entirely) or `learned` for a sigmoid-bounded opacity
like the other codecs. `--color-lr` / `--alpha-lr` set the rates.

The codec declares its range to the renderer, which validates the whole chain at
construction instead of clamping at run time: multiply/screen require unit
range, the Reinhard operators require nonnegative light (signed scenes want
`smooth` or `none`), and a nonneg tonemap rejects a negative background at
upload. The reference signed recipe is C-infinity end to end (linear composite,
identity codec, poleless odd tonemap, no clamp or guard between parameters and
loss):

```sh
npm run demo:l2 -- --blend=add --bg=black --tonemap=smooth \
  --style=raw --channels=gray --transfer=identity --alpha=1
```

## Scale knobs

`--lr-scale` multiplies every parameter group's learning rate. The models'
defaults are tuned against src-over; the order-independent blends deliver
unattenuated gradients whose effective step grows with overlap density, so dense
add/screen scenes (high `--n`, large canvases) want `0.2–0.5` to stay monotone.

Blur defaults are resolution-aware above a 512px reference: with no explicit
`--blur`/`--blur-floor`, the anneal (and learned-blur init) covers the same
fraction of a 2048px canvas as of a 512px one. Explicit values stay absolute,
and canvases at or below 512px use the defaults in the table above.

Scaling a fit up is a budget question, not just a schedule question: hold
shapes-per-pixel roughly constant (`--n` ∝ canvas area) and start learned blur
near the init grid's cell size (`≈ 0.6 · canvas / √n`). In smooth regions the
monotone modes (add/screen) prefer tilings of small crisp shapes over overlapping
soft ones, since they can only brighten and overlaps over-add; screen's
saturating product softens this and src-over avoids it entirely.

## Learnable per-shape blur

`--learn-blur` (shape and lines models) trains each primitive's filter size
instead of following the global blur anneal: every primitive starts at `--blur`
and its gradient decides how crisp it ends up, floored at `--blur-floor`. A
per-shape filter always overrides the annealed global one, so scenes without
learned blur are untouched. At the renderer level this is
`train: { blur: true }` (off by default): it widens the shape-gradient stride
from 4 to 6 to carry `(dL/dsx, dL/dsy)`, so default pipelines pay nothing. The
filter-size adjoints come from per-axis scaling homogeneity of the box integral
(`dL/dsx = −Σ qx·dL/dqx / sx`, a contraction of the piece gradients the backward
pass already computes), which keeps them exactly consistent with the JAX
oracle's autodiff.

## Performance knobs

### Freezing parameter groups

A renderer can train any subset of geometry, colour and alpha. Frozen groups are
specialized out of the backward pipeline, so their gradient code, buffers,
uploads, reductions and readbacks all disappear:

```js
const renderer = await Renderer.create(device, {
  width, height, maxShapes, maxPieces, maxCurves,
  train: { geometry: false, colour: true, alpha: false }, // default: all true
});
renderer.train;   // { geometry, colour, alpha, blur } -- frozen, readable
renderer.frozen;  // ['geometry', 'alpha']
```

`stepGpuLoss()` and `backward()` **omit a frozen group's array** (`undefined`)
rather than returning a zero-filled one, since returning zeros would keep the
transfer this exists to avoid. Consumers must branch on `renderer.train` before
touching `curveGrads` or `shapeGrads`. `shapeGrads` keeps its 4-floats-per-shape
stride whatever is frozen: a frozen lane is left at zero, so every model's
`pullback` sees one layout.

What freezing is worth depends on where the step's time goes. It helps when the
backward pass dominates, and very little on a small raster where binning
dominates instead.

### Dense tiles and the sort capacity

`bin_sort` restores painter order one workgroup per tile. It sorts ordinary
lists in workgroup memory. Larger lists use an ordered parallel gather that
reconstructs the same ascending shape IDs, removing the former serial comb-sort
cliff. Its cost depends on total shape count; the bitonic path depends on tile
occupancy.

Capacity is chosen twice: `sortCapacity` takes the largest power of two the
device's `maxComputeWorkgroupStorageSize` allows (2,048 entries on WebGPU's
guaranteed 16 KiB, 8,192 on a 48 KiB device), and `sortCapacityFor` then picks
the smallest specialization a scene needs from its mean tile occupancy. Too
small invokes the gather more often; too large uses extra workgroup storage
and can cost occupancy where tiles are many and short.

### A/B switches

Three environment variables select the form a kernel change replaced, so a
before/after claim can be reproduced by wall clock on the same hardware in the
same session:

| Variable | Effect |
| --- | --- |
| `WF_SCAN=serial` | Use the original single-lane tile-offset scan instead of the parallel scan. |
| `WF_FUSE_L2=0` | Run forward and the L2 gradient as two passes instead of the fused `forward_l2` kernel. |
| `WF_SORT_CAPACITY=N` | Force the `bin_sort` capacity (a power of two). Small capacities exercise the ordered gather even on sparse scenes. |

All three produce the same image and gradients as the defaults; only the timing
differs.
