# Demo flags

Options for `demo:l2`, `demo:clip`, `demo:lines`, and `demo:plot`.
Pass flags after `--`: `npm run demo:l2 -- --n=1024`.
Shared parsing lives in [demos/util/run.js](../demos/util/run.js).
`render` and `demo:roundtrip` use separate options listed below.

## Common flags

`--target`, `--prompt`, `--n`, `--opt-size`, `--steps`, and `--seconds` accept
comma-separated lists or repeated flags; every combination runs separately.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--target=PATH` | `fixtures/wikimedia/farlev-dip-in-road.jpg` | L2 target image (PNG or JPEG). The default is fetched by `npm run fixtures:farlev`. |
| `--prompt=TEXT` | `a hot air balloon festival` | CLIP prompt. |
| `--n=N` | 512 (`plot`: 4000) | Number of primitives. |
| `--k=N` | 8 | Quadratic curves per loop (shape model only). |
| `--opt-size=N\|max` | 512 | Longest side of the optimization canvas. `max` is the source size for L2 and 224 for CLIP. |
| `--steps=N` | 500 (`plot`: 800) | Step budget. |
| `--seconds=S` | — | Wall-clock budget instead of a step budget. |
| `--max-steps=N` | 1000000 | Step cap for a `--seconds` run. |
| `--schedule-steps=N` | the step budget | Length of the blur anneal. Set it for a `--seconds` run, whose step count is not known in advance. |
| `--seed=N` | 7 | Seed for initialization. Runs are deterministic for a given seed and device. |
| `--blur=PX`, `--blur-floor=PX` | 7, 1 (`plot` floor: 2) | Start and end width of the annealed box filter (see notes below). |
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
| `--loss=l2\|clip` | per command | Objective; see primitives below. |
| `--augs=N` | 4 | CLIP: augmented views per step. |
| `--loss-url=URL` | `ws://127.0.0.1:8765` | CLIP: loss server address. |
| `--clipag` | off | CLIP: use the CLIPAG weights (see the README's CLIP section). |
| `--clip-weights=TAG` | `openai` | CLIP: `open_clip` pretrained tag, or `clipag`. |

## Model and color options

| Flag | Meaning |
| --- | --- |
| `--primitive=shape\|capsule\|line\|point` | Quadratic loop, round-capped stroke, rectangular stroke, or disc. Plot supports `line` and `point`. |
| `--colors=#0b1e3a,#e8dcc4` | Fixed palette. Quote values containing `#` when needed by your shell. |
| `--color-count=N` | Learn a palette of N ≥ 2 colors; L2, lines, and plot only. Mutually exclusive with `--colors`. Final shapes snap to palette entries. |
| `--palette-fidelity=W` | Learned palette's target-color regularization strength; default 0, around 1 for a balanced fit. |
| `--opaque` | Fix opacity at 1. Plot is opaque by default; `--translucent` enables blended marks. |
| `--grayscale` | Convert the target to grayscale. Plot does this by default unless `--colors` is given. |
| `--optimize-bg` | Train background color from its initial `--bg` value. |
| `--bg-lr=N` | Background learning rate; default 0.01. |
| `--blend=src-over\|add\|multiply\|screen` | Scene compositing; default `src-over`. Dark backgrounds suit add/screen; light ones suit multiply. |
| `--tonemap=none\|reinhard\|reinhard-white\|smooth` | Output/loss tonemapping; default `none`. Signed light requires `smooth` or `none`. |
| `--exposure=N`, `--exposure-lr=N` | Initial exposure and its learning rate. |
| `--white=N`, `--white-lr=N` | Initial white point and its learning rate for normalized tonemaps. |
| `--style=raw` | Train RGB/gray parameters directly instead of the default anchor color codec. |
| `--transfer=identity\|softplus\|sigmoid` | Raw color range: signed, nonnegative, or bounded to (0, 1). Multiply/screen require bounded colors; Reinhard requires nonnegative colors and background. |
| `--channels=gray\|rgb` | Raw color parameter count. |
| `--alpha=N\|learned` | Raw style opacity; default 1. |
| `--color-lr=N`, `--alpha-lr=N` | Raw style learning rates. |
| `--lr-scale=N` | Multiply all parameter learning rates. Dense add/screen scenes may need 0.2–0.5. |
| `--learn-blur` | Train per-primitive filter sizes instead of the global anneal (shape and lines models). Uses `--blur` for initialization and `--blur-floor` as the lower bound. |

Plot-specific options: `--canvas` (longest side in cm, default 18), `--pen`
(width in mm, default 0.45), `--min-len`/`--max-len` (line length bounds in cm,
defaults 0.1/1). Point markers ignore length bounds.

Without explicit blur values, filter widths scale with resolution above 512px.
Explicit `--blur`/`--blur-floor` values stay absolute.
Learned palettes, background, exposure, and white point are saved in `config.json`.

```sh
npm run demo:l2 -- --n=2000 --color-count=6 --opaque
npm run demo:l2 -- --blend=add --bg=black --tonemap=smooth \
  --style=raw --channels=gray --transfer=identity --alpha=1
```

## Renderer integration

Freeze parameter groups with `Renderer.create(device, { …,
train: { geometry: false, colour: true, alpha: false } })`.
Consumers must check `renderer.train`: `stepGpuLoss()` and `backward()` omit
frozen groups' arrays rather than returning zero arrays. `shapeGrads` keeps
its four-float stride; frozen lanes are zero. Learned blur gradients are
returned separately as `blurGrads`.

Dense src-over tile lists can exceed workgroup sort capacity and fall back to
a slower serial sort. Large shapes on small canvases can trigger this.

Optional kernel comparison switches:

| Environment variable | Effect |
| --- | --- |
| `WF_SCAN=serial` | Use the serial tile-offset scan. |
| `WF_FUSE_L2=0` | Run forward and L2 gradient as separate passes. |
| `WF_SORT_CAPACITY=N` | Force tile sort capacity; N must be a power of two. |

## SVG commands

`render`: `--svg=PATH` accepts files or a directory. `--dimension=N` sets the
longest side (default 1024); `--width=N --height=N` sets exact dimensions.
Other options: `--scale-mode=expand|letterbox`, `--letterbox-color=COLOR`,
`--background=COLOR`, `--depth=8|16`, `--dpi=N`, `--chunk=N`, `--debug`, `--out=PATH`.
An embedded background rect takes precedence over `--background`.

SVG inputs support solid fills, an optional background rect, and closed
`M/L/H/V/Q/Z` paths with either fill rule. Strokes, transforms, CSS, filters,
and references are rejected. Large raster outputs are limited by host memory.

`demo:roundtrip`: `--svg=PATH`, `--opt-size=N` (default 512), `--steps=N`,
`--offset=x,y` (initial translation in SVG units), `--save-every=N`, `--out=PATH`.

## Runtime and checks

Node is the default. Set `WF_RUNTIME=deno` or `WF_RUNTIME=deno-dawn` for Deno 2.
The Dawn addon requires macOS 15+; on macOS 13/14 use Node or built-in Deno WebGPU.
Without npm:

```sh
WF_WEBGPU_BACKEND=wgpu deno run --unstable-webgpu -A --node-modules-dir=auto demos/cli.js l2
```

`npm run test:jax` runs JAX checks; `npm run check:deno` checks Deno imports.
`PYTHON` overrides the parity check's `.venv` interpreter.
`node tools/oracle/check.js output/oracle` retains exchanged JSON files.
