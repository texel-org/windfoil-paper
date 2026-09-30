# Windfoil reference code

Reference materials for the Windfoil preprint. This repo puts analytic rendering
and its VJP in the same WebGPU shader; the algorithm write-up and original demo
are at [texel-org/windfoil-algorithm](https://github.com/texel-org/windfoil-algorithm),
with the coverage derivation in its
[algorithm notes](https://github.com/texel-org/windfoil-algorithm/blob/main/docs/ALGORITHM.md).

The code is deliberately small: closed quadratic loops, L2 and CLIP demos, a
browser L2 demo, a JAX oracle, and comparison harnesses.

**Status: frozen reference.** This repository exists so the paper's results can
be inspected and reproduced. It is not an npm package and is not developed
further; after publication it only receives fixes that keep the paper's claims
reproducible.

## Install

Bring your own GPU machine or pod. This repo contains no provider configuration
or credentials.

Use Node 22+ (the default) or Deno 2 with a WebGPU-capable GPU. Python 3.11+
is optional and only needed for CLIP, JAX validation, and comparison renderers.

### JavaScript

For Node and the browser demo:

```sh
npm install
```

Deno-only users can skip `npm install`; Deno resolves the JavaScript packages on
first run. On macOS 13/14, use Node or Deno's built-in WebGPU. The published
Deno+Dawn addon requires macOS 15 or newer.

### Python / JAX

Base Python environment for the oracle and CLIP:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r jax/requirements.txt
```

### CLIP

Complete the Python/JAX step first, then install the CLIP loss server:

```sh
.venv/bin/python -m pip install -r requirements-clip.txt
```

### Scripts

| Script | Purpose |
| --- | --- |
| `demo:l2`, `demo:clip`, `demo:lines`, `demo:plot`, `demo:roundtrip` | Headless demos; Node by default |
| `render` | Strict fill-only SVG rasterizer |
| `clip:server` | CLIP loss server |
| `web:demo`, `web:build`, `web:preview` | Browser demo |
| `fixtures`, `fixtures:farlev` | Fetch and verify every declared target image; just the demos' default image |
| `bench`, `bench:suite` | The reference sweep; one target matrix |
| `perf`, `perf:image` | Focused renderer/optimizer and target-resizing measurements |
| `test`, `test:jax`, `check:deno`, `validate` | JavaScript, JAX, and Deno checks |
| `test:oracle` | WebGPU/JAX forward and VJP parity |

## Demos

### L2

L2 needs only the JavaScript install. No imagery is tracked in this
repository, so fetch the default target once: W. Carter's CC0 photograph
[*A dip in the road in Färlev*](https://commons.wikimedia.org/wiki/File:A_dip_in_the_road_in_F%C3%A4rlev.jpg),
14 MB from Wikimedia Commons, checked against a pinned size and SHA256 so every
run starts from the same bytes:

```sh
npm run fixtures:farlev
```

It lands in the git-ignored `fixtures/wikimedia/`. L2 then fits filled quadratic
shapes to it by default; `--target` takes any other PNG or JPEG:

```sh
npm run demo:l2
```

Common overrides and parameter sweeps:

```sh
npm run demo:l2 -- --target=photo.png --n=1024 --steps=800 --opt-size=256
npm run demo:l2 -- --target=a.png --target=b.png --n=64,512 --steps=200,800
npm run demo:l2 -- --target=photo.png --opt-size=max
```

L2 preserves the source aspect ratio: `--opt-size=N` sets its longest side,
while `max` uses the native dimensions. Sweeps can mix both forms, such as
`--opt-size=128,max`.

Shader and startup findings, measured gains, and reproduction commands are in
the [performance investigation](docs/performance.md).

### Output resolution

`--save-size=N|max` is the one output-resolution knob. It sets the longest
side of every raster except the canonical `final.png`, which always stays at the
optimization size — so a small, fast fit still yields a large, crisp image:

```sh
npm run demo:l2 -- --target=photo.png --n=1024 --opt-size=max --save-size=2048
```

That writes `final-2048px.png` beside `final.png`, re-rendered from the hardened
scene by the Windfoil shader itself rather than by redrawing the SVG. The filter
width is world-space, so it renders at `s = 1/f` for ~1px of antialiasing at the
export resolution instead of a blurred upscale. L2 keeps the source aspect
ratio; CLIP sizes are square and `max` means 224. It never downsamples below the
optimization size.

Headless demos can also write progress PNGs without affecting optimization
timing. They land in the run's `frames/` directory, are off by default, and use
the same `--save-size` resolution:

```sh
npm run demo:l2 -- --save-every=25 --save-size=max --save-blur=false
```

Crisp frames use `--save-blur=false`; the default retains the current annealed
filter.

Node is the default runtime. The same script selects either Deno host with one
environment variable:

```sh
WF_RUNTIME=deno npm run demo:l2
WF_RUNTIME=deno-dawn npm run demo:l2
```

Without Node or npm, run Deno directly (replace `l2` with `clip`, `lines`, or
`plot`):

```sh
WF_WEBGPU_BACKEND=wgpu deno run --unstable-webgpu -A --node-modules-dir=auto demos/cli.js l2
```

Each invocation writes
`output/YYYYMMDD-HHMMSS-{deno|deno-dawn|node}-{l2|clip|roundtrip}/`. Array-valued
targets, prompts, shape counts, or step counts create separate cells.
`--out=PATH` overrides the run directory.

### CLIP

CLIP requires both Python install steps above. Start the loss server in one
terminal, then run the shapes demo in another. It optimizes at 128×128 by
default:

```sh
npm run clip:server
npm run demo:clip -- --prompt="a hot air balloon festival"
npm run demo:clip -- --prompt="another prompt" --opt-size=192
npm run demo:clip -- --prompt="another prompt" --opt-size=max
```

For CLIP, `--opt-size=max` resolves to its native 224×224 input size.

`--clipag` swaps in [CLIPAG](https://arxiv.org/abs/2306.16805), a ViT-B/32
finetuned for perceptually aligned gradients, which tends to give cleaner
shapes than the stock OpenAI weights:

```sh
npm run demo:clip -- --prompt="a hot air balloon festival" --clipag
```

On first use the loss server downloads the checkpoint (about 1.3 GB) from
[Zenodo record 10446026](https://zenodo.org/records/10446026) into
`demos/clip/`, which is git-ignored. Those weights are published by Roy Ganz
under CC BY 4.0 and are not redistributed here. `--clip-weights=TAG` selects any
other `open_clip` pretrained tag.

### Lines

The JavaScript-only generic-autodiff example uses a custom round-capped line
model:

```sh
npm run demo:lines -- --target=photo.png --n=1000
```

Its custom parameterization is in
[`demos/lines/model.js`](demos/lines/model.js).

### Plot

The plot demo mimics a pen plotter: fixed-width pen marks on white paper, with
only geometry optimized and parameters in physical centimeters. `--primitive`
picks the marker:

- `line` (default) — straight round-capped strokes, each a trained center,
  angle, and length softly bounded to `[--min-len, --max-len]` centimeters;
- `point` — stipple discs the radius of the pen, trained only in position.

```sh
npm run demo:plot -- --target=photo.png --n=2000
npm run demo:plot -- --primitive=line --pen=0.5 --min-len=0.1 --max-len=1
npm run demo:plot -- --primitive=point --n=8000
```

`--canvas` is the longest canvas side in centimeters (default 18) and `--pen`
the pen width in millimeters. In line mode `--min-len` and `--max-len` bound the
stroke length in centimeters — a smooth squash keeps every stroke inside the
band no matter where the optimizer pushes it; point markers ignore both. The
parameterization is in [`demos/plot/model.js`](demos/plot/model.js).

By default the pen is black and the target is converted to grayscale in memory,
so mark density expresses tone. `--colors` names one or more pen colors as a
comma-separated hex list, e.g. `--colors=#ff00ff,#00ff00`; each marker is seeded
with the palette color nearest the target where it lands and keeps that color,
so a two- or three-pen plot lets each pen settle into the regions it fits best.
With `--colors` the target stays in color unless `--grayscale` is also passed.

A fixed pen expresses tone only as mark density, so the plot loss is
band-limited: the target is box-filtered to match the renderer's current
annealed filter, and the anneal settles at `--blur-floor` (default 2 pixels for
plot; other demos keep the crisp floor of 1) instead of sharpening fully.
Comparing equally filtered images makes local ink density track local tone.
`final.png` still renders crisp.

`final.svg` is sized in real `cm` — line markers as round-capped `<line>`
strokes, point markers as filled `<circle>` discs of the pen radius. Marks are
emitted in creation (paint) order rather than grouped by pen color, so the SVG
occludes exactly like the optimized render: overlapping opaque marks depend on
draw order, and grouping by color would resurface marks the fit had covered.

`--pad` (a fraction of the longest side, clamped to `[0, 0.5]`, default 0) fixes
edge artifacts: it expands the optimization canvas by that fraction on every
side, filling the border with the edge-clamped target, so marks at the frame
edge are fit like interior ones instead of piling into a rim against the
boundary. The raster is cropped back to the visible frame and the SVG clips to
it. It applies to every headless demo (`l2`, `clip`, `lines`, `plot`), and a pad
of 0 leaves the pipeline untouched.

### More flags

Palettes, primitives, blend modes, HDR tonemapping, raw linear color, learned
blur, background optimization, and every other option are in
[`docs/flags.md`](docs/flags.md).

### SVG roundtrip

The roundtrip demo reads a fill-only SVG, rasterizes its target with Windfoil at
`--opt-size`, translates the same anchors and quadratic controls for its initial
guess, and fits them back to the target with the fused RGB L2 loss:

```sh
npm run demo:roundtrip
npm run demo:roundtrip -- --svg=demos/roundtrip/star-nonzero.svg --opt-size=256 --steps=300
npm run demo:roundtrip -- --svg=demos/roundtrip/star-evenodd.svg --opt-size=256 --steps=300
```

The two bundled pentagrams have identical geometry and differ only in
`fill-rule`, so their centers exercise `nonzero` and `evenodd` differently.
Source style and the discrete fill-rule choice stay fixed while geometry is
optimized. The output directory contains the source, target, translated
initial state, final PNG/SVG, configuration, result summary, and loss trace.
The default translation is one-sixteenth of each viewBox dimension;
`--offset=x,y` overrides it in SVG curve units, and `--save-every=N` writes
crisp progress frames.

WebGPU and the JAX oracle use the same differentiable fold of box-averaged
winding for both rules. As a result, antialiasing at pixels that straddle a
self-intersection is the renderer's analytic approximation to pointwise SVG
parity; target and fit remain identical in the roundtrip objective.

### SVG render

The render demo runs the forward compute pass as a high-resolution SVG
rasterizer. With no arguments it renders 180 radial, 4-unit filled spokes at
1024×1024:

```sh
npm run render
npm run render -- --svg=output/example/final.svg --dimension=4096
npm run render -- --dimension=18000
npm run render -- --svg=art.svg --width=5008 --height=7087 --dpi=300 --out=output/art-render
npm run render -- --svg=art.svg --width=5008 --height=7087 --dpi=300 --depth=16 --out=output/art-render-16
npm run render -- --svg=output/series --width=1240 --height=1125 --scale-mode=letterbox --out=output/series-letterbox
npm run render -- --svg=output/series --width=1240 --height=1125 --scale-mode=expand --out=output/series-expand
```

`--dimension` sets the longest output side and defaults to 1024; the other side
preserves the SVG aspect ratio. Use `--width` and `--height` together for an
exact raster size; Windfoil fits the viewBox without distortion and centers it.
`--scale-mode` decides what fills the spare area when the raster aspect differs
from the viewBox. `expand` (the default) renders a wider or taller view of the
same scene, so geometry that the viewBox crops, such as a shape half outside the
frame, continues into the extra area, and any background color extends with it.
`letterbox` paints the spare area with `--letterbox-color` (default `#000000`):
bars on the sides or on the top and bottom, whichever the aspects require. In
letterbox mode the content offset is snapped to whole pixels, so the bars are
crisp and the content matches an unpadded render of the viewBox; a fractional
content edge blends by exact pixel coverage.
`--svg` also accepts a directory (every `*.svg` in it), or several files
repeated or comma-separated. A single file writes `render.png`; a batch writes
one `<name>.png` per SVG into the same output directory and parses every input
before rendering. `--dpi` writes a PNG `pHYs` print-resolution chunk
with `png-tools` and defaults to 300. PNG output defaults to 8-bit RGBA;
`--depth=16` retains the renderer's float readback as 16-bit samples and uses
`png-tools` for encoding. The 16-bit host canvas uses twice the memory of the
8-bit canvas (about 271 MiB rather than 135 MiB at 5008x7087), in addition to
temporary encoding storage. Outputs larger than one GPU pass (device
buffer-binding and dispatch limits, at most 4096px per side) are rendered as a
grid of chunks with shifted origins and composited on the CPU, so the output
size is bounded by host memory rather than the GPU. With `--debug`, each chunk
logs its progress. Coverage is analytic per pixel, so the chunked result is
identical to a single pass. `--chunk` overrides the chunk side length, mostly
for testing.
An SVG without a full-viewBox background rect produces a transparent PNG;
`--background=none` and `--background=transparent` request the same explicitly,
while `--background='#ffffff'` supplies an opaque fallback. An embedded
background rect takes precedence. By default the command is quiet and writes
only `render.png`; pass `--debug` to log progress and also retain `source.svg`
and `render.json`. The parser accepts unitless root sizes or matching
physical sizes such as `424mm` by `600mm`, an optional filled background rect,
and closed `M/L/H/V/Q/Z` paths with solid fills and either SVG fill rule
(`nonzero` when omitted);
fills are `#rrggbb` or integer `rgb(r,g,b)`, and custom `data-*` metadata
attributes are accepted on any element and ignored;
line segments are represented exactly as quadratics. It rejects strokes,
transforms, filters, CSS, references, and unknown elements or attributes before
touching the GPU.

Outputs are written to
`output/YYYYMMDD-HHMMSS-{node|deno|deno-dawn}-render/`. `WF_RUNTIME=deno` and
`WF_RUNTIME=deno-dawn` select the same alternate hosts as the other headless
demos.

### Browser

The browser demo uses Vite, so it requires Node/npm but not Python:

```sh
npm run web:demo
npm run web:build
npm run web:preview
```

Open the printed Vite URL in a WebGPU browser. The page supports shapes, lines,
image drop, and webcam L2 fitting. It has no CLIP path. It starts on the Färlev
image when `npm run fixtures:farlev` has fetched it, and otherwise waits for a
dropped image.

## GPU pods

Any rented NVIDIA GPU machine with an Ubuntu/PyTorch image works. Create it with
`NVIDIA_DRIVER_CAPABILITIES` including `graphics`: CUDA alone is not enough for
WebGPU, which on Linux means Vulkan, and a container exposing only the compute
half of the driver silently falls back to the `llvmpipe` CPU rasterizer:
correct pictures, meaningless timings.

`bench/pod-setup.sh` installs Node, the JavaScript deps, fixtures, both
comparison engines and the report tools, and repairs that driver when it is
broken. It is idempotent and is the source of truth for pod setup:

```sh
git clone https://github.com/texel-org/windfoil-paper.git /root/wf
cd /root/wf
git submodule update --init bench/bezier/upstream bench/bezier/gsplat
bash bench/pod-setup.sh
source ~/.bashrc      # node, npm and the Vulkan driver pin that setup just installed
npm run demo:l2
```

Setup appends Node and the Vulkan pin to `~/.bashrc`. New shells read it by
themselves; the shell that ran the script has to `source` it once, or `npm` is
not found.

Do not clone with `--recursive`: the pinned Bezier Splatting commit carries a
nested `gsplat` gitlink with no `.gitmodules` entry, so recursion fails. The
two-path `git submodule update` above is the working form.

For a Windfoil-only machine, `npm install` plus the Vulkan repair is enough;
skip the Python engines. Copy the ignored `output/` artifacts off the machine
before terminating it.

Serverless GPU containers that expose CUDA compute but not the NVIDIA
graphics/Vulkan capability cannot run this renderer; they suit a standalone CLIP
or JAX job only.

## Benchmarks

Windfoil against DiffVG and Bezier Splatting on a shared init, target and
budget. Two commands, everything under the ignored `output/`:

```sh
npm run fixtures                     # fetch + verify targets (~30 MB)
npm run bench -- --skip=clip         # the sweep, into output/<timestamp>-bench
.venv-bezier/bin/python bench/report.py output/<run>   # CSVs + SVG figures
```

The `clip` stage needs `npm run clip:server` running in a second shell; drop
`--skip=clip` once it is, and the sweep refuses to start otherwise. The sweep
finishes by printing the report command with an interpreter that exists on the
machine it ran on, since the report's dependencies live in a venv
(`.venv-bezier` after `bench/pod-setup.sh`). A sweep runs for hours, so start it
inside `tmux` or under `nohup`; completed cells are reused, so an interrupted
one resumes where it stopped.

None of it has to be run whole: `--only=opt512` is one image against all three
engines, `--env=node` runs Windfoil alone with no CUDA engine setup at all, and
`--skip` drops any stage. [`bench/README.md`](bench/README.md#running-less) has
the combinations.

`npm run bench -- --dry-run` prints the plan and checks fixtures without
running or downloading anything. A full sweep is a matter of hours, nearly all
of it spent waiting on DiffVG; `--skip=kodak` leaves a much shorter one. Results
are reported in the paper rather than here. The matrix uses the demo's Windfoil path
directly; `bench/` holds only the DiffVG and Bezier adapters, the
orchestration, and uniform metrics. See [`bench/README.md`](bench/README.md)
for setup, stages, output layout and the caveats that make cross-engine rows
easy to over-read. Long benchmark runs are intentionally not part of
validation.

## Validation

The full validation command needs Node, Deno, and the Python/JAX install. Run
the individual scripts from the table when validating only one path.

```sh
npm run validate
npm run test:oracle  # requires headless WebGPU; checks nonzero + evenodd against JAX
```

`jax/oracle.py` is the clear, dense autodiff oracle. `js/` contains only the
core differentiable renderer, models, preparation, color VJP, and optimizer.
Demo I/O and runtime utilities live in `demos/util/`; generated data lives only
under the ignored `output/`.

## Citation and license

If you use this code, please cite the paper; [`CITATION.cff`](CITATION.cff) has
the reference. The code is released under the [Apache License 2.0](LICENSE).
Third-party components and the fetched target images are credited in
[`NOTICE`](NOTICE).
