# Windfoil reference code

Code for the Windfoil preprint: a differentiable WebGPU renderer, demos,
JAX oracle, and comparison harnesses. This is a frozen reference repository,
not an npm package; maintenance is limited to reproducibility fixes.

## Quick start

Requires Node 22+ and a WebGPU-capable GPU. Run commands from the repo root.

```sh
git clone https://github.com/texel-org/windfoil-paper.git
cd windfoil-paper
npm install
npm run fixtures:farlev
npm run demo:l2
```

Images are downloaded and verified, not tracked. See
[fixtures/README.md](fixtures/README.md) for sources and terms.
Generated files go in the ignored `output/` directory; use `--out=PATH` to override.

On Linux, WebGPU requires Vulkan. NVIDIA containers must expose
`NVIDIA_DRIVER_CAPABILITIES` including `graphics`; a CUDA-only container may
fall back to the `llvmpipe` CPU renderer, making GPU timings invalid.
For comparison engines, use the [benchmark setup](bench/README.md#setup).
**Do not clone with `--recursive`**: a pinned upstream submodule has a broken
nested gitlink. Initialize only the two paths listed in the benchmark setup.

## Demos

```sh
npm run demo:l2 -- --target=photo.png --n=1024 --steps=800 --opt-size=256
npm run demo:lines -- --target=photo.png --n=1000
npm run demo:plot -- --target=photo.png --n=2000 --pen=0.5
npm run demo:plot -- --target=photo.png --primitive=point --n=8000
npm run demo:roundtrip -- --svg=demos/roundtrip/star-evenodd.svg --steps=300
```

`--opt-size=N|max` sets the longest optimization side, preserving image aspect
ratio; `max` uses native dimensions. `final.png` stays at the optimization size.
`--save-size=N|max` adds a larger final render and sets progress-frame resolution;
`--save-every=N --save-blur=false` saves crisp frames.

Plot units: `--canvas` and `--min-len`/`--max-len` are centimeters;
`--pen` is millimeters. Plot SVGs use physical dimensions.
Roundtrip accepts closed fill-only SVGs; `--offset=x,y` sets the initial
translation in SVG units.

See [docs/flags.md](docs/flags.md) for other demo options.

### Browser

```sh
npm run web:demo
# Production build and local preview:
npm run web:build
npm run web:preview
```

Open the printed URL in a WebGPU browser. Drop an image or use the webcam;
the default image is available after `npm run fixtures:farlev`.

### CLIP and Python

Requires Python 3.11+. JAX checks and CLIP share `.venv`:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r jax/requirements.txt
.venv/bin/python -m pip install -r requirements-clip.txt
npm run clip:server
```

In another terminal:

```sh
npm run demo:clip -- --prompt="a hot air balloon festival"
```

CLIP uses a square canvas; `--opt-size=max` means 224.
Optional `--clipag` downloads a roughly 1.3 GB checkpoint into the ignored
`demos/clip/` directory on first use. Attribution is in [NOTICE](NOTICE).

### SVG rasterizer

```sh
npm run render -- --svg=art.svg --dimension=4096
npm run render -- --svg=art.svg --width=5008 --height=7087 --dpi=300 --depth=16
```

`--svg` accepts a file, directory, or repeated/comma-separated files.
`--dimension` sets the longest side; `--width` and `--height` together set an
exact size. `--scale-mode=expand|letterbox` controls aspect-ratio differences
(default `expand`); `--letterbox-color` defaults to black. Large renders are
chunked and limited by host memory. `--depth` is 8 or 16; `--dpi` defaults to 300.
`--debug` retains the source and render metadata and logs progress.

Supported SVGs contain solid fills, an optional background rect, and closed
`M/L/H/V/Q/Z` paths with `nonzero` or `evenodd` fill rules. Strokes, transforms,
CSS, filters, references, and other elements are rejected. Fills accept
`#rrggbb` or integer `rgb(r,g,b)`. Without a background rect, output is
transparent; `--background='#ffffff'` supplies a fallback.

### Deno

Deno 2 can run headless demos; Deno-only users can skip `npm install`.
The `deno-dawn` addon requires macOS 15+; on macOS 13/14 use Node or built-in Deno WebGPU.

```sh
WF_RUNTIME=deno npm run demo:l2
WF_RUNTIME=deno-dawn npm run demo:l2
# Without npm:
WF_WEBGPU_BACKEND=wgpu deno run --unstable-webgpu -A --node-modules-dir=auto demos/cli.js l2
```

## Tests and benchmarks

```sh
npm test                  # JavaScript tests
npm run test:jax          # requires .venv with JAX
npm run check:deno        # requires Deno
npm run validate          # all three above
npm run test:oracle       # WebGPU/JAX parity; requires GPU and JAX
```

`PYTHON` overrides the oracle's `.venv` interpreter.
`node tools/oracle/check.js output/oracle` retains parity-check artifacts.
Benchmark setup, run, and report commands are in [bench/README.md](bench/README.md).

Core code is in `js/`, demo utilities in `demos/util/`, and the JAX oracle in
`jax/oracle.py`.

## License and citation

[Apache 2.0](LICENSE). See [NOTICE](NOTICE) for third-party credits and
[CITATION.cff](CITATION.cff) for the paper reference.
