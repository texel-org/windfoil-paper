# Windfoil

Reference code for the Windfoil preprint.

## Quick start

Requires Node 22+ and a WebGPU-capable GPU. From the repo root:

```sh
npm install && npm run fixtures:farlev && npm run demo:l2
```

Outputs go in `output/`. Options are in [docs/flags.md](docs/flags.md);
image sources and downloads are in [fixtures/README.md](fixtures/README.md).

| Task | Command |
| --- | --- |
| Fit an image | `npm run demo:l2 -- --target=photo.png` |
| Lines / plotter | `npm run demo:lines` / `npm run demo:plot` |
| SVG roundtrip | `npm run demo:roundtrip` |
| Rasterize SVG | `npm run render -- --svg=art.svg` |
| Browser demo | `npm run web:demo` |
| Tests | `npm test` |

## Python (optional)

Requires Python 3.11+ for JAX checks or CLIP:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r jax/requirements.txt -r requirements-clip.txt
```

For CLIP, run `npm run clip:server`, then `npm run demo:clip` in another terminal.
`npm run validate` requires JAX and Deno; `npm run test:oracle` also requires WebGPU.

## GPU benchmarks

See [bench/README.md](bench/README.md). Linux requires Vulkan;
NVIDIA containers need `graphics` driver capability. `llvmpipe` means CPU execution.
Do not clone recursively; use the submodule command in the benchmark setup.

[Apache 2.0](LICENSE) · [Credits](NOTICE) · [Citation](CITATION.cff)
