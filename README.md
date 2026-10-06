# Windfoil

Reference code for the [Windfoil paper](https://arxiv.org/abs/2610.02468).

Abstract:

> We present Windfoil, a GPU-friendly algorithm that treats rasterisation and differentiable vector graphics as two sides of the same problem by evaluating the box-filtered winding number of quadratic Bézier contours in closed form. We implement this in WebGPU, allowing it to run across a range of environments, including a web browser on a consumer laptop, and apply the system to real-time 2D rendering, high-resolution rasterisation for print media, and a differentiable renderer. We compare our renderer against Skia, a production-grade engine, and Slug, a popular GPU rasterisation algorithm for games and real-time applications, measuring fidelity to a reference box-filtered coverage. Our renderer matches the reference more closely than either, at performance comparable to Slug. We also compare our optimiser against DiffVG and Bézier Splatting, where it reaches equivalent or better reconstruction quality at a fraction of the per-step cost, scaling to tens of thousands of shapes at interactive rates.

## Live Demo

You can see a web demo of image fitting (L2) here:

https://texel-org.github.io/windfoil-paper/

## Quick start

Requires Node 22+ and a WebGPU-capable GPU. From the repo root, after cloning:

```sh
# first setup
npm install

# download test/fixture image
npm run fixtures:farlev

# run image fitting demo
npm run demo:l2
```

This will run the image fitting demo (Node.js + Dawn runtime) using a test image (![Färlev](https://commons.wikimedia.org/wiki/File:A_dip_in_the_road_in_F%C3%A4rlev.jpg)) and write results to the `./output/` folder.

## Commands

| Task            | Command                                 |
| --------------- | --------------------------------------- |
| Fit an image    | `npm run demo:l2 -- --target=photo.png` |
| Pen Plotter     | `npm run demo:plot`                     |
| SVG roundtrip   | `npm run demo:roundtrip`                |
| Browser L2 demo | `npm run web:demo`                      |
| Rasterize SVG   | `npm run render -- --svg=art.svg`       |
| Tests           | `npm test`                              |
| CLIP Loss       | (see [Python](#python) setup)           |

See [docs/flags.md](docs/flags.md) for a full list of options.

Some other examples:

```sh
# run L2 demo with other options
npm run demo:l2 -- --opt-size=1024 --n=1000 --steps=450

# anneal a scene-wide blur from 10px -> 1px
npm run demo:l2 -- --blur-start=10 --blur=1

# allow per-shape blur 10px -> 4px (cannot render this with SVG alone)
# also add some padding so the edges look a bit better
npm run demo:l2 -- --blur-start=10 --blur=4 --learn-blur --pad=0.1

# rasterise a (simple/strict) SVG at a specific size, 16-bit PNG, 300 DPI
npm run render -- --svg=output/print/final.svg --width=7016 --height=4961 \
  --depth=16 --dpi=300 --out=output/print-a2

# four-color pen plotter optimisation at 21cm canvas with a 1.5mm pen nib
npm run demo:plot -- --colors=#4287f5,#fa0ab6,#faeb19,#000000 --n=5000 --canvas=21 --pen=1.5

# similar as above, but learn a palette of 8 colors by optimisation instead
npm run demo:plot -- --color-count=8 --n=5000 --canvas=21 --pen=1.5 --palette-fidelity=1

# signed grey w/ add blending (fixed alpha), marks add or subtract light on black, fit through a smooth tonemap
npm run demo:l2 -- --n=2000 --blend=add --bg=black --tonemap=smooth \
  --style=raw --channels=gray --transfer=identity --alpha=1 --lr-scale=0.3

# additive RGB light: non-negative (softplus) capsules with a learned Reinhard exposure
npm run demo:l2 -- --primitive=capsule --n=5000 --blend=add --bg=black \
  --style=raw --channels=rgb --transfer=softplus --tonemap=reinhard --lr-scale=0.3

# run the demo through Deno WebGPU instead of node.js
WF_RUNTIME=deno npm run demo:l2 -- --opt-size=1024 --n=1000 --steps=450

# use Deno but with the Dawn backend rather than builtin (potentially faster)
WF_RUNTIME=deno-dawn npm run demo:l2
```

## Python

This is optional; but if you wish to verify the JAX oracle, or run a CLIP loss demo, you will need Python 3.11+.

```sh
# first setup
python3 -m venv .venv
.venv/bin/python -m pip install -r jax/requirements.txt -r requirements-clip.txt
```

To run the CLIP loss demo, first run `npm run clip:server` to start the loss server, then in another terminal
tab, run `npm run demo:clip`. On the first run, this will take a while as it will have to download the CLIP weights. The CLIP command
takes similar options as the above demos, but defaults to an optimisation size of 224px (CLIP's maximum).

Note that this demo can be quite slow on a laptop, with most of the cost going to CLIP and the boundary between WebGPU and Pytorch. A future release of a pure-WebGPU CLIP will allow this to run much faster.

Other Python-related commands:

- `npm run validate` requires JAX and Deno (CPU only tests/checks)
- `npm run test:oracle` compares JAX with WebGPU render

## Benchmarks & Validation

For benchmark comparisons against DiffVG and Bézier Splatting, see [bench/README.md](bench/README.md). Note that Linux will require Vulkan and NVIDIA containers need `graphics` driver capability. `llvmpipe` might suggest that CPU is being utilised instead of GPU. I've struggled to get WebGPU working with Modal but it works fine with most cheap Runpod GPUs. Also note: do not clone recursively, instead use the submodule command in the benchmark setup.

The filter kernel exploration resides in another branch [filter-kernels](https://github.com/texel-org/windfoil-paper/tree/filter-kernels), it also contains some scripts for generating delta/comparison outputs used in the paper.

Most of the code in this repo is optimised for differentiable rendering. The original repo, [windfoil-algorithm](https://github.com/texel-org/windfoil-algorithm/), contains benchmarks against Slug and Skia, and also uses acceleration structures and techniques for more optimal display and real-time rendering. Its fragment shaders use the same box-filtered winding integral. A future repo simply titled `windfoil` may one day be published that unifies all the code into a more user-friendly API, with additional features (e.g. strokes) not yet explored in the current research.

## Details & Disclosures

The original formulation of closed-form 2D coverage that runs efficiently and independently in a pixel shader was discovered by Claude Code (Fable 5) during an author-directed one-hour search for a new rasterisation algorithm. You can read more details in the [original renderer repo](https://github.com/texel-org/windfoil-algorithm), which also includes comparisons and tests against Skia and [Slug](https://terathon.com/blog/decade-slug.html). After releasing the rendering code, I soon realised that the closed-form nature of this algorithm would be well suited for differentiable rendering, whereas most differentiable renderers today use (rather slow) approximations, and do not generally optimise the same image that is intended for display. Since then, I've continued to develop "Windfoil" into a unified system, which encompasses both display/rendering and differentiability, and run a number of benchmarks against other renderers.

> ⚠️ The code here is experimental, primarily for research purposes, and as a companion to a preprint on Arxiv. It is not meant to be used for production; use at your own risk.

## Citation

If you'd like to cite the paper:

```bibtex
@misc{deslauriers2026windfoil,
  title         = {Windfoil: Closed-Form Coverage for Real-Time and Differentiable Vector Graphics},
  author        = {Matt DesLauriers},
  year          = {2026},
  eprint        = {2610.02468},
  archivePrefix = {arXiv},
  primaryClass  = {cs.GR},
  url           = {https://arxiv.org/abs/2610.02468}
}
```

[Apache 2.0](LICENSE) · [Credits](NOTICE) · [Citation](CITATION.cff)
