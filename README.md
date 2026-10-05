# Windfoil

Reference code for the [Windfoil preprint](https://arxiv.org/abs/2610.02468).

Abstract:

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

## GPU benchmarks

See [bench/README.md](bench/README.md). Linux requires Vulkan. NVIDIA containers need `graphics` driver capability. `llvmpipe` means CPU execution. Note: do not clone recursively; use the submodule command in the benchmark setup.

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
