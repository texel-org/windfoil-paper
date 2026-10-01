# Benchmarks

Commands for running Windfoil, DiffVG, and Bézier Splatting comparisons.
Results and methodology belong in the paper; generated artifacts stay in `output/`.

## Setup

Windfoil alone needs the [root setup](../README.md#quick-start).
All three engines require a CUDA machine with a PyTorch image that includes
`nvcc` and NVIDIA `graphics` driver capability for Vulkan.

```sh
git clone https://github.com/texel-org/windfoil-paper.git /root/wf
cd /root/wf
git submodule update --init bench/bezier/upstream bench/bezier/gsplat
bash bench/pod-setup.sh
source ~/.bashrc
```

Do not clone recursively: the pinned upstream has a nested `gsplat` gitlink
without a `.gitmodules` entry. Setup installs dependencies and fixtures,
builds the engines, repairs the Vulkan driver if needed, and updates `~/.bashrc`.
Source it once in the shell that ran setup.

Keep the Python environments separate:

| Environment | Dependencies |
| --- | --- |
| `.venv` | `jax/requirements.txt`, `requirements-clip.txt` |
| `.venv-diffvg` | `bench/diffvg/requirements.txt` |
| `.venv-bezier` | `bench/bezier/requirements.txt`, `bench/requirements.txt` |

Pod setup inherits the image's CUDA PyTorch rather than installing the local
CLIP requirements' torch pin.

Verify forward and backward GPU execution before running comparisons:

```sh
.venv-diffvg/bin/python bench/gpu-verify.py --engine=diffvg
.venv-bezier/bin/python bench/gpu-verify.py --engine=bezier
```

Both must print `PASS: forward+backward on GPU`. Windfoil's `env.json`
`vulkan_device` must name the GPU; `llvmpipe` means CPU execution.

## Run

```sh
npm run fixtures
npm run bench -- --dry-run
npm run bench -- --skip=clip --out=output/my-run
```

Use `tmux` or `nohup` for long sweeps. Reuse the same `--out` to resume
completed cells; `--force` reruns them. Fixtures are fetched as needed;
`npm run fixtures` verifies existing files too. The CLIP stage requires
`npm run clip:server` in another shell, or `--skip=clip`.

### Running less

```sh
npm run bench -- --only=opt512 --env=node     # Windfoil only; no CUDA engines
npm run bench -- --only=opt512               # one target, all engines
npm run bench -- --skip=clip,big-image,kodak
npm run bench -- --skip=clip --env=node,bezier
```

`--only` and `--skip` select stages; `--env` selects
`node,deno,deno-dawn,diffvg,bezier`. Other run options:
`--repeats=N`, `--seed=N`, and `--out=PATH`.
`--help` lists stage names; `--dry-run` prints the exact matrix without running
or downloading. Stage definitions are in [plan.js](plan.js).
`kodak-long` also accepts `--images=1-24`, `--steps=N`, or `--seconds=N`.
Give optional stages their own output directory to avoid averaging different
budgets together in reports.

A custom single suite:

```sh
npm run bench:suite -- --loss=l2 --target=fixtures/wikimedia/farlev-dip-in-road.jpg \
  --env=node,diffvg,bezier --n=256,512 --opt-size=512 --steps=800
```

`bench:suite` accepts `--windfoil-variant=anneal` and
`--windfoil-style=anchor`; the planned sweep rejects these overrides.
`DIFFVG_PYTHON` and `BEZIER_PYTHON` override engine interpreters.
Bézier supports L2 only. The suite caps comparison shape counts via
`--diffvg-max`, `--bezier-max`, and `--diffvg-clip-max`.
DiffVG rejects sample grids that exceed its 32-bit count limit;
reduce the adapter's `--final-samples` for larger canvases.

## Report

The sweep prints a report command using an available interpreter:

```sh
.venv-bezier/bin/python bench/report.py output/my-run
```

Install `bench/requirements.txt` in the interpreter used for reports.
`REPORT_PYTHON` overrides the sweep's choice. Report options include
`--no-perceptual` (omit torch-based metrics), `--format=png`, and `--out=PATH`.
Reports read saved artifacts and can run on another machine.

Run metadata is in `plan.json`, `state.json`, and `env.json`.
Suites are under `suites/<stage>/<name>/`; cells contain `result.json`,
`trace.jsonl`, and final renders. SVGs are available for Windfoil and DiffVG.
Reports write CSVs and figures under `report/`.
The result format is described by [schema.json](schema.json).
