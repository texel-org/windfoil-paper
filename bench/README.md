# Benchmarks

## Setup

For all three engines, use a CUDA/PyTorch image with `nvcc` and NVIDIA
`graphics` driver capability. From the repo root:

```sh
git submodule update --init bench/bezier/upstream bench/bezier/gsplat
bash bench/pod-setup.sh
source ~/.bashrc
```

Do not initialize recursively: the pinned upstream has a broken nested gitlink.
Keep the script's Python environments separate.

Verify GPU execution:

```sh
.venv-diffvg/bin/python bench/gpu-verify.py --engine=diffvg
.venv-bezier/bin/python bench/gpu-verify.py --engine=bezier
```

Both should print `PASS`. Windfoil's `env.json` must name the GPU, not `llvmpipe`.

## Run

```sh
npm run bench -- --skip=clip --out=output/my-run
```

Use `tmux` for long runs; reuse `--out` to resume. For CLIP, start
`npm run clip:server` separately and omit `--skip=clip`.
`--help` lists options; `--dry-run` prints the matrix.
Windfoil alone needs only the [root setup](../README.md#quick-start):

```sh
npm run bench -- --only=opt512 --env=node
```

## Report

```sh
.venv-bezier/bin/python bench/report.py output/my-run
```

Reports go under the run's `report/`. Give different protocols separate run directories.
