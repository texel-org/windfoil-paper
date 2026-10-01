# Benchmarks

Windfoil against DiffVG and Bézier Splatting on a shared init, a shared target
and an equal budget. Two commands: `npm run bench` produces raw artifacts,
`bench/report.py` reduces them to CSVs and figures. Everything generated lands
under the ignored `output/`.

This file describes the method and how to reproduce it. It quotes no results:
those depend on the GPU, the driver and the engines' builds, and are reported in
the paper.

## How the pieces fit

| File | Role |
| --- | --- |
| `plan.js` | A library: which suites, targets, shape counts and budgets make up each stage. It runs nothing. |
| `run.js` | `npm run bench`. The resumable orchestrator: expands the plan, fetches and verifies fixtures, records the machine, and runs each suite through `suite.js`. |
| `suite.js` | `npm run bench:suite`. One target matrix: builds the shared init, runs every engine cell, and scores each final PNG with one shared scorer. |
| `diffvg/run.py`, `bezier/run.py` | Adapters that run the comparison engines on that shared init and write the same `result.json`/`trace.jsonl` as Windfoil. |
| `optimizer_state.py` | Imported by both adapters. Takes one zero-gradient optimizer step and then restores the parameters and zeroes the state, so Adam's buffers are allocated before the timed loop rather than inside its first step. |
| `report.py` | Reduces a finished run to CSVs and figures. Deliberately not an npm script, so a long sweep never depends on torch or matplotlib being installed. |
| `gpu-verify.py` | Proves an engine really runs forward and backward on the GPU. |
| `pod-setup.sh` | One-shot, idempotent machine setup. |
| `schema.json` | Documentation of the `result.json` format. Nothing loads or validates against it. |

Windfoil itself is not reimplemented here: `suite.js` drives the same
`demos/cli.js` path the demos use.

Every cell ends with `final.png`, the engine's own renderer drawing its fitted
scene at full quality, and one scorer reads all of them: Windfoil's exact
render; DiffVG's at 16×16 samples per pixel (`--final-samples`), not the
2×2-sample render it optimises with, whose noise a perceptual metric can see;
Bézier Splatting's evaluation forward pass. Windfoil and DiffVG also write the
scene as `final.svg`, exactly; Bézier Splatting's closed mode has no exact
vector form (degree-4 curves rendered as Gaussian splats, and the pinned
upstream converts only its cubic line mode), so it writes none.

## Setup

None of what follows is required to run *part* of the benchmark: see
[Running less](#running-less). For the whole three-engine sweep, use a CUDA
machine on a **PyTorch image that includes `nvcc`** (DiffVG and gsplat compile
from source against the image's PyTorch, which setup inherits rather than
installs, so a bare CUDA image is not enough), created with
`NVIDIA_DRIVER_CAPABILITIES` including `graphics`. 24 GB of VRAM and 20 GB of
disk are comfortable headroom rather than minimums: only the 4096px stages are
memory-hungry, and a whole sweep writes well under a gigabyte. One script
installs Node, the JavaScript deps, fixtures, both CUDA engines and the report
tools, and repairs the Vulkan driver if the container exposes only its compute
half:

```sh
git clone https://github.com/texel-org/windfoil-paper.git /root/wf
cd /root/wf
git submodule update --init bench/bezier/upstream bench/bezier/gsplat
bash bench/pod-setup.sh
source ~/.bashrc      # node, npm and the Vulkan driver pin that setup just installed
```

Setup appends Node and the Vulkan pin to `~/.bashrc`. New shells read it by
themselves; the shell that ran the script has to `source` it once, or `npm` is
not found.

Do not clone with `--recursive`. At its pinned commit, Bézier Splatting carries
a nested `gsplat` gitlink with no `.gitmodules` entry, so recursion dies with
`No url found for submodule path 'gsplat'`. The two-path `git submodule update`
above is the working form; `bezier/run.py` checks both pins at run time.

Each requirements file belongs to one venv. The version conflicts between them
are intentional isolation, not something to reconcile:

| Requirements file | Venv | For |
| --- | --- | --- |
| `jax/requirements.txt` | `.venv` | JAX oracle (`test:jax`, `test:oracle`) |
| `requirements-clip.txt` | `.venv` | CLIP loss server |
| `bench/diffvg/requirements.txt` | `.venv-diffvg` | DiffVG adapter |
| `bench/bezier/requirements.txt` | `.venv-bezier` | Bézier Splatting adapter |
| `bench/requirements.txt` | `.venv-bezier` | `report.py` (that venv already has CUDA torch) |

`requirements-clip.txt` pins `torch==2.11.0` for a local install.
`pod-setup.sh` deliberately does not use that pin: it builds `.venv` with
`--system-site-packages` and installs only `open_clip_torch` and `websockets`,
so the loss server runs on the CUDA build of torch the image already ships.

Then prove each engine really computes on the GPU. `torch.cuda.is_available()`
does not: DiffVG can import cleanly and render *forward* on the device, then die
in the first backward pass when its build lacks the GPU's arch.

```sh
.venv-diffvg/bin/python bench/gpu-verify.py --engine=diffvg
.venv-bezier/bin/python bench/gpu-verify.py --engine=bezier
```

Both must print `PASS: forward+backward on GPU`. For Windfoil, `env.json`'s
`vulkan_device` must name the NVIDIA card — `llvmpipe` there means every
Windfoil timing in the sweep is a CPU number.

## Run

```sh
npm run fixtures                    # fetch and verify every target (~30 MB)
npm run bench -- --dry-run          # plan and fixture check only
npm run bench -- --skip=clip        # the default sweep, without its CLIP stage
npm run bench -- --out=output/my-run --only=opt512,kodak --repeats=3
```

The engines run from the venvs setup builds, `.venv-diffvg` and `.venv-bezier`;
`DIFFVG_PYTHON` and `BEZIER_PYTHON` override them. A sweep runs for hours, so
start it inside `tmux` or under `nohup`.

`npm run fixtures` is optional — the sweep fetches whatever its stages need and
SHA256-verifies it before the first cell. Run it first anyway to get the
downloads out of the way, and to re-verify targets already on disk: the sweep
only auto-fetches *missing* files, and a corrupt one stops it instead.

Completed cells are reused, so an interrupted sweep resumes; `--force` reruns
everything. Other flags are `--skip`, `--env`
(`node,deno,deno-dawn,diffvg,bezier`), and `--seed`. The `clip` stage needs
`npm run clip:server` running, or `--skip=clip`; the sweep checks the loss
server is listening before it starts the first cell rather than failing an hour
in, since `clip` runs after the L2 stages.

### Running less

Nothing here has to be run whole. Stages are independent, results land cell by
cell, and the report works on whatever a run directory contains:

```sh
npm run bench -- --only=opt512                 # one image, three engines: the shortest complete comparison
npm run bench -- --skip=clip,big-image,kodak   # no CLIP server, no three-engine 4096px cells, not the long stage
npm run bench -- --only=opt512 --env=node      # Windfoil alone
npm run bench -- --skip=clip --env=node,bezier # leave out DiffVG, which is most of the wall clock
```

`--env=node` needs none of the CUDA engine setup — no PyTorch image, `nvcc`,
engine venvs or `pod-setup.sh` — only `npm install` and a working Vulkan driver.
Leaving an engine out of `--env` likewise removes the need to build it. Fixtures
are fetched per stage, so a run that skips `kodak` never downloads the Kodak
set. `bench/report.py --no-perceptual` drops the report's torch dependency, at
the cost of the SSIM, MS-SSIM and LPIPS columns.

### Stages

`npm run bench -- --dry-run` prints the exact suites and cells. The default
stages run cheapest-first, so an interrupted run still has everything except
Kodak. Nearly all of the wall clock is DiffVG, and a full sweep is a matter of
hours.

| stage | content |
| --- | --- |
| `opt512` | Färlev at 512px, N=256/512/4096, 800 steps, all three engines |
| `big-image` | Färlev at 4096px, N=256, all three engines, under two protocols: 800 steps, and a 60-second wall clock |
| `large-n` | Windfoil only: N=10,000/50,000 at 4096px, 800 steps |
| `clip` | Windfoil vs DiffVG at 128², N=256, one prompt, 800 steps |
| `kodak` | all 24 Kodak images at native 768×512, N=512, 800 steps, all three engines |

Optional stages, selected with `--only`:

| stage | content |
| --- | --- |
| `full-schedule` | Bézier's published 10,000-step protocol: Färlev at 512px (Windfoil and Bézier, N=512/4096) and four Kodak images (Bézier only, N=256/512) |
| `kodak-long` | the `kodak` protocol on a longer budget, timed from process launch: `--images` (one image by default), `--steps` (10,000 by default) or `--seconds` |
| `probes` | one-step resolution probes on Färlev at 512–4096px, N=256/4096 |

Why the stages look the way they do:

- **`kodak` uses all 24 images and one shape count.** The Windfoil−DiffVG
  difference is small next to its spread between images, so a subset can change
  its sign: sampling the corpus would not shorten the run so much as invalidate
  it. One shape count, because `opt512` already sweeps N. It is the longest
  stage and runs last; `--skip=kodak` leaves a much shorter sweep that keeps
  everything except the corpus quality claim.
- **`big-image` runs its target twice**, as two suites. The 800-step one is the
  equal-step protocol of every other L2 stage, at 4096px. The 60-second one asks
  what each engine reaches in equal time, where they complete very different
  numbers of steps. DiffVG's 800-step cell at this size is the stage's cost;
  `--skip=big-image` drops both.
- **`opt512` runs on Färlev alone**, a hard target. One image is one point on
  the difficulty range, so read its low-N result with that qualifier; `kodak`
  supplies the range, at one N.
- **`full-schedule` exists because 800 steps is not Bézier's protocol.** Its
  first prune/densify fires at step 1,000, which an 800-step run never reaches,
  so this stage is the like-for-like quality comparison against it. DiffVG is
  excluded there: its per-step cost makes a 10,000-step budget impractical.
- **Give the optional stages their own `--out`.** The report's `mean/<subject>`
  figures average every cell of that subject in the run directory, so a
  10,000-step Kodak cell sharing a directory with the 800-step `kodak` stage is
  folded into its mean.

Windfoil is benchmarked in one configuration, chosen once and applied
everywhere: **crisp** (a constant one-pixel box filter — exact pixel-area
antialiasing, not a blur schedule the other engines lack) and the **raw** colour
codec (free RGB plus alpha, matching what DiffVG optimises, so equal N means
equal colour parameters). That codec trains a colour's logit where DiffVG trains
the colour itself, so its colour learning rate is set for a comparable step in
colour rather than an equal nominal rate. Neither choice is about quality — over a corpus the
alternatives come out level — both are about removing an objection.

A single suite can be run directly, outside the plan:

```sh
npm run bench:suite -- --loss=l2 --target=fixtures/wikimedia/farlev-dip-in-road.jpg \
  --env=node,diffvg,bezier --n=256,512,4096 --opt-size=512 --steps=800
```

The alternatives are `bench:suite` flags, `--windfoil-variant=anneal` and
`--windfoil-style=anchor`; `npm run bench` always runs the plan's configuration
and rejects them.

## Comparing Windfoil revisions on Färlev

To compare a performance branch with an earlier commit on one machine, use the
original suite's `inputs/` directory (its prepared 512 × 288 target and shared
N=512, K=8 initialization):

```sh
node tools/perf-farlev.js \
  --inputs=output/pod-2026-09-29-rtx2000ada/farlev-300s/suites/farlev-300s/inputs \
  --baseline=88d6d7c --candidate=1cc8e36 --out=output/farlev-perf
python3 tools/perf-farlev-plot.py output/farlev-perf
```

Only Windfoil runs. The script snapshots each committed revision, shares the
installed Node dependencies, and launches each cell in a fresh Node process.
It reproduces the crisp raw-RGB/learned-alpha configuration and learning rates
of the original 300-second suite. No image download, resizing of the original
JPEG, or other engine is involved.

The default runs ten pairs of 50 steps, alternating which revision runs first.
Startup is measured from process launch to the first optimization step and
included in the plots. `--steps=800 --repeats=4 --seconds=300` adds a longer
comparison with the original time budget. The seconds budget counts optimization
only, matching the original; `--seconds-repeats=2` repeats the long comparison
with reversed order. `--steps` and `--repeats` adjust the fixed-step comparison;
the schedule stays pinned to the original 800 steps.
`--first=candidate` starts with the candidate revision for an independent
comparison in reverse order.

`--n=2048` tests a different shape count. If its `init-n2048-s1.json` exists
in the input suite, it is reused. Otherwise the script generates it once using
the same neutral initialization helper as `bench:suite`, with seed 1 and the
original background. Both revisions receive that exact file, saved under the
comparison's `inputs/`; the original suite is left intact.

To compare sort capacities, use the same commit for both revisions and set
`--baseline-sort=default --candidate-sort=256`. These options set
`WF_SORT_CAPACITY` independently in each child process. The synthetic
`tools/perf.js --sort=default,32,128,256,512,2048` sweep also reports actual
tile-list sizes, the fraction of tiles using gather, and exact image/loss/gradient
equivalence. See [the threshold findings](../docs/performance.md#sort-threshold-sweep)
before choosing a global cutoff.

`summary.json` reports arithmetic means, medians, timing variation, speed ratios,
step counts, final floating-point render PSNR, and exact equivalence of the
fixed-step PNGs and loss trajectories.
`measurements.json` records every run; each cell retains the CLI artifacts and
logs. The optional plotting script uses the existing benchmark report's
matplotlib dependency and writes SVG, PDF, and PNG. Use a new output directory
for each comparison; existing results are never overwritten. This measures a
single renderer's training workload; device-cache reuse and dense-tile overflow
gains depend on other workloads.

## Report

The sweep prints the exact command to paste when it finishes, naming an
interpreter that exists on this machine. It is one of:

```sh
.venv-bezier/bin/python bench/report.py output/my-run   # GPU machine: engine venv
.venv/bin/python bench/report.py output/my-run          # local: base venv
python3 bench/report.py output/my-run                   # deps installed globally
```

The dependencies live in a venv rather than the system Python, so a bare
`python3` usually is not it. `REPORT_PYTHON` overrides the choice. Whichever you
use needs `pip install -r bench/requirements.txt`; a missing one names itself
and the file to install.

Re-renders nothing, so it is cheap to re-run and can run on another machine.
`--no-perceptual` drops the torch dependency; `--format=png` and `--out` are
also accepted.

```text
report/
  cells.csv          one row per cell: timings, PSNR/MSE, SSIM, MS-SSIM, LPIPS,
                     and time-to-threshold read back from the traces
  comparison.csv     Windfoil against each competitor cell by cell, including
                     how long Windfoil took to reach the quality the competitor
                     *finished* at (`parity_speedup` -- the number to quote)
  meta.json          machine, GPU, arguments, tool versions
  figures/
    <stage>/<target>/  convergence.svg   mean PSNR against time from the first
                                         optimisation step, one curve per engine
                       00-target.png  <engine>-<variant>-n<N>.png/.svg
    mean/<subject>/    the same figure averaged per subject -- farlev, kodim
```

There is one figure. Its clock starts at each engine's first optimisation step,
so interpreter start, library import, device init and warm-up are excluded —
they belong to the stack an engine ships in, not to the method (`cells.csv`'s
`startup_ms` has them). The axis is linear and uncropped: every curve ends when
its budget does, so a fast engine is a near-vertical line at the left and a
slow one shows how far it gets in the same seconds. A suite that sweeps
several N writes one file per shape count, `convergence-n<N>`, because a curve
averaged over 256 and 4,096 shapes describes nothing; a single-N suite keeps
the bare name. With `--repeats` above 1, the repeats of a cell are averaged.

Each target's own renders sit beside the figures that describe it, so nothing
has to be dug out of `cells/<env>/<variant>/n<N>/...`. The `mean/` directories
hold figures only: an average over a corpus has no single render.

One averaging rule worth knowing: every cell contributes at every instant.
Before its first step it holds the quality that step produced, and after its
last it holds its final quality, so the mean is over the same cells along the
whole axis rather than a curve whose membership changes under it.

## Output and caveats

```text
output/<run>/
  plan.json  state.json  env.json          # what ran, how far it got, on what
  suites/<stage>/<name>/
    config.json  results.jsonl  gpu.csv
    inputs/target.png  inputs/init-n<N>-s<seed>.json
    cells/<env>[/<variant>]/n<N>/<budget>/r<repeat>/
      result.json  trace.jsonl  final.png  final.svg   # svg: Windfoil and DiffVG
  report/                                   # written by bench/report.py
```

`result.json` follows [schema.json](schema.json), which is documentation only:
nothing loads or validates against it. `optimize_ms` excludes setup,
warm-up, final render, metrics and file output; compare `process_ms` for
end-to-end time. MSE and PSNR are recomputed from each final PNG by one shared
scorer, so no engine grades its own homework.

Things that make cross-engine rows easy to over-read:

- **GPU clocks ramp.** An idle card sits far below boost, and the same cell
  rerun back to back gets faster purely as it warms. Compare A against B
  interleaved in one sweep, never against a number from an earlier session, and
  treat modest cross-session timing differences as unresolved.
- **A step is not equal work across engines.** Bézier needs many more steps for
  the same quality, so its per-step ratio *understates* the real gap;
  `comparison.csv`'s `parity_speedup` is the honest number.
- **`parity_speedup` is only defined where Windfoil gets there.** It is empty
  for a cell in which Windfoil never reaches the quality the competitor finished
  at, so a median over it covers only the cells that did. Quote that count
  beside it.
- **The 800-step budget stops before Bézier's first prune/densify at step
  1,000**, and its published Kodak numbers use the full 10,000-step schedule.
  Treat equal-step Kodak rows as equal-step only, and use `--only=full-schedule`
  for a like-for-like quality claim.
- **PSNR is the metric every engine optimises.** `cells.csv` also reports SSIM,
  MS-SSIM and LPIPS, which need not rank the engines the same way; read them
  together.
- **A trace's last point is not the final score.** The per-step PSNR comes
  from each engine's own training loss: DiffVG's is its 2×2-sample render
  where `final.png` uses 16×16, and Bézier Splatting's is a float image where
  `final.png` is 8-bit. Both sit a little off `psnr_db`; the trace is for the
  convergence figures and the time-to-threshold columns, `psnr_db` is the
  quality to quote.

DiffVG and Bézier cells are capped at N=4,096 (`--diffvg-max`, `--bezier-max`);
that is a sweep-runtime policy, not a demonstrated ceiling of either method.
Bézier is L2-only, and DiffVG CLIP defaults to a 512 cap (`--diffvg-clip-max`).

**DiffVG has a canvas-size limit.** diffvg counts its sample grid, width ×
height × samples², in a 32-bit `int`; past 2³¹ the count wraps negative and it
silently returns a background-coloured image with no error. DiffVG's scored
`final.png` is rendered at 8×8 samples per pixel — the highest count that keeps
every benchmark target under the limit; sampling noise falls as 1/samples, so
more would add nothing visible — which allows canvases up to about 16
megapixels. Its 2×2 training render allows far more. `diffvg/run.py` refuses a
render over the limit rather than score a blank frame; a larger target needs a
lower `--final-samples`.
