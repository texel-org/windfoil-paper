#!/usr/bin/env python3
"""Reduce a finished sweep to CSVs, figures and a metadata record.

Step two of the benchmark. `npm run bench` writes raw artifacts and nothing
else; this reads them and re-renders nothing, so it is cheap to re-run and can
run on a different machine from the sweep.

    python3 bench/report.py output/<run>

Everything lands in `<run>/report/`:

    cells.csv        one row per completed cell, every metric
    comparison.csv   Windfoil against each competitor, cell by cell
    meta.json        machine, GPU, arguments, tool versions
    figures/
      <stage>/<target>/  convergence: mean PSNR against time from the first
                         optimisation step, one curve per engine (one file per
                         shape count when a suite sweeps N), and beside it
                         00-target and every cell's rendered png/svg, so reading
                         a result never means walking into cells/<env>/<n>/...
      mean/<target>/     the same figure averaged per subject: farlev, kodim, ...

Needs matplotlib, torch, pytorch-msssim and lpips (see bench/requirements.txt).
`--no-perceptual` drops the last three, and with them the SSIM/MS-SSIM/LPIPS
columns, for a sweep run without the comparison engines installed.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import platform
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

try:
    import matplotlib
except ImportError as error:  # pragma: no cover - depends on the environment
    raise SystemExit(
        f"{error}\ninstall the report dependencies:\n"
        "  python3 -m pip install -r bench/requirements.txt"
    ) from error

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

WINDFOIL_ENVIRONMENTS = {"node", "deno", "deno-dawn"}
# Slots 1-3 of the reference categorical palette: the only three that clear the
# all-pairs CVD and normal-vision floors in both light and dark. Marker shapes
# carry the same identity for print and forced-colour rendering.
ENGINES = {
    "windfoil": ("Windfoil", "#2a78d6", "o"),
    "diffvg": ("DiffVG", "#eb6834", "s"),
    "bezier-splatting": ("Bézier Splatting", "#1baf7a", "^"),
}
COMPETITORS = ("diffvg", "bezier-splatting")
# Pure white, not an off-white tint: these are exported to sit on a white page,
# where a near-white panel reads as a visible grey rectangle rather than as
# background. Sets the figure, the axes and the saved file alike.
SURFACE = "#ffffff"
MUTED = "#52514e"
THRESHOLDS = (25.0, 28.0, 30.0)


# ---------------------------------------------------------------- reading

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def discover_suites(root: Path) -> list[Path]:
    suites = sorted(
        path.parent
        for path in root.rglob("results.jsonl")
        if (path.parent / "config.json").is_file()
    )
    if not suites:
        raise SystemExit(f"{root}: no suites with config.json and results.jsonl")
    return suites


def trace_points(path: Path, engine: str) -> list[tuple[float, float]]:
    """(elapsed_ms, psnr_db) per completed step. The JS runner writes elapsedMs,
    the Python adapters elapsed_ms; Bézier reports its own psnr_self while the
    other two carry raw RGB MSE as `loss`."""
    points = []
    for row in read_jsonl(path):
        elapsed = row.get("elapsed_ms", row.get("elapsedMs"))
        if engine == "bezier-splatting":
            psnr = row.get("psnr_self")
        else:
            loss = row.get("loss")
            psnr = -10 * math.log10(loss) if isinstance(loss, (int, float)) and loss > 0 else None
        if elapsed is None or psnr is None or not math.isfinite(float(psnr)):
            continue
        points.append((float(elapsed), float(psnr)))
    return points


def first_at_or_above(points: list[tuple[float, float]], target: float) -> float | None:
    for elapsed, psnr in points:
        if psnr >= target:
            return elapsed
    return None


# ---------------------------------------------------------------- scoring

class Perceptual:
    """SSIM, MS-SSIM and LPIPS against the suite target. One instance holds the
    LPIPS network and a one-target cache, since a suite scores every cell
    against the same image."""

    def __init__(self, device: str | None) -> None:
        try:
            import torch
            from pytorch_msssim import ms_ssim, ssim
            import lpips
        except ImportError as error:
            raise SystemExit(
                f"{error}\ninstall the report dependencies:\n"
                "  python3 -m pip install -r bench/requirements.txt\n"
                "or re-run with --no-perceptual"
            ) from error
        self.torch = torch
        self.ssim, self.ms_ssim = ssim, ms_ssim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
        self._target_path: Path | None = None
        self._target = None

    def load(self, path: Path):
        from PIL import Image

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            tensor = self.torch.frombuffer(bytearray(rgb.tobytes()), dtype=self.torch.uint8)
            tensor = tensor.reshape(rgb.height, rgb.width, 3).permute(2, 0, 1)
        return tensor.unsqueeze(0).to(self.device).float() / 255.0

    def target(self, path: Path):
        if path != self._target_path:
            self._target_path, self._target = path, self.load(path)
        return self._target

    def score(self, render_path: Path, target_path: Path) -> dict[str, float | None]:
        target = self.target(target_path)
        render = self.load(render_path)
        if render.shape != target.shape:
            raise SystemExit(f"{render_path}: dimensions do not match {target_path}")
        with self.torch.no_grad():
            try:
                # MS-SSIM needs at least 161px per side for its five scales.
                multi = float(self.ms_ssim(render, target, data_range=1.0))
            except (AssertionError, ValueError, RuntimeError):
                multi = None
            return {
                "ssim": float(self.ssim(render, target, data_range=1.0)),
                "ms_ssim": multi,
                "lpips_alex": float(self.model(render, target, normalize=True)),
            }


def collect(root: Path, perceptual: Perceptual | None) -> list[dict[str, Any]]:
    rows = []
    for suite_root in discover_suites(root):
        config = json.loads((suite_root / "config.json").read_text(encoding="utf-8"))
        name = suite_root.relative_to(root).as_posix() if suite_root != root else suite_root.name
        # The sweep nests every suite under `suites/`; the prefix is noise in a
        # CSV column and an extra directory level under figures/.
        name = name[len("suites/"):] if name.startswith("suites/") else name
        target = config.get("target")
        target_path = suite_root / target if target else None

        for record in read_jsonl(suite_root / "results.jsonl"):
            if record.get("status") != "ok":
                continue
            workload = record.get("workload") or {}
            engine = (record.get("engine") or {}).get("name")
            timing = record.get("timing") or {}
            quality = record.get("quality") or {}
            budget = record.get("budget") or {}
            size = workload.get("opt_size") or [None, None]
            steps = (record.get("progress") or {}).get("steps_completed")
            optimize_ms = timing.get("optimize_ms")
            row: dict[str, Any] = {
                "suite": name,
                "run_id": record.get("run_id"),
                "engine": engine,
                "engine_label": ENGINES.get(engine, (engine, None, None))[0],
                "environment": (record.get("engine") or {}).get("environment"),
                "variant": workload.get("variant"),
                "style": (workload.get("engine_options") or {}).get("style"),
                "loss": workload.get("loss"),
                "target_label": workload.get("target_label"),
                "source_id": workload.get("source_id") or config.get("source_id"),
                "n": workload.get("n"),
                "opt_width": size[0],
                "opt_height": size[1],
                "budget": f"{budget.get('mode')}-{budget.get('requested')}",
                "repeat": workload.get("repeat"),
                "steps_completed": steps,
                "optimize_ms": optimize_ms,
                "process_ms": timing.get("process_ms"),
                "ms_per_step": optimize_ms / steps if optimize_ms and steps else None,
                "startup_ms": startup_ms(timing),
                "loss_end": (record.get("loss") or {}).get("end"),
                "mse_rgb": quality.get("mse_rgb"),
                "psnr_db": quality.get("psnr_db"),
            }

            if workload.get("loss") == "l2":
                relative = (record.get("artifacts") or {}).get("trace")
                trace = suite_root / relative if relative else None
                # A cell can be ok with no trace (a one-step probe that wrote
                # none, or a partially copied sweep); that costs its timing
                # columns, not the whole report.
                points = trace_points(trace, engine) if trace and trace.is_file() else []
                row["_trace"] = points
                if points:
                    best = max(psnr for _, psnr in points)
                    row["best_trace_psnr_db"] = best
                    row["ms_to_1db_of_best"] = first_at_or_above(points, best - 1.0)
                    for threshold in THRESHOLDS:
                        row[f"ms_to_{threshold:g}db"] = first_at_or_above(points, threshold)
                final = (record.get("artifacts") or {}).get("final_png")
                if perceptual and final and target_path and target_path.is_file():
                    row.update(perceptual.score(suite_root / final, target_path))
            artifacts = record.get("artifacts") or {}
            row["_renders"] = {
                kind: suite_root / artifacts[key]
                for kind, key in (("png", "final_png"), ("svg", "final_svg"))
                if artifacts.get(key) and (suite_root / artifacts[key]).is_file()
            }
            row["_target"] = target_path if target_path and target_path.is_file() else None
            rows.append(row)
    return rows


def compare(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Windfoil against each competitor within a cell. `parity_ms` is how long
    Windfoil took to reach the quality the competitor finished at -- a fixed dB
    bar silently drops every cell where neither engine crosses it, which on a
    hard corpus is most of them. The bar is the competitor's *scored* final
    PSNR, the same number the tables print: its training trace can sit below
    that (DiffVG optimises at 2x2 samples and is scored at more), and a bar
    taken from the trace would be easier to clear than the figure beside it."""
    keyed: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        if row.get("loss") != "l2":
            continue
        # Trace comparisons take the crisp variant: the anneal variant's early
        # blur depresses trace PSNR by design.
        if row["engine"] == "windfoil" and row.get("variant") not in (None, "crisp"):
            continue
        key = (row["suite"], row["n"], row["budget"], row["repeat"])
        keyed.setdefault(key, {})[row["engine"]] = row

    out = []
    for (suite, n, budget, repeat), engines in sorted(keyed.items(), key=lambda item: str(item[0])):
        windfoil = engines.get("windfoil")
        if not windfoil:
            continue
        for engine in COMPETITORS:
            other = engines.get(engine)
            if not other:
                continue
            final = other.get("psnr_db")
            if final is None:  # no scored final (a probe, or a partial copy): fall back to the trace
                final = (other.get("_trace") or [(None, None)])[-1][1]
            parity = first_at_or_above(windfoil.get("_trace") or [], final) if final else None
            out.append({
                "suite": suite,
                "n": n,
                "budget": budget,
                "repeat": repeat,
                "competitor": ENGINES[engine][0],
                "windfoil_psnr_db": windfoil.get("psnr_db"),
                "competitor_psnr_db": other.get("psnr_db"),
                "delta_db": subtract(windfoil.get("psnr_db"), other.get("psnr_db")),
                "windfoil_ms_per_step": windfoil.get("ms_per_step"),
                "competitor_ms_per_step": other.get("ms_per_step"),
                "step_speedup": divide(other.get("ms_per_step"), windfoil.get("ms_per_step")),
                "windfoil_optimize_ms": windfoil.get("optimize_ms"),
                "competitor_optimize_ms": other.get("optimize_ms"),
                "parity_target_db": final,
                "parity_ms": parity,
                "parity_speedup": divide(other.get("optimize_ms"), parity),
            })
    return out


def subtract(a: float | None, b: float | None) -> float | None:
    return a - b if a is not None and b is not None else None


def divide(a: float | None, b: float | None) -> float | None:
    return a / b if a and b else None


def startup_ms(timing: dict) -> float | None:
    """Wall clock from process spawn to the first optimiser step: interpreter
    start, module import, device init, model setup and warmup. The runner stamps
    the spawn and each adapter stamps the instant it begins optimising, both as
    absolute epochs, because the two are measured in different processes and
    have no shared monotonic clock. Missing on sweeps recorded before those
    stamps existed, which costs the figures their true origin and nothing else."""
    spawn = timing.get("spawn_epoch_ms")
    begin = timing.get("optimize_start_epoch_ms")
    if not isinstance(spawn, (int, float)) or not isinstance(begin, (int, float)):
        return None
    return max(0.0, begin - spawn)


# ---------------------------------------------------------------- figures

def time_label(value: float, _pos: object = None) -> str:
    """Log decades land on 10^n seconds; switch units late so "100 s" wins over
    "1.66667 min"."""
    if value <= 0:
        return "0"
    if value < 1:
        return f"{value * 1000:.3g} ms"
    if value < 1000:
        return f"{value:.3g} s"
    if value < 36000:
        return f"{value / 60:.3g} min"
    return f"{value / 3600:.3g} h"


TIME = FuncFormatter(time_label)
def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, alpha=0.25, linewidth=0.6, color=MUTED)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)


def figure(rows: int, columns: int, width: float, height: float):
    fig, axes = plt.subplots(
        rows, columns, figsize=(width * columns, height * rows),
        squeeze=False, facecolor=SURFACE,
    )
    flat = [axes[i // columns][i % columns] for i in range(rows * columns)]
    for axis in flat:
        axis.set_facecolor(SURFACE)
        style_axis(axis)
    return fig, flat


def save(fig: plt.Figure, path: Path) -> Path:
    # No figure title: these are meant to be embedded, where the surrounding
    # caption names the figure and a baked-in title is redundant at best and
    # contradictory once the caption is edited.
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def mean_by_time(curves, points=300):
    """Mean PSNR on a time grid from zero, log-spaced so the fast early rise is
    sampled as finely as the slow tail.

    Every cell contributes at every x: before its first step it holds the
    quality that step produced, after its last it holds its final quality, so
    the average is over the same cells along the whole axis."""
    lo = min(xs[0] for xs, _ in curves)
    hi = max(xs[-1] for xs, _ in curves)
    if not hi > lo > 0:
        return None
    grid = [0.0] + [lo * (hi / lo) ** (i / (points - 1)) for i in range(points)]
    return grid, [statistics.mean(interpolate(x, xs, ys) for xs, ys in curves) for x in grid]


def budget_note(rows) -> str:
    budgets = {str(row.get("budget", "")) for row in rows}
    if len(budgets) != 1:
        return ""
    mode, _, value = budgets.pop().partition("-")
    if mode == "steps" and value.isdigit():
        return f" ({int(value):,} steps per engine)"
    if mode == "seconds":
        return f" ({value} s per engine)"
    return ""


def convergence(rows, path: Path) -> Path | None:
    """Mean PSNR against time from each engine's first optimisation step, one
    curve per engine, on a linear axis.

    The clock starts at the first optimiser step, so interpreter start, library
    import, device init and warm-up are excluded: they belong to the stack an
    engine ships in, not to the method. Each curve ends when its budget does.
    The axis is linear and uncropped, so a fast engine is a near-vertical line
    at the left and a slow one shows how far it gets in the same seconds."""
    fig, (axis,) = figure(1, 1, 7.5, 4.6)
    drawn = 0
    for engine, (label, color, _) in ENGINES.items():
        cells = [row for row in rows if row["engine"] == engine and row.get("_trace")]
        if not cells:
            continue
        curves = [([ms / 1000 for ms, _ in row["_trace"]], [psnr for _, psnr in row["_trace"]])
                  for row in cells]
        series = mean_by_time(curves)
        if not series:
            continue
        axis.plot(series[0], series[1], label=label, color=color, linewidth=2)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    axis.set_xlim(left=0)
    axis.xaxis.set_major_formatter(TIME)
    axis.set_xlabel("Time from first optimisation step" + budget_note(rows))
    axis.set_ylabel("Mean PSNR (dB)")
    axis.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=MUTED)
    return save(fig, path)


def interpolate(x, xs, ys):
    """Linear in time, holding the final value once a run has ended -- a
    finished run's quality *is* its final quality, so this is a statement about
    the cell rather than an extrapolation."""
    if x >= xs[-1]:
        return ys[-1]
    index = bisect.bisect_left(xs, x)
    if index == 0:
        return ys[0]
    x0, x1, y0, y1 = xs[index - 1], xs[index], ys[index - 1], ys[index]
    return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def copy_renders(cells, directory: Path) -> int:
    """Put each cell's rendered output next to the figures that describe it, so
    reading a result never means walking down into cells/<env>/<variant>/..."""
    copied = 0
    target = next((row["_target"] for row in cells if row.get("_target")), None)
    if target:
        shutil.copyfile(target, directory / f"00-target{target.suffix}")
    for row in cells:
        variant = f"-{row['variant']}" if row.get("variant") else ""
        stem = f"{row['engine']}{variant}-n{row['n']}"
        for kind, source in (row.get("_renders") or {}).items():
            shutil.copyfile(source, directory / f"{stem}.{kind}")
            copied += 1
    return copied


def family(row) -> str | None:
    """Group targets by identity, not by suite: farlev at 512, 2048 and 4096 px
    are one subject, and kodim01..24 are one corpus."""
    source = row.get("source_id")
    if not source:
        return None
    return source.rstrip("0123456789") or source


def render_figures(rows, out: Path, suffix: str) -> list[Path]:
    """One figure kind everywhere. A group that sweeps N gets one file per
    shape count (`convergence-n<N>`), because a curve averaged over 256 and
    4,096 shapes describes nothing; a single-N group keeps the bare name."""
    written = []

    def group(subset, directory: Path):
        counts = sorted({row["n"] for row in subset if row.get("_trace")})
        if not counts:
            return []
        directory.mkdir(parents=True, exist_ok=True)
        made = []
        for n in counts:
            tag = f"-n{n}" if len(counts) > 1 else ""
            made.append(convergence([row for row in subset if row["n"] == n],
                                    directory / f"convergence{tag}.{suffix}"))
        return [path for path in made if path]

    # Trace comparisons take the crisp variant: the anneal variant's early blur
    # depresses trace PSNR by design.
    comparable = [row for row in rows
                  if row.get("loss") == "l2" and row.get("variant") in (None, "crisp")]

    for suite in sorted({row["suite"] for row in comparable}):
        cells = [row for row in comparable if row["suite"] == suite]
        directory = out / suite
        made = group(cells, directory)
        if made:
            copy_renders([row for row in rows if row["suite"] == suite], directory)
        written += made

    for name in sorted({family(row) for row in comparable if family(row)}):
        subset = [row for row in comparable if family(row) == name]
        written += group(subset, out / "mean" / name)
    return written


# ---------------------------------------------------------------- output

def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([
                "" if row.get(column) is None
                else f"{row[column]:.6g}" if isinstance(row[column], float)
                else row[column]
                for column in columns
            ])


def cell_columns(rows: list[dict[str, Any]], perceptual: bool) -> list[str]:
    columns = [
        "suite", "run_id", "engine_label", "environment", "variant", "style", "loss",
        "target_label", "n", "opt_width", "opt_height", "budget", "repeat",
        "steps_completed", "startup_ms", "optimize_ms", "process_ms", "ms_per_step", "loss_end",
        "mse_rgb", "psnr_db",
    ]
    if perceptual:
        columns += ["ssim", "ms_ssim", "lpips_alex"]
    columns += ["best_trace_psnr_db", *(f"ms_to_{t:g}db" for t in THRESHOLDS), "ms_to_1db_of_best"]
    return [column for column in columns if any(column in row for row in rows)]


def summarize(comparisons: list[dict[str, Any]]) -> None:
    for engine in COMPETITORS:
        label = ENGINES[engine][0]
        rows = [c for c in comparisons if c["competitor"] == label]
        if not rows:
            continue
        deltas = [c["delta_db"] for c in rows if c["delta_db"] is not None]
        speedups = sorted(c["parity_speedup"] for c in rows if c["parity_speedup"])
        print(f"\nvs {label} ({len(rows)} cells)")
        if deltas:
            ahead = sum(1 for value in deltas if value > 0)
            print(f"  quality  mean {sum(deltas)/len(deltas):+.2f} dB, ahead in {ahead}/{len(deltas)}")
        if speedups:
            median = speedups[len(speedups) // 2]
            print(f"  speed    to matched quality: median {median:.0f}x, "
                  f"range {speedups[0]:.0f}-{speedups[-1]:.0f}x "
                  f"({len(speedups)}/{len(rows)} cells reached it)")


def tool_versions(perceptual: Perceptual | None) -> dict[str, str | None]:
    versions = {"python": sys.version.split()[0], "matplotlib": matplotlib.__version__}
    if perceptual:
        versions["torch"] = perceptual.torch.__version__
        versions["torch_cuda"] = perceptual.torch.version.cuda
        versions["perceptual_device"] = perceptual.device
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="a sweep directory written by npm run bench")
    parser.add_argument("--out", type=Path, default=None, help="default: <run>/report")
    parser.add_argument("--format", default="svg", choices=["svg", "png", "pdf"])
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    parser.add_argument("--no-perceptual", action="store_true",
                        help="skip SSIM/MS-SSIM/LPIPS and their dependencies")
    args = parser.parse_args()

    root = args.run.resolve()
    out = (args.out or root / "report").resolve()
    (out / "figures").mkdir(parents=True, exist_ok=True)

    perceptual = None if args.no_perceptual else Perceptual(args.device)
    rows = collect(root, perceptual)
    if not rows:
        raise SystemExit(f"{root}: no completed cells")
    comparisons = compare(rows)
    figures = render_figures(rows, out / "figures", args.format)

    write_csv(out / "cells.csv", rows, cell_columns(rows, perceptual is not None))
    if comparisons:
        write_csv(out / "comparison.csv", comparisons, list(comparisons[0].keys()))

    env = {}
    env_path = root / "env.json"
    if env_path.is_file():
        env = json.loads(env_path.read_text(encoding="utf-8"))
    (out / "meta.json").write_text(json.dumps({
        "schema_version": 1,
        "run": root.name,
        "cells": len(rows),
        "suites": sorted({row["suite"] for row in rows}),
        "comparisons": len(comparisons),
        "perceptual": perceptual is not None,
        "report_host": platform.platform(),
        "report_tools": tool_versions(perceptual),
        "figures": sorted(path.relative_to(out).as_posix() for path in figures),
        "sweep": env,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"{len(rows)} cells from {len(set(row['suite'] for row in rows))} suites")
    summarize(comparisons)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
