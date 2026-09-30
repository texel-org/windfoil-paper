"""Plot a tools/perf-farlev.js comparison. Requires matplotlib (bench/requirements.txt).

    python3 tools/perf-farlev-plot.py output/farlev-perf
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1]).resolve()
summary = json.loads((root / "summary.json").read_text())
rows = json.loads((root / "measurements.json").read_text())
colors = {"baseline": "#777777", "candidate": "#007F73"}
labels = {"baseline": "Before (" + summary["config"]["revisions"]["baseline"][:7] + ")",
          "candidate": "codex/perf (" + summary["config"]["revisions"]["candidate"][:7] + ")"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})
fig, (full, early, timings) = plt.subplots(1, 3, figsize=(14, 4.8),
                                         gridspec_kw={"width_ratios": [1.3, 1.1, 0.8]})
for label in colors:
    timed = [row for row in rows if row["label"] == label and row["mode"] == "seconds"]
    if not timed:
        timed = [row for row in rows if row["label"] == label and row["mode"] == "steps"][:1]
    for repeat, row in enumerate(timed):
        trace = [json.loads(line) for line in (Path(row["cell"]) / "trace.jsonl").read_text().splitlines()]
        xs = np.array([(row["startupMs"] + t["elapsedMs"]) / 1000 for t in trace])
        ys = -10 * np.log10([t["loss"] for t in trace])
        stride = max(1, len(trace) // 5000)
        indices = np.unique(np.r_[np.arange(0, len(trace), stride), len(trace) - 1])
        full.plot(xs[indices], ys[indices], color=colors[label], linewidth=1.5,
                  alpha=0.8, label=labels[label] if repeat == 0 else None)
        early.plot(xs[:800], ys[:800], color=colors[label], linewidth=1.5, alpha=0.8)
        crossed = np.flatnonzero(ys >= 21.4)
        if len(crossed):
            i = crossed[0]
            early.plot(xs[i], ys[i], marker="o", markersize=6, markerfacecolor="white",
                       markeredgecolor=colors[label])
    fixed = [row for row in rows if row["label"] == label and row["mode"] == "steps"]
    index = list(colors).index(label)
    times = [row["optimizeMs"] / 1000 for row in fixed]
    timings.scatter(np.linspace(index - 0.12, index + 0.12, len(times)), times,
                    color=colors[label], s=35, zorder=3)
    mid = np.median(times)
    timings.plot([index - 0.22, index + 0.22], [mid, mid], color=colors[label], linewidth=2)
for i in range(summary["config"]["repeats"]):
    pair = [next(row for row in rows if row["label"] == label and row["mode"] == "steps"
                 and row["repeat"] == i)["optimizeMs"] / 1000 for label in colors]
    timings.plot([0, 1], pair, color="#CCCCCC", linewidth=0.7, zorder=1)
for ax in (full, early):
    ax.set_xlabel("Wall clock from Node launch (seconds)")
    ax.set_ylabel("Training-loss PSNR (dB)")
    ax.set_xlim(left=0)
    ax.grid(alpha=0.15)
full.set_title("Same 300-second optimization budget" if summary["timed"]["baseline"] else "Fixed-step run")
full.legend(frameon=False, fontsize=9, loc="lower right")
early.set_title("First 800 steps; circle = first 21.4 dB")
early.axhline(21.4, color="#999999", linewidth=0.8, linestyle=":")
timings.set_title(f"800-step timing ({summary['config']['repeats']} pairs)")
timings.set_ylabel("Optimization time (seconds)")
timings.set_xticks([0, 1], ["Before", "codex/perf"])
timings.set_xlim(-0.4, 1.4)
timings.grid(axis="y", alpha=0.15)
fig.suptitle("Färlev · 512 × 288 · N=512 · K=8 · " + summary["config"]["machine"]["cpu"], fontsize=13)
note = f"Median fixed-step speedup: {summary['speedup']['fixedOptimize']:.3f}×. "
if summary["speedup"]["timedSteps"]:
    note += f"300-second step-count ratio: {summary['speedup']['timedSteps']:.3f}×. "
note += "Identical 800-step loss trajectories and PNGs." if summary["fixedLossTrajectoriesIdentical"] and summary["fixedOutputsIdentical"] else "See summary.json for output equivalence."
fig.text(0.5, 0.015, note, ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.045, 1, 0.95))
for suffix in ("svg", "pdf", "png"):
    path = root / f"comparison.{suffix}"
    fig.savefig(path, dpi=180, facecolor="white")
    print(path)
