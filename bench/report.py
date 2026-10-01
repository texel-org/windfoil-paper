#!/usr/bin/env python3
"""Summarize benchmark CSVs and export the paper's convergence SVG/PDF.

Usage: python3 bench/report.py RUN [RUN ...] [--out DIRECTORY]
Each RUN contains cells.csv and traces.csv exported by bench/csv.js.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

ENGINES = {
    'windfoil': ('Windfoil', '#2a78d6'),
    'diffvg': ('DiffVG', '#eb6834'),
    'bezier-splatting': ('Bézier Splatting', '#1baf7a'),
}
NUMBERS = {'n', 'opt_width', 'opt_height', 'repeat', 'seed', 'steps_completed',
           'startup_ms', 'optimize_ms', 'process_ms', 'ms_per_step', 'mse_rgb', 'psnr_db'}


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def load(runs, environment='node'):
    cells, traces = [], defaultdict(list)
    for run in runs:
        if not (run / 'cells.csv').is_file() or not (run / 'traces.csv').is_file():
            raise ValueError(f'{run}: missing CSVs; run node bench/csv.js {run}')
        for row in read_csv(run / 'cells.csv'):
            for key in NUMBERS:
                row[key] = float(row[key]) if row.get(key) else None
            row['_run'] = str(run)
            if row['engine'] == 'windfoil' and (
                row['environment'] != environment or row['variant'] not in ('', 'crisp')
            ):
                continue
            cells.append(row)
        for row in read_csv(run / 'traces.csv'):
            traces[(str(run), row['run_id'])].append(
                (int(row['step']), float(row['elapsed_ms']), float(row['psnr_db'])))
    for points in traces.values():
        points.sort(key=lambda p: p[0])
    return cells, traces


def cell_key(row):
    return tuple(row.get(k) for k in ('_run', 'suite', 'n', 'budget', 'repeat', 'seed'))


def comparisons(cells, traces):
    grouped = defaultdict(dict)
    for row in cells:
        if row['loss'] != 'l2':
            continue
        key = cell_key(row)
        if row['engine'] in grouped[key]:
            raise ValueError(f'duplicate engine cell: {key} {row["engine"]}')
        grouped[key][row['engine']] = row
    result = []
    for engines in grouped.values():
        w = engines.get('windfoil')
        if not w:
            continue
        points = traces.get((w['_run'], w['run_id']), [])
        for engine in ('diffvg', 'bezier-splatting'):
            other = engines.get(engine)
            if not other:
                continue
            target = other['psnr_db']
            parity = next((ms for _, ms, psnr in points if target is not None and psnr >= target), None)
            # The paper reports parity only for fixed-step protocols.
            if not w['budget'].startswith('steps-'):
                parity = None
            result.append({
                'source_run': w['_run'], 'suite': w['suite'], 'n': w['n'],
                'budget': w['budget'], 'repeat': w['repeat'], 'seed': w['seed'],
                'competitor': engine, 'windfoil_psnr_db': w['psnr_db'],
                'competitor_psnr_db': target,
                'delta_db': w['psnr_db'] - target if w['psnr_db'] is not None and target is not None else None,
                'windfoil_optimize_ms': w['optimize_ms'], 'competitor_optimize_ms': other['optimize_ms'],
                'parity_ms': parity,
                'parity_speedup': other['optimize_ms'] / parity if parity and other['optimize_ms'] is not None else None,
            })
    return result


COMPARISON_COLUMNS = ['source_run', 'suite', 'n', 'budget', 'repeat', 'seed', 'competitor',
                      'windfoil_psnr_db', 'competitor_psnr_db', 'delta_db',
                      'windfoil_optimize_ms', 'competitor_optimize_ms', 'parity_ms', 'parity_speedup']
SUMMARY_COLUMNS = ['source_run', 'stage', 'benchmark', 'opt_size', 'n', 'budget', 'protocol']
for engine in ENGINES:
    SUMMARY_COLUMNS += [f'{engine}_cells', f'{engine}_targets', f'{engine}_psnr_db',
                        f'{engine}_optimize_s', f'{engine}_steps']
for engine in ('diffvg', 'bezier-splatting'):
    SUMMARY_COLUMNS += [f'{engine}_delta_db', f'{engine}_parity_speedup',
                        f'{engine}_parity_reached', f'{engine}_parity_total']


def mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(values) if values else None


def summary(cells, compared):
    groups = defaultdict(list)
    for row in cells:
        if row['loss'] != 'l2':
            continue
        source = row['source_id'] or row['target_label']
        kodak = source.startswith('kodim')
        size = (int(max(row['opt_width'], row['opt_height'])), int(min(row['opt_width'], row['opt_height']))) if kodak else (
            int(row['opt_width']), int(row['opt_height']))
        key = (row['_run'], row['suite'].split('/')[0], 'kodak' if kodak else source,
               f'{size[0]}x{size[1]}', row['n'], row['budget'], row.get('protocol', ''))
        groups[key].append(row)
    result = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        out = dict(zip(SUMMARY_COLUMNS[:7], key))
        for engine in ENGINES:
            subset = [r for r in rows if r['engine'] == engine]
            out[f'{engine}_cells'] = len(subset)
            out[f'{engine}_targets'] = len({r['source_id'] or r['target_label'] for r in subset})
            out[f'{engine}_psnr_db'] = mean(subset, 'psnr_db')
            ms = mean(subset, 'optimize_ms')
            out[f'{engine}_optimize_s'] = ms / 1000 if ms is not None else None
            out[f'{engine}_steps'] = mean(subset, 'steps_completed')
        suites = {r['suite'] for r in rows}
        for engine in ('diffvg', 'bezier-splatting'):
            pairs = [c for c in compared if c['source_run'] == key[0] and c['suite'] in suites
                     and c['n'] == key[4] and c['budget'] == key[5] and c['competitor'] == engine]
            out[f'{engine}_delta_db'] = mean(pairs, 'delta_db')
            ratios = [p['parity_speedup'] for p in pairs if p['parity_speedup'] is not None]
            out[f'{engine}_parity_speedup'] = statistics.median(ratios) if ratios else None
            fixed = key[5].startswith('steps-')
            out[f'{engine}_parity_reached'] = len(ratios) if fixed else None
            out[f'{engine}_parity_total'] = len(pairs) if fixed else None
        result.append(out)
    return result


def convergence(cells, traces, out, suite=None, repeat=0, threshold=21.4):
    candidates = [r for r in cells if r['loss'] == 'l2' and r['source_id'] == 'farlev'
                  and r['n'] == 512 and r['opt_width'] == 512 and r['opt_height'] == 288
                  and r['budget'] == 'seconds-300' and r['repeat'] == repeat
                  and (suite is None or r['suite'] == suite)]
    if not candidates:
        print('No 300 s Färlev convergence cells; run --only=farlev-wallclock to export the plot.')
        return
    if len({(r['_run'], r['suite'], r['seed']) for r in candidates}) != 1:
        raise ValueError('multiple convergence suites; report them separately or select --plot-suite')
    if any(r['startup_ms'] is None or not traces.get((r['_run'], r['run_id'])) for r in candidates):
        raise ValueError('convergence needs startup timestamps and traces for every selected cell')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    muted = '#52514e'
    fig, axis = plt.subplots(figsize=(7.5, 4.6), facecolor='white')
    axis.grid(True, alpha=0.25, linewidth=0.6, color=muted)
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axis.spines[side].set_color(muted)
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(colors=muted, labelsize=9)
    for engine, (label, color) in ENGINES.items():
        row = next((r for r in candidates if r['engine'] == engine), None)
        if row is None:
            continue
        points = traces[(row['_run'], row['run_id'])]
        xs = [(row['startup_ms'] + ms) / 1000 for _, ms, _ in points]
        ys = [psnr for _, _, psnr in points]
        axis.plot([0, xs[0]] + xs, [ys[0], ys[0]] + ys, label=label, color=color, linewidth=2)
        at = next((x for x, y in zip(xs, ys) if y >= threshold), None)
        if at is not None:
            axis.plot([at], [threshold], marker='o', markersize=7, markerfacecolor='none',
                      markeredgecolor=color, markeredgewidth=1.6, linestyle='none', zorder=5)
        mark = next((x for x, (step, _, _) in zip(xs, points) if step == 800), None)
        if mark is not None:
            axis.axvline(mark, color=color, linestyle=(0, (5, 3)), linewidth=1.4, alpha=0.9)
        print(f'{label}: startup {row["startup_ms"] / 1000:.3g} s; '
              f'first {threshold:g} dB at {at} s; step 800 at {mark} s')
    axis.axhline(threshold, color=muted, linewidth=0.8, alpha=0.5)
    axis.set_xlim(left=0)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: '0' if v == 0 else f'{v:g} s'))
    axis.set_xlabel('Wall clock from launch', color=muted)
    axis.set_ylabel('PSNR (dB)', color=muted)
    handles, labels = axis.get_legend_handles_labels()
    handles += [Line2D([], [], marker='o', markerfacecolor='none', markeredgecolor='black',
                       markeredgewidth=1.4, linestyle='none', markersize=6),
                Line2D([], [], color=muted, linestyle=(0, (5, 3)), linewidth=1.4)]
    labels += [f'first reaches {threshold:g} dB', 'step 800']
    axis.legend(handles, labels, frameon=False, fontsize=9, loc='lower right', labelcolor=muted)
    fig.tight_layout()
    for suffix in ('svg', 'pdf'):
        fig.savefig(out / f'convergence-farlev-wallclock.{suffix}', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--environment', choices=['node', 'deno', 'deno-dawn'], default='node')
    parser.add_argument('--plot-suite')
    parser.add_argument('--plot-repeat', type=int, default=0)
    parser.add_argument('--threshold-db', type=float, default=21.4)
    args = parser.parse_args()
    runs = [r.resolve() for r in args.runs]
    if len(set(runs)) != len(runs):
        parser.error('each input run may be supplied only once')
    out = (args.out or runs[0] / 'report').resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        cells, traces = load(runs, args.environment)
        if not cells:
            raise ValueError('no completed cells in CSVs')
        compared = comparisons(cells, traces)
        table = summary(cells, compared)
        write_csv(out / 'comparison.csv', compared, COMPARISON_COLUMNS)
        write_csv(out / 'summary.csv', table, SUMMARY_COLUMNS)
        print('Benchmark / size / N / budget: PSNR dB | optimise seconds (Windfoil, DiffVG, Bézier)')
        for row in table:
            values = lambda key: ', '.join('—' if row[f'{e}_{key}'] is None else format(row[f"{e}_{key}"], ".2f" if key == "psnr_db" else ".3g") for e in ENGINES)
            print(f'{row["stage"]}/{row["benchmark"]} / {row["opt_size"]} / '
                  f'{row["n"]:g} / {row["budget"]}: {values("psnr_db")} | {values("optimize_s")}')
        convergence(cells, traces, out, args.plot_suite, args.plot_repeat, args.threshold_db)
    except ValueError as error:
        parser.error(str(error))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
