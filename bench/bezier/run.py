#!/usr/bin/env python3
"""In-process adapter around the pinned upstream Bézier Splatting model."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]
sys.path.insert(0, str(SCRIPT_DIR.parent))
from optimizer_state import prime_optimizer


PINNED_COMMIT = "9612a228bc0662e26e06840f2ed2b187bc366f8c"
GSPLAT_COMMIT = "bcca3ecae966a052e3bf8dd1ff9910cf7b8f851d"
DEFAULT_REPO = Path(__file__).resolve().parent / "upstream"
DEFAULT_GSPLAT = Path(__file__).resolve().parent / "gsplat"
# Sweep-runtime policy, not a demonstrated ceiling; raise via WF_BENCH_MAX_N.
MAX_N = int(os.environ.get("WF_BENCH_MAX_N", "4096"))
SAMPLES_PER_STRAND = 64
BOUNDARY_STRANDS = 2
AREA_STRANDS = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", default="l2", choices=("l2", "clip"))
    parser.add_argument("--target")
    parser.add_argument("--n", type=int, required=True)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--steps", type=int)
    budget.add_argument("--seconds", type=float)
    parser.add_argument("--schedule-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--keep-upstream", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def skip(args: argparse.Namespace, kind: str, reason: str) -> None:
    write_json(Path(args.out) / "result.json", {
        "schema_version": 1,
        "status": "skipped",
        "skip": {"kind": kind, "reason": reason},
        "engine": {"name": "bezier-splatting", "environment": "bezier", "backend": "cuda-gsplat"},
    })


def verify_checkout(repo: Path, commit: str, label: str) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
    ).strip()
    if head != commit:
        raise RuntimeError(f"{label} HEAD is {head}; expected pinned {commit}")
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if dirty:
        raise RuntimeError(f"{label} checkout has tracked changes; use the unpatched pinned revision")


def verify_upstream(repo: Path) -> None:
    if not (repo / "train_withsvg.py").is_file():
        raise RuntimeError("Bézier checkout missing; initialize the submodule or set BEZIER_SPLATTING_DIR")
    verify_checkout(repo, PINNED_COMMIT, "Bézier Splatting")


def verify_gsplat(repo: Path) -> None:
    if not (repo / "gsplat" / "__init__.py").is_file():
        raise RuntimeError("gsplat checkout missing; initialize bench/bezier/gsplat")
    verify_checkout(repo, GSPLAT_COMMIT, "gsplat")
    try:
        package = importlib.import_module("gsplat")
        origin = Path(package.__file__).resolve() if package.__file__ else None
        if origin is None or repo not in origin.parents:
            raise RuntimeError(f"imported gsplat from {origin}, expected editable install from {repo}")
        for module in ("project_gaussians_2d_scale_rot", "rasterize_sum", "rasterize"):
            importlib.import_module(f"gsplat.{module}")
    except Exception as error:
        raise RuntimeError(f"pinned gsplat CUDA extension is unavailable: {error}") from error


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as functional
    import torchvision
    from PIL import Image

    if not args.target:
        raise ValueError("--target is required for L2")
    if args.n < 1 or args.n > MAX_N:
        raise ValueError(f"Bézier Splatting benchmark N must be in [1, {MAX_N}]")
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.seconds is not None and args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    if args.schedule_steps < 1:
        raise ValueError("--schedule-steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Bézier Splatting benchmark requires CUDA")

    process_started = time.perf_counter()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(os.environ.get("BEZIER_SPLATTING_DIR", DEFAULT_REPO)).expanduser().resolve()
    gsplat_repo = Path(os.environ.get("BEZIER_GSPLAT_DIR", DEFAULT_GSPLAT)).expanduser().resolve()

    setup_started = time.perf_counter()
    verify_upstream(repo)
    verify_gsplat(gsplat_repo)
    token = hashlib.sha256(str(out).encode()).hexdigest()[:12]
    stem = f"windfoil_bench_{token}"
    input_dir = out / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_path = input_dir / f"{stem}.png"
    with Image.open(args.target) as source:
        target_image = source.convert("RGB")
        width, height = target_image.size
        target_image.save(target_path)
    background = [
        float(value) for value in np.asarray(target_image, dtype=np.float32).mean(axis=(0, 1)) / 255.0
    ]

    old_cwd = Path.cwd()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    trainer = None
    try:
        upstream = importlib.import_module("train_withsvg")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        schedule_steps = args.steps if args.steps is not None else args.schedule_steps
        upstream_args = argparse.Namespace(
            num_curves=args.n,
            mode="closed",
            image_name=target_path.name,
            data_name="windfoil_bench",
            save_imgs=False,
            num_samples=SAMPLES_PER_STRAND,
            # Closed mode lays out num_beziers*(degree+1) control points, and
            # num_beziers is fixed at 2 upstream, so degree is the only lever on
            # how complex a single curve's outline can be. 4 is upstream's own
            # value and stays the default; the env var exists to match a
            # competitor's parameter or control-point budget in a fairness run.
            bezier_degree=int(os.environ.get("WF_BEZIER_DEGREE", 4)),
            lr=0.01,
            iterations=schedule_steps,
        )

        def seed_all() -> None:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)

        def make_trainer():
            instance = upstream.SimpleTrainer2d(
                image_path=target_path,
                imagesvg_path=input_dir / "unused.svg",
                num_points=50000,
                model_name="GaussianImage_Cholesky_svg",
                iterations=schedule_steps,
                args=upstream_args,
            )
            instance.gaussian_model.background.copy_(
                instance.gaussian_model.background.new_tensor(background)
            )
            return instance

        seed_all()
        trainer = make_trainer()
        setup_ms = (time.perf_counter() - setup_started) * 1000

        warmup_started = time.perf_counter()
        trainer.gaussian_model.train()
        trainer.gaussian_model.train_iter(trainer.gt_image)
        trainer.gaussian_model.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        warmup_ms = (time.perf_counter() - warmup_started) * 1000
        shutil.rmtree(trainer.log_dir, ignore_errors=True)
        trainer = None
        torch.cuda.empty_cache()

        rebuild_started = time.perf_counter()
        seed_all()
        trainer = make_trainer()
        prime_optimizer(trainer.gaussian_model.optimizer, torch)
        setup_ms += (time.perf_counter() - rebuild_started) * 1000

        model = trainer.gaussian_model
        # EXPERIMENT ONLY, off unless WF_BEZIER_OPACITY_MODE is set. Upstream
        # hard-codes opacity_mode=1 next to a comment saying it "only works for
        # line-mode", while this benchmark runs closed mode; mode 0 is the flat
        # per-curve alpha the comment points at. Off-label either way, so it is
        # an env var rather than a flag: the benchmark's default stays the
        # unmodified published configuration.
        override = os.environ.get("WF_BEZIER_OPACITY_MODE")
        if override is not None:
            model.opacity_mode = int(override)
            width = 3 if model.opacity_mode == 1 else 1
            model._opacity = torch.nn.Parameter(
                torch.ones(model.num_curves, width, device=model._opacity.device)
            )
            for group in model.optimizer.param_groups:
                if group.get("name") == "opacity":
                    group["params"] = [model._opacity]
            prime_optimizer(model.optimizer, torch)
        model.train()
        remove_iter = 500
        remove_num = 0
        timeline: list[dict[str, float | int]] = []
        # Wall-clock epoch of the first optimiser step. The report places traces
        # on an axis whose origin is when the *command* was launched, so it needs
        # the absolute instant optimisation began -- interpreter start, torch and
        # gsplat import, device init, setup and warmup all sit to the left of it.
        optimize_start_epoch_ms = time.time() * 1000
        optimize_started = time.perf_counter()
        deadline = math.inf if args.seconds is None else optimize_started + args.seconds
        iteration = 0
        while args.steps is None or iteration < schedule_steps:
            iteration += 1
            step_started = time.perf_counter()
            loss, psnr, prediction = model.train_iter(trainer.gt_image)
            removed = 0
            densified = 0
            with torch.no_grad():
                if iteration % remove_iter == 0 and 1000 <= iteration < 9200:
                    if (iteration // remove_iter) % 2 == 1:
                        prune_mask = model.remove_curves_mask()
                        model.num_curves = prune_mask.sum()
                        removed = int((~prune_mask).sum().item())
                        remove_num += (~prune_mask).sum()
                        model.prune_beizer_curves(prune_mask)
                    elif remove_num > 0:
                        densified = int(remove_num.item() if hasattr(remove_num, "item") else remove_num)
                        position = upstream.sparse_coord_init(trainer.gt_image, prediction)
                        model.densify(remove_num, position, trainer.gt_image)
                        remove_num = 0
                model.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            now = time.perf_counter()
            elapsed_ms = (now - optimize_started) * 1000
            timeline.append({
                "step": iteration,
                "elapsed_ms": elapsed_ms,
                "step_ms": (now - step_started) * 1000,
                "loss_self": float(loss.item()),
                "psnr_self": float(psnr),
                "curves": int(model._control_points.shape[0]),
                "removed": removed,
                "densified": densified,
            })
            if now >= deadline:
                break
        optimize_ms = (time.perf_counter() - optimize_started) * 1000

        model.eval()
        # The plain forward pass is upstream's own evaluation render (its
        # denser-sampling and upscaling options are unused there too), so this
        # is the method's final image at its full quality. No final.svg:
        # closed mode has no exact vector form -- degree-4 curves rendered as
        # Gaussian splats -- and the pinned upstream converts only its cubic
        # line mode.
        final_started = time.perf_counter()
        with torch.no_grad():
            rendered = model()["render"].float()
        torch.cuda.synchronize()
        final_render_ms = (time.perf_counter() - final_started) * 1000

        output_started = time.perf_counter()
        image = rendered.squeeze(0).detach().cpu().clamp(0, 1)
        upstream.transforms.ToPILImage()(image).save(out / "final.png")
        (out / "trace.jsonl").write_text("".join(json.dumps(item) + "\n" for item in timeline))
        mse = functional.mse_loss(rendered, trainer.gt_image.float()).item()
        final_psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
        output_ms = (time.perf_counter() - output_started) * 1000

        best = max(timeline, key=lambda item: item["psnr_self"])
        completed = len(timeline)
        step_times = np.asarray([item["step_ms"] for item in timeline], dtype=np.float64)
        budget_mode = "steps" if args.steps is not None else "seconds"
        requested = args.steps if args.steps is not None else args.seconds
        result = {
            "schema_version": 1,
            "status": "ok",
            "engine": {
                "name": "bezier-splatting",
                "environment": "bezier",
                "backend": "cuda-gsplat",
                "bezier_degree": int(model.bezier_degree),
                "control_points_per_curve": int(model._control_points.shape[1]),
                "opacity_mode": int(model.opacity_mode),
                "opacity_alpha": {
                    "mean": float(torch.sigmoid(model._opacity).mean()),
                    "min": float(torch.sigmoid(model._opacity).min()),
                    "max": float(torch.sigmoid(model._opacity).max()),
                },
                "upstream_commit": PINNED_COMMIT,
                "gsplat_commit": GSPLAT_COMMIT,
                "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__),
            },
            "workload": {
                "loss": "l2",
                "primitive": "closed-degree4-curve",
                "n": args.n,
                "k": None,
                "opt_size": [width, height],
                "seed": args.seed,
                "init": {"kind": "native"},
                "engine_options": {
                    "degree": 4,
                    "samples_per_strand": SAMPLES_PER_STRAND,
                    "boundary_strands": BOUNDARY_STRANDS,
                    "area_strands": AREA_STRANDS,
                    "splats_per_shape": SAMPLES_PER_STRAND * (BOUNDARY_STRANDS + AREA_STRANDS),
                    "schedule_steps": schedule_steps,
                    "optimizer_state_primed": True,
                    "background": background,
                },
            },
            "budget": {"mode": budget_mode, "requested": requested, "clock": "optimize"},
            "progress": {
                "steps_completed": completed,
                "stop_reason": budget_mode,
                "deadline_overshoot_ms": max(0.0, optimize_ms - args.seconds * 1000) if args.seconds else 0.0,
            },
            "timing": {
                "process_internal_ms": (time.perf_counter() - process_started) * 1000,
                "setup_ms": setup_ms,
                "warmup_ms": warmup_ms,
                "optimize_ms": optimize_ms,
                "optimize_start_epoch_ms": optimize_start_epoch_ms,
                "final_render_ms": final_render_ms,
                "output_ms": output_ms,
                "steps_timed": completed,
                "step_ms": {
                    "mean": float(step_times.mean()),
                    "median": float(np.median(step_times)),
                    "p95": float(np.percentile(step_times, 95)),
                },
            },
            "self_reported": {
                "best_psnr": best["psnr_self"],
                "best_step": best["step"],
                "final_psnr": final_psnr,
            },
            "artifacts": {"final_png": "final.png", "final_svg": None, "trace": "trace.jsonl"},
        }
        write_json(out / "result.json", result)
    finally:
        if trainer is not None and not args.keep_upstream:
            shutil.rmtree(trainer.log_dir, ignore_errors=True)
        os.chdir(old_cwd)
        if sys.path and sys.path[0] == str(repo):
            sys.path.pop(0)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.loss != "l2":
        skip(args, "unsupported-loss", "Bézier Splatting adapter supports L2 only")
        return
    if args.n > MAX_N:
        skip(args, "cap", f"Bézier Splatting is capped at N={MAX_N}")
        return
    try:
        run(args)
    except Exception as error:
        write_json(out / "result.json", {
            "schema_version": 1,
            "status": "error",
            "engine": {"name": "bezier-splatting", "environment": "bezier", "backend": "cuda-gsplat"},
            "error": {"stage": "adapter", "message": str(error), "detail": traceback.format_exc()},
        })
        raise


if __name__ == "__main__":
    main()
