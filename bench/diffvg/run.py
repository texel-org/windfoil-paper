#!/usr/bin/env python3
"""DiffVG adapter for the shared closed-quadratic-loop workload."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizer_state import prime_optimizer

# pydiffvg's ShapeGroup default. Stated here because the exported SVG has to
# declare the rule the fit was optimised under.
FILL_RULE = "evenodd"

# Sweep-runtime policy, not a demonstrated ceiling; raise via WF_BENCH_MAX_N.
MAX_N = int(os.environ.get("WF_BENCH_MAX_N", "4096"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", default="l2", choices=("l2", "clip"))
    parser.add_argument("--target")
    parser.add_argument("--prompt", default="a hot air balloon festival")
    parser.add_argument("--loss-url", default="ws://127.0.0.1:8765")
    parser.add_argument("--clip-weights", default="openai")
    parser.add_argument("--augs", type=int, default=4)
    parser.add_argument("--init", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--opt-size", type=int, help="square optimization size")
    parser.add_argument("--opt-width", type=int)
    parser.add_argument("--opt-height", type=int)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--steps", type=int)
    budget.add_argument("--seconds", type=float)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--samples", type=int, default=2)
    # The scored final.png is the fitted scene at full render quality, not the
    # training render: 2x2 samples is DiffVG's optimisation setting, and its
    # noise is visible to the perceptual metrics. 8x8 is the highest count that
    # keeps every benchmark target under diffvg's per-render sample limit (see
    # render()); sampling noise falls as 1/samples, so more adds nothing visible.
    parser.add_argument("--final-samples", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class ClipClient:
    def __init__(self, args: argparse.Namespace):
        from websockets.sync.client import connect

        self.socket = connect(args.loss_url, max_size=None)
        self.next_id = 1
        try:
            self.socket.send(json.dumps({
                "type": "config",
                "loss": "clip",
                "prompt": args.prompt,
                "augs": args.augs,
                "seed": args.seed,
                "model": "ViT-B-32-quickgelu",
                "pretrained": args.clip_weights,
            }))
            reply = json.loads(self.socket.recv())
            if reply.get("type") != "ready":
                raise RuntimeError(f"loss server: {reply}")
            self.device = reply.get("device")
        except Exception:
            self.socket.close()
            raise

    def grad(self, image, step: int):
        import numpy as np

        height, width = image.shape[:2]
        request_id = self.next_id
        self.next_id += 1
        self.socket.send(json.dumps({
            "type": "grad", "id": request_id, "step": step,
            "w": width, "h": height,
        }))
        self.socket.send(np.ascontiguousarray(image, dtype=np.float32).tobytes())
        header = json.loads(self.socket.recv())
        if header.get("type") == "error" or header.get("loss") is None:
            raise RuntimeError(header.get("message", "CLIP loss became non-finite"))
        data = self.socket.recv()
        gradient = np.frombuffer(data, dtype=np.float32)
        if gradient.size != height * width * 4:
            raise RuntimeError("loss server returned a gradient with the wrong size")
        return float(header["loss"]), gradient.reshape(height, width, 4)

    def close(self) -> None:
        self.socket.close()


def validate_args(args: argparse.Namespace) -> None:
    axes = args.opt_width is not None or args.opt_height is not None
    if args.opt_size is not None and axes:
        raise ValueError("--opt-size cannot be combined with --opt-width or --opt-height")
    if axes and (args.opt_width is None or args.opt_height is None):
        raise ValueError("--opt-width and --opt-height must be provided together")
    if args.opt_size is not None:
        args.opt_width = args.opt_size
        args.opt_height = args.opt_size
    elif not axes:
        args.opt_width = 128
        args.opt_height = 128

    if args.loss == "l2" and not args.target:
        raise ValueError("--target is required for L2")
    if not 1 <= args.n <= MAX_N:
        raise ValueError(f"DiffVG benchmark N must be in [1, {MAX_N}]")
    if args.k < 3 or args.opt_width < 1 or args.opt_height < 1:
        raise ValueError("--k must be at least 3 and optimization dimensions must be positive")
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.seconds is not None and args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    if args.samples < 1 or args.final_samples < 1 or args.warmup < 0 or args.augs < 0:
        raise ValueError("--samples and --final-samples must be positive; --warmup and --augs must be non-negative")


def validate_init(initial: dict, args: argparse.Namespace) -> None:
    size = initial.get("size")
    width = initial.get("width", size)
    height = initial.get("height", size)
    if (
        initial.get("n") != args.n
        or width != args.opt_width
        or height != args.opt_height
        or initial.get("k") != args.k
    ):
        raise ValueError("shared init n/dimensions/k does not match command line")
    expected = args.n * args.k
    params = initial.get("params", {})
    for key in ("ax", "ay", "cx", "cy"):
        if len(params.get(key, ())) != expected:
            raise ValueError(f"shared init params.{key} must contain {expected} values")
    if len(initial.get("colors", ())) != args.n or len(initial.get("alphas", ())) != args.n:
        raise ValueError("shared init colors/alphas length mismatch")
    if len(initial.get("background", ())) != 3:
        raise ValueError("shared init background must be RGB")


def make_state(initial, args, device, pydiffvg, torch):
    params = initial["params"]
    def coordinate(key):
        return torch.tensor(
            params[key], dtype=torch.float32, device=device
        ).reshape(args.n, args.k)

    anchors_x, anchors_y = coordinate("ax"), coordinate("ay")
    controls_x, controls_y = coordinate("cx"), coordinate("cy")
    points_x = torch.stack([anchors_x, controls_x], dim=2).reshape(args.n, 2 * args.k)
    points_y = torch.stack([anchors_y, controls_y], dim=2).reshape(args.n, 2 * args.k)
    points_x = points_x.detach().requires_grad_(True)
    points_y = points_y.detach().requires_grad_(True)
    points = torch.stack([points_x, points_y], dim=2)
    colors = torch.tensor(
        initial["colors"], dtype=torch.float32, device=device, requires_grad=True
    )
    alphas = torch.tensor(
        initial["alphas"], dtype=torch.float32, device=device, requires_grad=True
    )
    control_counts = torch.ones(args.k, dtype=torch.int32)
    stroke_width = torch.tensor(1.0, device=device)
    shapes = [
        pydiffvg.Path(
            num_control_points=control_counts,
            points=points[index],
            stroke_width=stroke_width,
            is_closed=True,
        )
        for index in range(args.n)
    ]
    groups = [
        pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([index]),
            fill_color=torch.zeros(4, device=device),
            use_even_odd_rule=FILL_RULE == "evenodd",
            stroke_color=None,
        )
        for index in range(args.n)
    ]
    lr_x, lr_y = geometry_lrs(args)
    optimizer = torch.optim.Adam([
        {"params": [points_x], "lr": lr_x},
        {"params": [points_y], "lr": lr_y},
        {"params": [colors], "lr": 0.02},
        {"params": [alphas], "lr": 0.03},
    ])
    return {
        "points_x": points_x,
        "points_y": points_y,
        "colors": colors,
        "alphas": alphas,
        "shapes": shapes,
        "groups": groups,
        "optimizer": optimizer,
    }


# diffvg counts its sample grid, width * height * samples_x * samples_y, in a
# C `int`. Past 2^31 the count wraps negative, no render kernel runs, and the
# image comes back as bare background with no error raised. Refuse before that
# point rather than score a blank frame; the bound leaves room for the further
# integer arithmetic the kernel does on the count.
MAX_SAMPLES_PER_RENDER = 2 ** 30


def render(state, background, args, seed, pydiffvg, torch, samples=None):
    samples = args.samples if samples is None else samples
    grid = args.opt_width * args.opt_height * samples * samples
    if grid > MAX_SAMPLES_PER_RENDER:
        raise RuntimeError(
            f"{args.opt_width}x{args.opt_height} at {samples}x{samples} samples per pixel is "
            f"{grid:,} samples, over diffvg's limit of {MAX_SAMPLES_PER_RENDER:,} per render; "
            "lower --samples or --final-samples"
        )
    rgba = torch.cat([state["colors"], state["alphas"].unsqueeze(1)], dim=1)
    points = torch.stack([state["points_x"], state["points_y"]], dim=2)
    for index in range(args.n):
        state["shapes"][index].points = points[index]
        state["groups"][index].fill_color = rgba[index]
    scene = pydiffvg.RenderFunction.serialize_scene(
        args.opt_width, args.opt_height, state["shapes"], state["groups"]
    )
    image = pydiffvg.RenderFunction.apply(
        args.opt_width, args.opt_height, samples, samples, seed, None, *scene
    )
    alpha = image[..., 3:4]
    return alpha * image[..., :3] + (1.0 - alpha) * background


def train_step(state, target, background, client, args, step, pydiffvg, torch, np) -> float:
    state["optimizer"].zero_grad(set_to_none=True)
    image = render(state, background, args, step, pydiffvg, torch)
    if args.loss == "l2":
        loss = ((image - target) ** 2).mean()
        loss.backward()
        value = float(loss.detach().item())
    else:
        alpha = torch.ones((args.opt_height, args.opt_width, 1), device=image.device)
        rgba = torch.cat([image, alpha], dim=2).detach().cpu().numpy()
        value, gradient = client.grad(rgba, step)
        cotangent = torch.from_numpy(np.ascontiguousarray(gradient[..., :3])).to(image.device)
        image.backward(gradient=cotangent)
    state["optimizer"].step()
    with torch.no_grad():
        state["colors"].clamp_(0.0, 1.0)
        state["alphas"].clamp_(0.0, 1.0)
    return value


# The fitted scene as an SVG in the fill dialect Windfoil's exporter writes and
# demos/render reads, so DiffVG's result is a vector deliverable like
# Windfoil's. It is exact: points are already in pixels, in DiffVG's
# [a0, c0, a1, c1, ...] order, and colours were clamped after every step.
def number(value: float) -> str:
    """Three decimals with trailing zeros dropped, as the JS writer prints."""
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def hex_color(rgb) -> str:
    return "#" + "".join(f"{max(0, min(255, round(float(c) * 255))):02x}" for c in rgb[:3])


def quadratic_loop(xs, ys) -> str:
    """k closed quadratic segments: segment j runs a_j -> c_j -> a_(j+1), wrapping."""
    k = len(xs) // 2
    parts = [f"M {number(xs[0])} {number(ys[0])}"]
    for j in range(k):
        control, end = 2 * j + 1, (2 * j + 2) % len(xs)
        parts.append(f"Q {number(xs[control])} {number(ys[control])} {number(xs[end])} {number(ys[end])}")
    return " ".join(parts) + " Z"


def write_final_svg(state, initial, args, path: Path) -> None:
    xs = state["points_x"].detach().cpu().tolist()
    ys = state["points_y"].detach().cpu().tolist()
    colors = state["colors"].detach().cpu().tolist()
    alphas = state["alphas"].detach().cpu().tolist()
    width, height = args.opt_width, args.opt_height
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">',
        f'  <rect width="{width}" height="{height}" fill="{hex_color(initial["background"])}"/>',
    ]
    for index in range(args.n):
        lines.append(
            f'  <path d="{quadratic_loop(xs[index], ys[index])}" fill="{hex_color(colors[index])}" '
            f'fill-opacity="{max(0.0, min(1.0, alphas[index])):.4f}" fill-rule="{FILL_RULE}"/>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def synchronize(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def geometry_lrs(args: argparse.Namespace) -> tuple[float, float]:
    return 0.5 * args.opt_width / 128.0, 0.5 * args.opt_height / 128.0


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import pydiffvg
    import torch
    from PIL import Image

    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("DiffVG benchmark requires CUDA")

    process_started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    pydiffvg.set_use_gpu(True)
    pydiffvg.set_device(device)

    setup_started = time.perf_counter()
    initial = json.loads(Path(args.init).read_text())
    validate_init(initial, args)
    target = None
    if args.loss == "l2":
        target_image = Image.open(args.target).convert("RGB")
        if target_image.size != (args.opt_width, args.opt_height):
            raise ValueError(
                f"target must be canonical {args.opt_width}x{args.opt_height}, got "
                f"{target_image.size[0]}x{target_image.size[1]}"
            )
        target = torch.from_numpy(np.asarray(target_image, dtype=np.float32).copy() / 255.0).to(device)
    background = torch.tensor(initial["background"], dtype=torch.float32, device=device)
    warm_state = make_state(initial, args, device, pydiffvg, torch)
    setup_pre_ms = (time.perf_counter() - setup_started) * 1000

    warmup_started = time.perf_counter()
    warm_client = ClipClient(args) if args.loss == "clip" and args.warmup else None
    try:
        for step in range(args.warmup):
            train_step(
                warm_state, target, background, warm_client, args, step,
                pydiffvg, torch, np,
            )
        synchronize(torch)
    finally:
        if warm_client:
            warm_client.close()
    warmup_ms = (time.perf_counter() - warmup_started) * 1000
    del warm_state

    reset_started = time.perf_counter()
    state = make_state(initial, args, device, pydiffvg, torch)
    client = ClipClient(args) if args.loss == "clip" else None
    prime_optimizer(state["optimizer"], torch)
    setup_ms = setup_pre_ms + (time.perf_counter() - reset_started) * 1000

    timeline: list[dict[str, float | int]] = []
    losses: list[float] = []
    # Wall-clock epoch of the first optimiser step. The report places traces on
    # an axis whose origin is when the *command* was launched, so it needs the
    # absolute instant optimisation began -- interpreter start, torch/pydiffvg
    # import, device init, setup and warmup all sit to the left of it.
    optimize_start_epoch_ms = time.time() * 1000
    optimize_started = time.perf_counter()
    deadline = math.inf if args.seconds is None else optimize_started + args.seconds
    step = 0
    try:
        while args.steps is None or step < args.steps:
            if time.perf_counter() >= deadline:
                break
            step_started = time.perf_counter()
            value = train_step(
                state, target, background, client, args, step,
                pydiffvg, torch, np,
            )
            synchronize(torch)
            now = time.perf_counter()
            losses.append(value)
            timeline.append({
                "step": step + 1,
                "elapsed_ms": (now - optimize_started) * 1000,
                "step_ms": (now - step_started) * 1000,
                "loss": value,
            })
            step += 1
            if now >= deadline:
                break
        optimize_ms = (time.perf_counter() - optimize_started) * 1000
    finally:
        if client:
            client.close()
    if not losses:
        raise RuntimeError("time budget ended before one optimizer step completed")

    final_started = time.perf_counter()
    with torch.no_grad():
        final = render(state, background, args, 0, pydiffvg, torch, samples=args.final_samples).clamp(0, 1)
    synchronize(torch)
    final_render_ms = (time.perf_counter() - final_started) * 1000

    output_started = time.perf_counter()
    pixels = np.rint(final.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(out / "final.png")
    write_final_svg(state, initial, args, out / "final.svg")
    (out / "trace.jsonl").write_text("".join(json.dumps(item) + "\n" for item in timeline))
    output_ms = (time.perf_counter() - output_started) * 1000
    step_times = np.asarray([item["step_ms"] for item in timeline], dtype=np.float64)
    best_step = min(range(len(losses)), key=losses.__getitem__) + 1
    budget_mode = "steps" if args.steps is not None else "seconds"
    requested = args.steps if args.steps is not None else args.seconds
    lr_x, lr_y = geometry_lrs(args)
    engine_options = {
        "samples": args.samples,
        "final_samples": args.final_samples,
        "seed_mode": "step",
        "hoist": True,
        "warmup_steps": args.warmup,
        "optimizer_state_primed": True,
        "geometry_lr": {"x": lr_x, "y": lr_y},
    }
    if args.loss == "clip":
        engine_options.update({
            "augs": args.augs,
            "model": "ViT-B-32-quickgelu",
            "pretrained": args.clip_weights,
        })
    result = {
        "schema_version": 1,
        "status": "ok",
        "engine": {
            "name": "diffvg",
            "environment": "diffvg",
            "backend": "cuda-diffvg",
            "torch": torch.__version__,
            "pydiffvg": getattr(pydiffvg, "__version__", None),
        },
        "workload": {
            "loss": args.loss,
            "primitive": "closed-quadratic-loop",
            "n": args.n,
            "k": args.k,
            "opt_size": [args.opt_width, args.opt_height],
            "seed": args.seed,
            "prompt": args.prompt if args.loss == "clip" else None,
            "init": {"kind": "shared-neutral"},
            "engine_options": engine_options,
        },
        "budget": {"mode": budget_mode, "requested": requested, "clock": "optimize"},
        "progress": {
            "steps_completed": step,
            "stop_reason": budget_mode,
            "deadline_overshoot_ms": max(0.0, optimize_ms - args.seconds * 1000)
            if args.seconds else 0.0,
        },
        "timing": {
            "process_internal_ms": (time.perf_counter() - process_started) * 1000,
            "setup_ms": setup_ms,
            "warmup_ms": warmup_ms,
            "optimize_ms": optimize_ms,
            "optimize_start_epoch_ms": optimize_start_epoch_ms,
            "final_render_ms": final_render_ms,
            "output_ms": output_ms,
            "steps_timed": step,
            "step_ms": {
                "mean": float(step_times.mean()),
                "median": float(np.median(step_times)),
                "p95": float(np.percentile(step_times, 95)),
            },
        },
        "loss": {
            "start": losses[0],
            "end": losses[-1],
            "min": min(losses),
            "best_step": best_step,
        },
        "artifacts": {"final_png": "final.png", "final_svg": "final.svg", "trace": "trace.jsonl"},
    }
    write_json(out / "result.json", result)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except Exception as error:
        write_json(out / "result.json", {
            "schema_version": 1,
            "status": "error",
            "engine": {"name": "diffvg", "environment": "diffvg", "backend": "cuda-diffvg"},
            "error": {"stage": "adapter", "message": str(error), "detail": traceback.format_exc()},
        })
        raise


if __name__ == "__main__":
    main()
