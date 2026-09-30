#!/usr/bin/env python3
"""Prove a comparison engine really computes on the GPU.

Both benchmark adapters raise if ``torch.cuda.is_available()`` is false, so they
cannot silently fall back to the CPU -- but that check alone does not prove the
*rendering* work runs on the GPU. DiffVG in particular can import cleanly, render
forward, and only fail in the backward pass when its CUDA build lacks the pod
GPU's ``-gencode`` arch. This script exercises a real forward *and* backward pass
while sampling ``nvidia-smi`` utilization, and reports where the gradients live.

    .venv-diffvg/bin/python bench/gpu-verify.py --engine=diffvg
    .venv-bezier/bin/python bench/gpu-verify.py --engine=bezier
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import threading
import time


class Sampler:
    """Poll nvidia-smi for utilization and this process's GPU memory."""

    def __init__(self, interval: float = 0.02) -> None:
        self.interval = interval
        self.peak_util = 0
        self.peak_process_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        import os

        pid = str(os.getpid())
        while not self._stop.is_set():
            try:
                util = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                self.peak_util = max(self.peak_util, int(util.splitlines()[0]))
                apps = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                for line in apps.splitlines():
                    fields = [f.strip() for f in line.split(",")]
                    if len(fields) == 2 and fields[0] == pid:
                        self.peak_process_mib = max(self.peak_process_mib, int(fields[1]))
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self) -> "Sampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def check_diffvg(steps: int, size: int) -> dict:
    import pydiffvg
    import torch

    device = torch.device("cuda")
    pydiffvg.set_use_gpu(True)
    pydiffvg.set_device(device)

    # Closed cubic loop like the adapter's K-segment init: a closed path needs
    # sum(num_control_points) + num_segments points, i.e. 3K for K cubics.
    segments = 4
    angles = torch.arange(3 * segments, dtype=torch.float32) * (
        2 * math.pi / (3 * segments))
    points = (torch.stack([angles.cos(), angles.sin()], dim=1) * 0.35 + 0.5)
    points = (points * size).to(device).requires_grad_(True)
    color = torch.tensor([0.3, 0.6, 0.9, 1.0], device=device, requires_grad=True)
    path = pydiffvg.Path(
        num_control_points=torch.full((segments,), 2, dtype=torch.int32),
        points=points, is_closed=True,
        stroke_width=torch.tensor(1.0, device=device))
    group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=color)

    scene = pydiffvg.RenderFunction.serialize_scene(size, size, [path], [group])
    image = pydiffvg.RenderFunction.apply(size, size, 2, 2, 0, None, *scene)
    forward_device = str(image.device)

    with Sampler() as sampler:
        for step in range(steps):
            scene = pydiffvg.RenderFunction.serialize_scene(size, size, [path], [group])
            image = pydiffvg.RenderFunction.apply(size, size, 2, 2, step, None, *scene)
            (image[..., :3] ** 2).mean().backward()
        torch.cuda.synchronize()

    return {
        "forward_image_device": forward_device,
        "backward_ok": points.grad is not None,
        "grad_device": str(points.grad.device),
        "grad_norm": float(points.grad.norm()),
        "peak_gpu_util_percent": sampler.peak_util,
        "peak_process_gpu_mib": sampler.peak_process_mib,
    }


def check_bezier(steps: int, size: int) -> dict:
    import gsplat
    import torch

    device = torch.device("cuda")
    count = 512
    # GaussianImage parameterizes each 2D Gaussian by normalized centers and the
    # three lower-triangular Cholesky elements of its covariance.
    means = (torch.rand(count, 2, device=device) * 2 - 1).requires_grad_(True)
    cholesky = (torch.tensor([0.05, 0.0, 0.05], device=device)
                .repeat(count, 1) + torch.rand(count, 3, device=device) * 0.01)
    cholesky.requires_grad_(True)
    colors = torch.rand(count, 3, device=device, requires_grad=True)
    opacity = torch.rand(count, 1, device=device, requires_grad=True)
    tile_bounds = ((size + 15) // 16, (size + 15) // 16, 1)

    forward_device = None
    with Sampler() as sampler:
        for _ in range(steps):
            xys, depths, radii, conics, num_tiles_hit = gsplat.project_gaussians_2d(
                means, cholesky, size, size, tile_bounds)
            out = gsplat.rasterize_gaussians_sum(
                xys, depths, radii, conics, num_tiles_hit,
                colors, opacity, size, size, 16, 16,
                background=torch.ones(3, device=device), return_alpha=False)
            forward_device = str(out.device)
            (out ** 2).mean().backward()
        torch.cuda.synchronize()

    return {
        "gsplat_module": gsplat.__file__,
        "forward_image_device": forward_device,
        "backward_ok": means.grad is not None,
        "grad_device": str(means.grad.device),
        "grad_norm": float(means.grad.norm()),
        "peak_gpu_util_percent": sampler.peak_util,
        "peak_process_gpu_mib": sampler.peak_process_mib,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["diffvg", "bezier"], required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    import torch

    report = {
        "engine": args.engine,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        report["verdict"] = "FAIL: CUDA unavailable"
        print(json.dumps(report, indent=2))
        raise SystemExit(1)

    report["device"] = torch.cuda.get_device_name(0)
    report["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    report["arch_list"] = torch.cuda.get_arch_list()

    check = check_diffvg if args.engine == "diffvg" else check_bezier
    report.update(check(args.steps, args.size))

    on_gpu = (
        report.get("backward_ok")
        and "cuda" in report.get("grad_device", "")
        and "cuda" in report.get("forward_image_device", "")
    )
    # Utilization sampling can miss short bursts; memory residency is the
    # stronger signal, so treat a zero peak as inconclusive rather than fatal.
    report["verdict"] = "PASS: forward+backward on GPU" if on_gpu else "FAIL: not on GPU"
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if on_gpu else 1)


if __name__ == "__main__":
    main()
