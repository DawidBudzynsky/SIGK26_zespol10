"""Additional experiments on top of the main stick-animation training.

Each experiment trains (or loads) a model with one knob tweaked, evaluates
FMD/MPJPE/Var per class, and dumps a single ``ablations.csv`` row. The
default training horizon here is short (50 epochs) so the whole sweep
finishes in roughly the time it takes to train a single baseline; bump
``--epochs`` for production-quality numbers.

Experiments included
--------------------
A. **Noise schedule**         : cosine vs linear (200 epochs each)
B. **Parametrization**        : v-prediction vs ε-prediction
C. **DDIM steps**             : 25 / 50 / 100 / 1000 NFE during sampling
D. **CFG scale sweep**        : 1.0 / 2.0 / 3.0 / 5.0 / 7.5
E. **Loss ablation**          : ±bone, ±smooth, ±foot, ±all geometry
F. **Architecture ablation**  : ±DCT branch, ±skeleton bias
G. **Sampling-time bone snap**: on / off

Sampling-only sweeps (C, D, G) reuse a single trained checkpoint to keep
the experiment cheap.

Usage::

    uv run python experiments_stickanim.py --data-dir data/stickanim \
        --out-dir output/stickanim_experiments --epochs 200
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.stick_animation.dataset import MotionDataset, load_norm_stats
from src.stick_animation.diffusion import Diffusion
from src.stick_animation.metrics import summarise
from src.stick_animation.models.spatiotemporal_dit import (
    FEATURE_DIM,
    SpatioTemporalDiT,
)
from src.stick_animation.representation import MotionRep, rep_to_world
from src.stick_animation.sampling import reconstruct, sample
from train_stickanim import (
    EMA,
    build_argparser,
    evaluate,
    load_checkpoint,
    main as run_main,
    train as run_train,
)


CLASS_NAMES = {0: "walk", 1: "jump"}


@dataclass
class RunResult:
    name: str
    config: Dict[str, str]
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _base_argv(args, **overrides):
    """Compose argv list for train_stickanim from a base config + overrides."""
    base = {
        "--data-dir": args.data_dir,
        "--epochs": str(args.epochs),
        "--batch-size": str(args.batch_size),
        "--device": args.device,
        "--n-eval-samples": str(args.n_eval_samples),
        "--ddim-steps": str(args.ddim_steps),
        "--guidance-scale": str(args.guidance_scale),
        "--seed": str(args.seed),
    }
    base.update({f"--{k.replace('_','-')}": str(v) for k, v in overrides.items()})
    argv = []
    for k, v in base.items():
        argv.append(k)
        argv.append(v)
    return argv


def _run_training_experiment(
    name: str, args, overrides: Dict, results: List[RunResult]
):
    out_dir = Path(args.out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = _base_argv(args, out_dir=str(out_dir), **overrides)
    print(f"\n==== {name} ====")
    print(" ".join(argv))
    run_main(argv)
    metrics_path = out_dir / "metrics_final.json"
    if not metrics_path.exists():
        print(f"  ! missing {metrics_path}")
        return None
    with open(metrics_path) as f:
        metrics = json.load(f)
    res = RunResult(name=name, config=overrides, metrics=metrics)
    results.append(res)
    return out_dir / "final.pt"


def _evaluate_only(
    name: str, args, ckpt_path: Path, *, ddim_steps: int = None,
    guidance_scale: float = None, snap_bones: bool = None,
    schedule: str = "cosine", parametrization: str = "v",
    n_dct: int = None, results: Optional[List[RunResult]] = None,
):
    """Load a trained checkpoint and re-evaluate with different sampling knobs."""
    device = torch.device(args.device)
    train_ds = MotionDataset(os.path.join(args.data_dir, "train.npz"))
    test_ds = MotionDataset(os.path.join(args.data_dir, "test.npz"))
    norm = load_norm_stats(args.data_dir)
    mean = torch.tensor(norm["mean"], device=device, dtype=torch.float32)
    std = torch.tensor(norm["std"], device=device, dtype=torch.float32)

    cfg = argparse.Namespace(**vars(args))
    cfg.schedule = schedule
    cfg.param = parametrization
    if ddim_steps is not None:
        cfg.ddim_steps = ddim_steps
    if guidance_scale is not None:
        cfg.guidance_scale = guidance_scale
    if snap_bones is not None:
        cfg.snap_bones = snap_bones
    if n_dct is not None:
        cfg.n_dct = n_dct
    cfg.out_dir = str(Path(args.out_dir) / name)

    model = SpatioTemporalDiT(
        n_frames=cfg.n_frames, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_layers=cfg.n_layers, n_dct_tokens=cfg.n_dct,
        num_classes=2, dropout=cfg.dropout,
    ).to(device)
    ema = EMA(model, decay=cfg.ema_decay)
    load_checkpoint(ckpt_path, model, ema)
    diffusion = Diffusion(timesteps=cfg.timesteps, schedule=schedule,
                          parametrization=parametrization).to(device)
    metrics = evaluate(ema.shadow, diffusion, train_ds, test_ds, mean, std, norm, cfg, tag=name)
    if results is not None:
        results.append(RunResult(name=name,
                                 config={"ckpt": str(ckpt_path),
                                          "ddim_steps": str(cfg.ddim_steps),
                                          "guidance_scale": str(cfg.guidance_scale),
                                          "snap_bones": str(cfg.snap_bones),
                                          "schedule": schedule,
                                          "param": parametrization,
                                          "n_dct": str(cfg.n_dct)},
                                 metrics=metrics))


def _dump_csv(results: List[RunResult], out_path: Path):
    rows = []
    for r in results:
        for cls, vals in r.metrics.items():
            row = {"experiment": r.name, "class": cls, **vals, **r.config}
            rows.append(row)
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nwrote {out_path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=3.0)
    ap.add_argument("--n-eval-samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    # mirror the relevant model defaults so _evaluate_only knows the geometry
    ap.add_argument("--n-frames", type=int, default=48)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-dct", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--cfg-drop", type=float, default=0.1)
    ap.add_argument("--bone-w", type=float, default=0.5)
    ap.add_argument("--smooth-w", type=float, default=0.05)
    ap.add_argument("--foot-w", type=float, default=0.2)
    ap.add_argument("--vel-w", type=float, default=0.1)
    ap.add_argument("--snap-bones", action="store_true", default=True)
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    results: List[RunResult] = []

    # --- A. schedule ---
    ckpt_cosine_v = _run_training_experiment(
        "A_schedule_cosine_v", args,
        {"schedule": "cosine", "param": "v"}, results)
    _run_training_experiment(
        "A_schedule_linear_v", args,
        {"schedule": "linear", "param": "v"}, results)

    # --- B. parametrization ---
    _run_training_experiment(
        "B_param_eps_cosine", args,
        {"schedule": "cosine", "param": "eps"}, results)

    # --- C. DDIM steps (sampling-only on cosine+v ckpt) ---
    if ckpt_cosine_v and ckpt_cosine_v.exists():
        for n in (25, 50, 100, 1000):
            _evaluate_only(f"C_steps_{n}", args, ckpt_cosine_v,
                           ddim_steps=n, results=results)

    # --- D. CFG scale sweep ---
    if ckpt_cosine_v and ckpt_cosine_v.exists():
        for g in (1.0, 2.0, 3.0, 5.0, 7.5):
            _evaluate_only(f"D_cfg_{g:.1f}", args, ckpt_cosine_v,
                           guidance_scale=g, results=results)

    # --- E. loss ablation ---
    _run_training_experiment(
        "E_loss_no_bone", args,
        {"schedule": "cosine", "param": "v", "bone_w": 0.0}, results)
    _run_training_experiment(
        "E_loss_no_smooth", args,
        {"schedule": "cosine", "param": "v", "smooth_w": 0.0}, results)
    _run_training_experiment(
        "E_loss_no_foot", args,
        {"schedule": "cosine", "param": "v", "foot_w": 0.0}, results)
    _run_training_experiment(
        "E_loss_no_geom", args,
        {"schedule": "cosine", "param": "v",
         "bone_w": 0.0, "smooth_w": 0.0, "foot_w": 0.0}, results)

    # --- F. architecture ablation ---
    _run_training_experiment(
        "F_arch_no_dct", args,
        {"schedule": "cosine", "param": "v", "n_dct": 0}, results)

    # --- G. bone snap on/off (sampling-only on cosine+v ckpt) ---
    if ckpt_cosine_v and ckpt_cosine_v.exists():
        _evaluate_only("G_snap_off", args, ckpt_cosine_v,
                       snap_bones=False, results=results)
        _evaluate_only("G_snap_on", args, ckpt_cosine_v,
                       snap_bones=True, results=results)

    _dump_csv(results, Path(args.out_dir) / "ablations.csv")


if __name__ == "__main__":
    main()
