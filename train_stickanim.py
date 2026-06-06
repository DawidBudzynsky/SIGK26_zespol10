"""Train the stick-animation diffusion model and evaluate FMD/MPJPE/Var.

Usage
-----
1. Place CMU MoCap walk and jump trials under ``data/raw/walk/`` and
   ``data/raw/jump/`` (either ``.bvh`` or pre-extracted ``.npy``).

2. Build the canonical feature tensors::

       uv run python -m src.stick_animation.prepare_data \
            --raw-dir data/raw --out-dir data/stickanim

3. Train + evaluate::

       uv run python train_stickanim.py --data-dir data/stickanim

4. Generate samples with a trained checkpoint::

       uv run python train_stickanim.py --data-dir data/stickanim \
            --skip-train --ckpt output/stickanim/final.pt
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.stick_animation.dataset import MotionDataset, load_norm_stats
from src.stick_animation.diffusion import Diffusion
from src.stick_animation.losses import LossWeights, total_loss
from src.stick_animation.metrics import summarise
from src.stick_animation.models.spatiotemporal_dit import (
    FEATURE_DIM,
    SpatioTemporalDiT,
)
from src.stick_animation.representation import MotionRep, rep_to_world, snap_bone_lengths
from src.stick_animation.sampling import reconstruct, sample
from src.stick_animation.visualize import animate_skeleton_3d, grid_static


SCRIPT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
CLASS_NAMES = {0: "walk", 1: "jump"}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        ema_params = list(self.shadow.parameters())
        model_params = list(model.parameters())
        # Single fused foreach kernel instead of a Python-level per-tensor loop.
        torch._foreach_lerp_(ema_params, model_params, 1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load(self, state):
        self.shadow.load_state_dict(state)


def build_model(args) -> SpatioTemporalDiT:
    return SpatioTemporalDiT(
        n_frames=args.n_frames,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_dct_tokens=args.n_dct,
        num_classes=2,
        dropout=args.dropout,
    )


def make_diffusion(args) -> Diffusion:
    return Diffusion(timesteps=args.timesteps,
                     schedule=args.schedule,
                     parametrization=args.param)


def save_checkpoint(path: Path, model, ema, opt, sched, epoch: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
        "epoch": epoch,
    }, path)


def load_checkpoint(path: Path, model, ema=None, opt=None, sched=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.load(ckpt["ema"])
    if opt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    if sched is not None and ckpt.get("scheduler") is not None:
        sched.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("epoch", 0))


def render_qualitative(
    samples_world: np.ndarray,
    out_dir: Path,
    class_name: str,
    fps: int = 24,
    max_gifs: int = 8,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_static(samples_world, str(out_dir / f"{class_name}_grid_lastframe.png"),
                frame_idx=-1, title=class_name)
    for i, motion in enumerate(samples_world[:max_gifs]):
        animate_skeleton_3d(motion,
                             output_filename=str(out_dir / f"{class_name}_s{i+1:02d}.gif"),
                             fps=fps, follow_root=True, title=f"{class_name} #{i+1}")


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------


def train(args):
    device = torch.device(args.device)

    train_ds = MotionDataset(os.path.join(args.data_dir, "train.npz"))
    test_ds = MotionDataset(os.path.join(args.data_dir, "test.npz"))
    norm = load_norm_stats(args.data_dir)
    mean = torch.tensor(norm["mean"], device=device, dtype=torch.float32)
    std = torch.tensor(norm["std"], device=device, dtype=torch.float32)

    print(f"train: {len(train_ds)}  test: {len(test_ds)}  feature_dim={train_ds.sequences.shape[-1]}")
    assert train_ds.sequences.shape[-1] == FEATURE_DIM, (
        f"Feature dim {train_ds.sequences.shape[-1]} != expected {FEATURE_DIM}. "
        "Re-run prepare_data.")

    loader = DataLoader(train_ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=(args.num_workers > 0),
                        drop_last=True)

    model = build_model(args).to(device)
    diffusion = make_diffusion(args).to(device)
    ema = EMA(model, decay=args.ema_decay)

    if args.compile and device.type == "cuda":
        model = torch.compile(model)

    # Mixed precision: prefer bf16 on Ampere+ (no GradScaler needed), else fp16.
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = (torch.bfloat16
                 if use_amp and torch.cuda.is_bf16_supported()
                 else torch.float16)
    scaler = torch.amp.GradScaler("cuda",
                                  enabled=(use_amp and amp_dtype == torch.float16))
    if use_amp:
        print(f"AMP enabled: {amp_dtype}")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    weights = LossWeights(bone=args.bone_w, smooth=args.smooth_w,
                          foot=args.foot_w, vel=args.vel_w)

    start_epoch = 1
    ckpt_dir = Path(args.out_dir) / "ckpts"
    if args.resume is not None and Path(args.resume).exists():
        start_epoch = load_checkpoint(Path(args.resume), model, ema, opt, sched) + 1
        print(f"resumed from {args.resume} (epoch {start_epoch})")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params/1e6:.2f}M  device: {device}  schedule: {args.schedule}  "
          f"param: {args.param}  steps: {args.timesteps}")

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = {"total": 0.0, "primary": 0.0, "bone": 0.0,
                   "smooth": 0.0, "foot": 0.0, "velocity": 0.0}
        n = 0
        bar = tqdm(loader, desc=f"epoch {epoch:03d}/{args.epochs}", leave=False)
        for batch in bar:
            x0 = batch["x0"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            bone_scale = batch["bone_scale"].to(device, non_blocking=True)
            rest_bones = batch["rest_bones"].to(device, non_blocking=True)
            t = torch.randint(0, diffusion.T, (x0.shape[0],), device=device)

            # Heavy transformer forward runs in mixed precision; the geometric
            # losses are recomputed in fp32 for numerical stability.
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                target, pred, x_t, noise, _ = diffusion.training_pred(
                    model, x0, t, label, cfg_drop_prob=args.cfg_drop)
            x0_pred = diffusion.x0_from_pred(pred.float(), x_t.float(), t)
            losses = total_loss(target.float(), pred.float(), x0_pred,
                                rest_bones, bone_scale, mean, std,
                                weights=weights,
                                use_world_losses=args.use_world_losses)
            opt.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            bs = x0.shape[0]
            for k, v in losses.items():
                if k in running:
                    running[k] += float(v.detach().cpu()) * bs
            n += bs
            bar.set_postfix(loss=f"{losses['total'].item():.4f}")

        sched.step()
        avg = {k: v / max(n, 1) for k, v in running.items()}
        avg["lr"] = float(opt.param_groups[0]["lr"])
        avg["epoch"] = epoch
        history.append(avg)
        msg = " | ".join(f"{k}={v:.4f}" for k, v in avg.items() if k != "epoch")
        print(f"epoch {epoch:03d}  {msg}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            _qualitative_snapshot(ema.shadow, diffusion, mean, std,
                                  norm["rest_bones"], args,
                                  out_dir=Path(args.out_dir) / "samples" / f"e{epoch:03d}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(ckpt_dir / f"ckpt_e{epoch:03d}.pt", model, ema, opt, sched, epoch)

    save_checkpoint(Path(args.out_dir) / "final.pt", model, ema, opt, sched, args.epochs)

    # Persist training history as CSV for easy plotting.
    with open(Path(args.out_dir) / "history.csv", "w", newline="") as f:
        if history:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            for row in history:
                w.writerow(row)

    return model, ema, diffusion, train_ds, test_ds, mean, std, norm


# ----------------------------------------------------------------------------
# Sampling / evaluation
# ----------------------------------------------------------------------------


@torch.no_grad()
def _qualitative_snapshot(model, diffusion, mean, std, rest_bones, args, out_dir: Path):
    model.eval()
    for cls in (0, 1):
        samples = sample(model, diffusion, class_label=cls,
                         n_samples=args.eval_samples,
                         feature_dim=FEATURE_DIM,
                         n_frames=args.n_frames,
                         n_steps=args.ddim_steps,
                         guidance_scale=args.guidance_scale)
        worlds = reconstruct(samples,
                              norm_mean=mean.cpu().numpy(),
                              norm_std=std.cpu().numpy(),
                              rest_bones=rest_bones,
                              bone_scale=1.0,
                              apply_bone_snap=args.snap_bones)
        render_qualitative(worlds, out_dir, CLASS_NAMES[cls],
                            fps=args.fps, max_gifs=args.eval_samples)


@torch.no_grad()
def evaluate(model, diffusion, train_ds, test_ds, mean, std, norm, args, tag: str = "final"):
    """Compute FMD/MPJPE/Var and save GIFs."""
    device = next(model.parameters()).device
    rest_bones = norm["rest_bones"]

    # Real motions, per class, reconstructed back to world space.
    real_world = {0: [], 1: []}
    for sample_dict in test_ds:
        x = sample_dict["x0"].numpy()
        x = x * norm["std"] + norm["mean"]
        rep = MotionRep.from_tensor(x, bone_scale=float(sample_dict["bone_scale"]))
        w = rep_to_world(rep)
        real_world[int(sample_dict["label"])].append(w)
    for cls in real_world:
        real_world[cls] = np.stack(real_world[cls], axis=0) if real_world[cls] else np.empty((0,))

    # Generate samples per class.
    metrics_table = {}
    gen_dir = Path(args.out_dir) / "generated" / tag
    for cls in (0, 1):
        samples = sample(model, diffusion, class_label=cls,
                         n_samples=args.n_eval_samples,
                         feature_dim=FEATURE_DIM,
                         n_frames=args.n_frames,
                         n_steps=args.ddim_steps,
                         guidance_scale=args.guidance_scale,
                         device=device)
        gen_world = reconstruct(samples, mean.cpu().numpy(), std.cpu().numpy(),
                                rest_bones=rest_bones,
                                bone_scale=1.0,
                                apply_bone_snap=args.snap_bones)
        report = summarise(real_world[cls], gen_world)
        metrics_table[CLASS_NAMES[cls]] = report
        render_qualitative(gen_world, gen_dir / CLASS_NAMES[cls],
                            CLASS_NAMES[cls], fps=args.fps,
                            max_gifs=min(12, args.n_eval_samples))

    out_path = Path(args.out_dir) / f"metrics_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(metrics_table, f, indent=2)
    print("\n=== Final metrics ===")
    for cls, vals in metrics_table.items():
        msg = "  ".join(f"{k}={v:.4f}" for k, v in vals.items())
        print(f"{cls:5s}  {msg}")
    print(f"saved → {out_path}")
    return metrics_table


# ----------------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/stickanim")
    ap.add_argument("--out-dir", default="output/stickanim")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", default=True,
                    help="Use CUDA automatic mixed precision (bf16/fp16).")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--compile", action="store_true", default=False,
                    help="Wrap the model in torch.compile (CUDA only).")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    # model
    ap.add_argument("--n-frames", type=int, default=48)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-dct", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    # diffusion
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--schedule", choices=["cosine", "linear"], default="cosine")
    ap.add_argument("--param", choices=["v", "eps"], default="v")
    ap.add_argument("--cfg-drop", type=float, default=0.1)
    ap.add_argument("--guidance-scale", type=float, default=3.0)
    ap.add_argument("--ddim-steps", type=int, default=50)
    # losses
    ap.add_argument("--bone-w", type=float, default=0.5)
    ap.add_argument("--smooth-w", type=float, default=0.05)
    ap.add_argument("--foot-w", type=float, default=0.2)
    ap.add_argument("--vel-w", type=float, default=0.1)
    ap.add_argument("--use-world-losses", action="store_true", default=True)
    ap.add_argument("--no-world-losses", dest="use_world_losses", action="store_false")
    # sampling / evaluation
    ap.add_argument("--n-eval-samples", type=int, default=64)
    ap.add_argument("--eval-samples", type=int, default=8)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--snap-bones", action="store_true", default=True)
    ap.add_argument("--no-snap-bones", dest="snap_bones", action="store_false")
    # checkpointing
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip training; load --ckpt and run evaluation only.")
    ap.add_argument("--ckpt", default=None,
                    help="Checkpoint to load when --skip-train.")
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Enable the fast CUDA math paths (TF32 matmuls + cuDNN autotuner).
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if args.skip_train:
        # Build the bare model and load checkpoint for evaluation.
        train_ds = MotionDataset(os.path.join(args.data_dir, "train.npz"))
        test_ds = MotionDataset(os.path.join(args.data_dir, "test.npz"))
        norm = load_norm_stats(args.data_dir)
        device = torch.device(args.device)
        mean = torch.tensor(norm["mean"], device=device, dtype=torch.float32)
        std = torch.tensor(norm["std"], device=device, dtype=torch.float32)
        model = build_model(args).to(device)
        ema = EMA(model, decay=args.ema_decay)
        diffusion = make_diffusion(args).to(device)
        if args.ckpt:
            load_checkpoint(Path(args.ckpt), model, ema)
        evaluate(ema.shadow, diffusion, train_ds, test_ds, mean, std, norm, args)
        return

    model, ema, diffusion, train_ds, test_ds, mean, std, norm = train(args)
    evaluate(ema.shadow, diffusion, train_ds, test_ds, mean, std, norm, args)


if __name__ == "__main__":
    main()
