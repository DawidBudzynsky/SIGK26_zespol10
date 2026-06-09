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
from src.stick_animation.losses import total_loss
from src.stick_animation.metrics import summarise
from src.stick_animation.models.spatiotemporal_dit import FEATURE_DIM, SpatioTemporalDiT
from src.stick_animation.sampling import feat_to_world, reconstruct, sample
from src.stick_animation.visualize import animate_skeleton_3d, grid_static

SCRIPT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
CLASS_NAMES = {0: "walk", 1: "jump"}


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for ep, p in zip(self.shadow.parameters(), model.parameters()):
            ep.lerp_(p, 1.0 - self.decay)

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
    return Diffusion(
        timesteps=args.timesteps, schedule=args.schedule, parametrization=args.param
    )


def save_checkpoint(path, model, ema, opt, sched, epoch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema else None,
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict() if sched else None,
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(path, model, ema=None, opt=None, sched=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if ema and ckpt.get("ema"):
        ema.load(ckpt["ema"])
    if opt and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    if sched and ckpt.get("scheduler"):
        sched.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("epoch", 0))


@torch.no_grad()
def render_qualitative(model, diffusion, norm, args, out_dir: Path):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls in (0, 1):
        samples = sample(
            model,
            diffusion,
            class_label=cls,
            n_samples=args.eval_samples,
            n_frames=args.n_frames,
            n_steps=args.ddim_steps,
            guidance_scale=args.guidance_scale,
        )
        worlds = reconstruct(samples, norm)
        name = CLASS_NAMES[cls]
        grid_static(
            worlds,
            str(out_dir / f"{name}_grid_lastframe.png"),
            frame_idx=-1,
            title=name,
        )
        for i, motion in enumerate(worlds[: args.eval_samples]):
            animate_skeleton_3d(
                motion,
                output_filename=str(out_dir / f"{name}_s{i+1:02d}.gif"),
                fps=args.fps,
                follow_root=True,
                title=f"{name} #{i+1}",
            )
    model.train()


def train(args):
    device = torch.device(args.device)
    train_ds = MotionDataset(os.path.join(args.data_dir, "train.npz"))
    test_ds = MotionDataset(os.path.join(args.data_dir, "test.npz"))
    norm = load_norm_stats(args.data_dir)

    print(
        f"train: {len(train_ds)}  test: {len(test_ds)}  feature_dim={train_ds.sequences.shape[-1]}"
    )
    assert train_ds.sequences.shape[-1] == FEATURE_DIM, (
        f"Feature dim mismatch: {train_ds.sequences.shape[-1]} != {FEATURE_DIM}. "
        "Re-run prepare_data."
    )

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args).to(device)
    diffusion = make_diffusion(args).to(device)
    ema = EMA(model, decay=args.ema_decay)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        loaded_opt = None if args.resume_fresh_opt else opt
        loaded_sched = None if args.resume_fresh_sched else sched
        prev = load_checkpoint(Path(args.resume), model, ema, loaded_opt, loaded_sched)
        start_epoch = 1 if args.resume_reset_epoch else prev + 1
        if args.resume_fresh_sched:
            for pg in opt.param_groups:
                pg["lr"] = args.lr
            sched = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs, eta_min=1e-6
            )
        print(
            f"resumed from {args.resume}  prev_epoch={prev}  "
            f"start={start_epoch}  lr={opt.param_groups[0]['lr']:.2e}"
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"params: {n_params/1e6:.2f}M  device: {device}  "
        f"schedule: {args.schedule}  param: {args.param}  steps: {args.timesteps}"
    )

    ckpt_dir = Path(args.out_dir) / "ckpts"
    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = {"total": 0.0, "primary": 0.0, "velocity": 0.0}
        n = 0

        bar = tqdm(loader, desc=f"epoch {epoch:03d}/{args.epochs}", leave=False)
        for batch in bar:
            x0 = batch["x0"].to(device)
            label = batch["label"].to(device)
            t = torch.randint(0, diffusion.T, (x0.shape[0],), device=device)

            target, pred, x_t, _, _ = diffusion.training_pred(
                model, x0, t, label, cfg_drop_prob=args.cfg_drop
            )

            loss, parts = total_loss(target, pred, vel_weight=args.vel_w)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            ema.update(model)

            bs = x0.shape[0]
            running["total"] += float(loss.detach()) * bs
            running["primary"] += float(parts["primary"].detach()) * bs
            running["velocity"] += float(parts["velocity"].detach()) * bs
            n += bs
            bar.set_postfix(loss=f"{loss.item():.4f}")

        sched.step()
        avg = {k: v / max(n, 1) for k, v in running.items()}
        avg["lr"] = float(opt.param_groups[0]["lr"])
        avg["epoch"] = epoch
        history.append(avg)
        print(
            f"epoch {epoch:03d}  "
            + " | ".join(f"{k}={v:.4f}" for k, v in avg.items() if k != "epoch")
        )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            render_qualitative(
                ema.shadow,
                diffusion,
                norm,
                args,
                Path(args.out_dir) / "samples" / f"e{epoch:03d}",
            )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                ckpt_dir / f"ckpt_e{epoch:03d}.pt", model, ema, opt, sched, epoch
            )

    save_checkpoint(
        Path(args.out_dir) / "final.pt", model, ema, opt, sched, args.epochs
    )

    with open(Path(args.out_dir) / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    return model, ema, diffusion, train_ds, test_ds, norm


@torch.no_grad()
def evaluate(model, diffusion, train_ds, test_ds, norm, args, tag="final"):
    device = next(model.parameters()).device

    real_world = {0: [], 1: []}
    for d in test_ds:
        w = feat_to_world(d["x0"].numpy()[None], norm)[0]
        real_world[int(d["label"])].append(w)
    for cls in real_world:
        if real_world[cls]:
            real_world[cls] = np.stack(real_world[cls])

    metrics_table = {}
    gen_dir = Path(args.out_dir) / "generated" / tag
    for cls in (0, 1):
        samples = sample(
            model,
            diffusion,
            class_label=cls,
            n_samples=args.n_eval_samples,
            n_frames=args.n_frames,
            n_steps=args.ddim_steps,
            guidance_scale=args.guidance_scale,
            device=device,
        )
        gen_world = reconstruct(samples, norm)

        real = real_world[cls]
        if len(real) == 0:
            continue
        metrics_table[CLASS_NAMES[cls]] = summarise(real, gen_world)

        cls_dir = gen_dir / CLASS_NAMES[cls]
        cls_dir.mkdir(parents=True, exist_ok=True)
        grid_static(
            gen_world,
            str(cls_dir / f"{CLASS_NAMES[cls]}_grid_lastframe.png"),
            frame_idx=-1,
            title=CLASS_NAMES[cls],
        )
        for i, motion in enumerate(gen_world[:12]):
            animate_skeleton_3d(
                motion,
                output_filename=str(cls_dir / f"{CLASS_NAMES[cls]}_s{i+1:02d}.gif"),
                fps=args.fps,
                follow_root=True,
            )

    out_path = Path(args.out_dir) / f"metrics_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(metrics_table, f, indent=2)

    print("\n=== Final metrics ===")
    for cls, vals in metrics_table.items():
        print(f"{cls:5s}  " + "  ".join(f"{k}={v:.4f}" for k, v in vals.items()))
    print(f"saved → {out_path}")
    return metrics_table


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/stickanim")
    ap.add_argument("--out-dir", default="output/stickanim")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
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
    ap.add_argument("--vel-w", type=float, default=0.1)
    # sampling / eval
    ap.add_argument("--n-eval-samples", type=int, default=64)
    ap.add_argument("--eval-samples", type=int, default=8)
    ap.add_argument("--fps", type=int, default=24)
    # checkpointing
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--resume-fresh-sched", action="store_true")
    ap.add_argument("--resume-fresh-opt", action="store_true")
    ap.add_argument("--resume-reset-epoch", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.skip_train:
        train_ds = MotionDataset(os.path.join(args.data_dir, "train.npz"))
        test_ds = MotionDataset(os.path.join(args.data_dir, "test.npz"))
        norm = load_norm_stats(args.data_dir)
        device = torch.device(args.device)
        model = build_model(args).to(device)
        ema = EMA(model)
        diffusion = make_diffusion(args).to(device)
        if args.ckpt:
            load_checkpoint(Path(args.ckpt), model, ema)
        evaluate(ema.shadow, diffusion, train_ds, test_ds, norm, args)
        return

    model, ema, diffusion, train_ds, test_ds, norm = train(args)
    evaluate(ema.shadow, diffusion, train_ds, test_ds, norm, args)


if __name__ == "__main__":
    main()
