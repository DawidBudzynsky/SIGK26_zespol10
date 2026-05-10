"""Train the conditional Phong-renderer GAN.

Pix2pix-style strata:
  L_D = 0.5 * (BCE(D(real, p), 1) + BCE(D(fake, p), 0))
  L_G =       BCE(D(fake, p), 1)  +  lambda_L1 * L1(fake, real)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models import Discriminator, Generator, PARAM_DIM
from utils.dataset import PhongDataset, split_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",        default="lab3/data")
    ap.add_argument("--epochs",      type=int,   default=80)
    ap.add_argument("--batch_size",  type=int,   default=32)
    ap.add_argument("--lr",          type=float, default=2e-4)
    ap.add_argument("--lambda_l1",   type=float, default=100.0)
    ap.add_argument("--no_gan",      action="store_true",
                    help="L1-only ablation (without adversarial loss)")
    ap.add_argument("--out",         default="lab3/checkpoints/cgan.pt")
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    full = PhongDataset(args.data)
    train_idx, test_idx = split_indices(len(full), test_frac=0.20, seed=args.seed)
    train_ds = Subset(full, train_idx)
    print(f"train: {len(train_ds)}  test: {len(test_idx)}  param_dim: {PARAM_DIM}")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    G = Generator().to(device)
    D = Discriminator().to(device)
    print(f"G params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"D params: {sum(p.numel() for p in D.parameters()):,}")

    opt_g = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    l1  = nn.L1Loss()
    mse = nn.MSELoss(reduction="none")

    def weighted_recon(fake, real):
        """Mask-weighted MSE (foreground = pixels with GT > 0.05).

        Mocniejszy gradient na pikselach kuli — bez tego sieć utyka w
        'all-black' lokalnym minimum (L1=0.005 to mean(GT) dla mostly-black GT).
        """
        mask = (real > 0.05).float()
        w = 1.0 + 9.0 * mask
        return (w * mse(fake, real)).mean()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.out).with_suffix(".log.txt")
    log = open(log_path, "w")
    log.write("epoch\tg_loss\td_loss\tl1\tdt_s\n")

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        t0 = time.time()
        agg = {"g": 0.0, "d": 0.0, "l1": 0.0}
        n = 0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for params, real in pbar:
            params = params.to(device, non_blocking=True)
            real   = real.to(device,   non_blocking=True)
            b = params.size(0)

            # ----- Discriminator ----------------------------------------
            # One-sided label smoothing (Salimans 2016): real target = 0.9
            # — daje D mniejszą pewność i wyraźniejszy gradient dla G.
            if not args.no_gan:
                with torch.no_grad():
                    fake = G(params)
                d_real = D(real, params)
                d_fake = D(fake, params)
                loss_d = 0.5 * (
                    bce(d_real, 0.9 * torch.ones_like(d_real)) +
                    bce(d_fake,      torch.zeros_like(d_fake))
                )
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
            else:
                loss_d = torch.zeros((), device=device)

            # ----- Generator --------------------------------------------
            fake = G(params)
            if args.no_gan:
                loss_gan = torch.zeros((), device=device)
            else:
                d_fake = D(fake, params)
                loss_gan = bce(d_fake, torch.ones_like(d_fake))
            loss_l1   = l1(fake, real)              # do logu
            loss_recon = weighted_recon(fake, real)
            loss_g     = loss_gan + args.lambda_l1 * loss_recon

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            agg["g"] += loss_g.item() * b
            agg["d"] += loss_d.item() * b
            agg["l1"] += loss_l1.item() * b
            n += b
            pbar.set_postfix(g=f"{loss_g.item():.3f}",
                             d=f"{loss_d.item():.3f}",
                             l1=f"{loss_l1.item():.4f}")

        dt = time.time() - t0
        print(f"epoch {epoch}/{args.epochs}  "
              f"G={agg['g']/n:.4f}  D={agg['d']/n:.4f}  "
              f"L1={agg['l1']/n:.5f}  ({dt:.1f}s)")
        log.write(f"{epoch}\t{agg['g']/n:.5f}\t{agg['d']/n:.5f}\t"
                  f"{agg['l1']/n:.6f}\t{dt:.2f}\n")
        log.flush()

        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save({
                "generator":     G.state_dict(),
                "discriminator": D.state_dict(),
                "epoch":         epoch,
                "param_dim":     PARAM_DIM,
                "train_idx":     train_idx,
                "test_idx":      test_idx,
                "no_gan":        args.no_gan,
                "lambda_l1":     args.lambda_l1,
            }, args.out)

    log.close()
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
