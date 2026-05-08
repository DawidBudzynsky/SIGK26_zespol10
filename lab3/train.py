"""Train a conditional GAN to mimic the Phong renderer.

We use the pix2pix-style cGAN loss: BCE on the discriminator + L1 between
generated and GT pixels (heavy weight). The L1 keeps the diffuse colour and
sphere position correct, while the adversarial term sharpens highlights.
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

# Allow `python lab3/train.py` to import siblings.
sys.path.insert(0, str(Path(__file__).parent))

from models import Discriminator, Generator, PARAM_DIM
from utils.dataset import PhongDataset, split_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="lab3/data")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_l1", type=float, default=100.0)
    parser.add_argument("--no_gan", action="store_true",
                        help="train generator with L1 only (no adversarial loss)")
    parser.add_argument("--encoding", default="relative", choices=["relative", "absolute"],
                        help="parameter encoding (spec hint #1 ablation)")
    parser.add_argument("--out", default="lab3/checkpoints/neural_renderer.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    full = PhongDataset(args.data, encoding=args.encoding)
    train_idx, test_idx = split_indices(len(full), test_frac=0.2, seed=args.seed)
    train_ds = Subset(full, train_idx)
    test_ds = Subset(full, test_idx)
    print(f"train: {len(train_ds)}  test: {len(test_ds)}")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    G = Generator().to(device)
    D = Discriminator().to(device)

    opt_g = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.out).with_suffix(".log.txt")
    log = open(log_path, "w")
    log.write("epoch\tg_loss\td_loss\tl1\n")

    for epoch in range(args.epochs):
        G.train(); D.train()
        t0 = time.time()
        agg = {"g": 0.0, "d": 0.0, "l1": 0.0}
        n = 0
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False)
        for params, real in pbar:
            params = params.to(device)
            real = real.to(device)
            b = params.size(0)

            if not args.no_gan:
                # ---------- Discriminator ----------
                with torch.no_grad():
                    fake = G(params)
                d_real = D(real, params)
                d_fake = D(fake, params)
                target_real = torch.ones_like(d_real)
                target_fake = torch.zeros_like(d_fake)
                loss_d = 0.5 * (bce(d_real, target_real) + bce(d_fake, target_fake))

                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
            else:
                loss_d = torch.tensor(0.0, device=device)

            # ---------- Generator ----------
            fake = G(params)
            if args.no_gan:
                loss_gan = torch.tensor(0.0, device=device)
            else:
                d_fake_for_g = D(fake, params)
                loss_gan = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))
            loss_l1 = l1(fake, real)
            loss_g = loss_gan + args.lambda_l1 * loss_l1

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
        line = (f"epoch {epoch+1}/{args.epochs}  "
                f"G={agg['g']/n:.4f}  D={agg['d']/n:.4f}  "
                f"L1={agg['l1']/n:.5f}  ({dt:.1f}s)")
        print(line)
        log.write(f"{epoch+1}\t{agg['g']/n:.5f}\t{agg['d']/n:.5f}\t{agg['l1']/n:.6f}\n")
        log.flush()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            torch.save({
                "generator": G.state_dict(),
                "discriminator": D.state_dict(),
                "epoch": epoch + 1,
                "param_dim": PARAM_DIM,
                "test_idx": test_idx,
                "train_idx": train_idx,
                "encoding": args.encoding,
            }, args.out)

    log.close()
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
