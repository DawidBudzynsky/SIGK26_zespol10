"""Dodatkowe wizualizacje:

  - metric_extremes.png  : dla każdej z 4 metryk pokazuje próbkę best/med/worst
                           (sieć i GT obok siebie). Pozwala wzrokowo
                           ocenić, czy metryki zgadzają się ze sobą.
  - loss_curves.png      : G/D/L1 w funkcji epoki (z `cgan.log.txt`).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Generator
from utils.dataset import PhongDataset


def load_rows(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out.append({"idx": int(r["idx"]),
                        "FLIP": float(r["FLIP"]),
                        "LPIPS": float(r["LPIPS"]),
                        "SSIM": float(r["SSIM"]),
                        "Hausdorff": float(r["Hausdorff"])})
    return out


def to_u8(t):
    return (np.clip(np.transpose(t.numpy(), (1, 2, 0)) * 255, 0, 255)).astype(np.uint8)


def label(img, txt):
    pil = Image.fromarray(img).resize((144, 144), Image.NEAREST)
    canvas = Image.new("RGB", (144, 172), (28, 28, 28))
    canvas.paste(pil, (0, 28))
    d = ImageDraw.Draw(canvas)
    for i, line in enumerate(txt.split("\n")):
        d.text((4, 4 + 12 * i), line, fill=(220, 220, 220))
    return np.array(canvas)


def extremes(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    G = Generator().to(device); G.load_state_dict(ck["generator"]); G.eval()
    full = PhongDataset(args.data)
    rows = load_rows(args.csv)
    pos = {r["idx"]: i for i, r in enumerate(rows)}

    metrics = ["FLIP", "LPIPS", "SSIM", "Hausdorff"]
    direction = {"FLIP": +1, "LPIPS": +1, "SSIM": -1, "Hausdorff": +1}

    grid = []
    for m in metrics:
        sorted_rows = sorted(rows, key=lambda r: r[m] * direction[m])
        best = sorted_rows[0]
        med  = sorted_rows[len(sorted_rows) // 2]
        worst = sorted_rows[-1]
        cells = []
        for tag, r in [("BEST", best), ("MED", med), ("WORST", worst)]:
            feats, gt = full[r["idx"]]
            feats = feats.unsqueeze(0).to(device)
            with torch.no_grad():
                pred = G(feats).cpu()[0]
            cells.append(label(to_u8(gt),   f"{m} {tag} GT"))
            cells.append(label(to_u8(pred), f"pred\n{m}={r[m]:.3f}"))
        grid.append(np.concatenate(cells, axis=1))
    canvas = np.concatenate(grid, axis=0)
    Image.fromarray(canvas).save(args.out)
    print(f"wrote {args.out}")

    print("\nPercentyle metryk:")
    for m in metrics:
        v = np.sort([r[m] for r in rows])
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        print(f"  {m:9s}  min={v.min():.4f}  p10={p10:.4f}  p50={p50:.4f}  p90={p90:.4f}  max={v.max():.4f}")


def loss_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs, g, d, l1 = [], [], [], []
    with open(args.log) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                epochs.append(int(parts[0]))
                g.append(float(parts[1]))
                d.append(float(parts[2]))
                l1.append(float(parts[3]))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, g, label="G total"); ax[0].plot(epochs, d, label="D")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("GAN losses")
    ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot(epochs, l1, color="tab:orange"); ax[1].set_yscale("log")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("L1 (log)"); ax[1].set_title("L1 (train)")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out); plt.close(fig)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_e = sub.add_parser("extremes")
    p_e.add_argument("--ckpt", default="lab3/checkpoints/cgan.pt")
    p_e.add_argument("--csv",  default="lab3/results/metrics_per_sample.csv")
    p_e.add_argument("--data", default="lab3/data")
    p_e.add_argument("--out",  default="lab3/results/metric_extremes.png")
    p_l = sub.add_parser("loss")
    p_l.add_argument("--log", default="lab3/checkpoints/cgan.log.txt")
    p_l.add_argument("--out", default="lab3/results/loss_curves.png")
    args = ap.parse_args()
    {"extremes": extremes, "loss": loss_plot}[args.cmd](args)


if __name__ == "__main__":
    main()
