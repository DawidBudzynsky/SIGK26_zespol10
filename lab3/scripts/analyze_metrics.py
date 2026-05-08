"""Look beyond the means: which metrics actually reflect render quality?

Reads metrics_per_sample.csv from lab3/results, finds the best/worst sample
for each metric and renders a grid that lets a human eye-judge agreement.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Generator
from utils.dataset import PhongDataset


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({"idx": int(r["idx"]),
                         "FLIP": float(r["FLIP"]),
                         "LPIPS": float(r["LPIPS"]),
                         "SSIM": float(r["SSIM"]),
                         "Hausdorff": float(r["Hausdorff"])})
    return rows


def render_pred(G, sample, ds, device):
    feats, real = ds[sample]   # encoded already, but we want the absolute idx
    feats = feats.unsqueeze(0).to(device)
    with torch.no_grad():
        out = G(feats).cpu().numpy()[0]
    pred = (np.transpose(out, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
    gt = (np.transpose(real.numpy(), (1, 2, 0)) * 255).astype(np.uint8)
    return gt, pred


def label_image(img, text):
    pil = Image.fromarray(img).resize((128, 128))
    canvas = Image.new("RGB", (128, 148), (32, 32, 32))
    canvas.paste(pil, (0, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), text, fill=(255, 255, 255))
    return np.array(canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="lab3/checkpoints/neural_renderer.pt")
    parser.add_argument("--csv", default="lab3/results/metrics_per_sample.csv")
    parser.add_argument("--data", default="lab3/data")
    parser.add_argument("--out", default="lab3/results/metric_extremes.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    G = Generator().to(device)
    G.load_state_dict(ck["generator"])
    G.eval()
    test_idx = ck["test_idx"]
    full = PhongDataset(args.data)

    rows = load_csv(args.csv)
    # rebuild map idx->row position in test set
    pos_for_idx = {idx: pos for pos, idx in enumerate(test_idx)}
    metrics = ["FLIP", "LPIPS", "SSIM", "Hausdorff"]

    # For SSIM bigger is better; rest smaller is better.
    direction = {"FLIP": +1, "LPIPS": +1, "SSIM": -1, "Hausdorff": +1}

    grid_rows = []
    for m in metrics:
        sorted_rows = sorted(rows, key=lambda r: r[m] * direction[m])
        best = sorted_rows[0]
        worst = sorted_rows[-1]
        median = sorted_rows[len(sorted_rows) // 2]
        cells = []
        for tag, r in [("BEST", best), ("MED", median), ("WORST", worst)]:
            ds_pos = pos_for_idx[r["idx"]]
            gt, pred = render_pred(G, ds_pos,
                                   torch.utils.data.Subset(full, test_idx),
                                   device)
            cells.append(label_image(gt, f"{m} {tag} GT"))
            cells.append(label_image(pred, f"{m}={r[m]:.3f}"))
        grid_rows.append(np.concatenate(cells, axis=1))

    canvas = np.concatenate(grid_rows, axis=0)
    Image.fromarray(canvas).save(args.out)
    print(f"wrote {args.out}")

    # Print per-metric percentiles to console
    print("\nPer-metric percentiles (best→worst):")
    for m in metrics:
        vals = np.sort([r[m] for r in rows])
        pcts = np.percentile(vals, [10, 50, 90])
        print(f"  {m:9s}  p10={pcts[0]:.4f}  p50={pcts[1]:.4f}  p90={pcts[2]:.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}")


if __name__ == "__main__":
    main()
