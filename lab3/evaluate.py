"""Evaluate the trained neural renderer on the held-out 600-image test set.

Computes FLIP / LPIPS / SSIM / Hausdorff per sample, prints aggregate stats,
saves a CSV of per-sample numbers, a montage of qualitative comparisons,
and a copy of the metric table for the README.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models import Generator
from utils.dataset import PhongDataset
from utils.metrics import all_metrics, to_uint8


def make_montage(pairs, path, max_rows=8):
    """pairs is a list of (gt_uint8, pred_uint8) — vertically stacked rows."""
    rows = []
    for gt, pred in pairs[:max_rows]:
        diff = np.abs(gt.astype(np.int16) - pred.astype(np.int16)).astype(np.uint8)
        rows.append(np.concatenate([gt, pred, diff], axis=1))
    canvas = np.concatenate(rows, axis=0)
    Image.fromarray(canvas).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="lab3/data")
    parser.add_argument("--ckpt", default="lab3/checkpoints/neural_renderer.pt")
    parser.add_argument("--out_dir", default="lab3/results")
    parser.add_argument("--montage", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    G = Generator().to(device)
    G.load_state_dict(ck["generator"])
    G.eval()
    test_idx = ck["test_idx"]
    print(f"loaded {args.ckpt} (epoch {ck.get('epoch', '?')})")

    encoding = ck.get("encoding", "relative")
    full = PhongDataset(args.data, encoding=encoding)
    test_ds = Subset(full, test_idx)
    print(f"test set: {len(test_ds)}")
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    rows = []
    montage_pairs = []
    pbar = tqdm(total=len(test_ds), desc="evaluating")
    with torch.no_grad():
        for params, real in loader:
            params = params.to(device)
            fake = G(params).cpu().numpy()
            real = real.numpy()
            for i in range(len(params)):
                pred = to_uint8(fake[i])
                gt = to_uint8(real[i])
                m = all_metrics(pred, gt, device=str(device))
                rows.append(m)
                if len(montage_pairs) < args.montage:
                    montage_pairs.append((gt, pred))
                pbar.update(1)
    pbar.close()

    # Aggregate
    keys = ["FLIP", "LPIPS", "SSIM", "Hausdorff"]
    means = {k: np.mean([r[k] for r in rows]) for k in keys}
    stds  = {k: np.std([r[k] for r in rows])  for k in keys}

    print("\nResults on 600 test images:")
    print(f"  FLIP     = {means['FLIP']:.4f}  ± {stds['FLIP']:.4f}")
    print(f"  LPIPS    = {means['LPIPS']:.4f}  ± {stds['LPIPS']:.4f}")
    print(f"  SSIM     = {means['SSIM']:.4f}  ± {stds['SSIM']:.4f}")
    print(f"  Hausdorff= {means['Hausdorff']:.4f}  ± {stds['Hausdorff']:.4f}")

    # Per-sample CSV
    with open(out / "metrics_per_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx"] + keys)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"idx": test_idx[i], **r})

    # Summary file
    with open(out / "metrics_summary.md", "w") as f:
        f.write("| Metoda | FLIP | LPIPS | SSIM | Hausdorff |\n")
        f.write("|--------|------|-------|------|-----------|\n")
        f.write(f"| neural_renderer | {means['FLIP']:.4f} | {means['LPIPS']:.4f} "
                f"| {means['SSIM']:.4f} | {means['Hausdorff']:.4f} |\n")

    make_montage(montage_pairs, out / "qualitative_montage.png", max_rows=args.montage)
    print(f"wrote {out}/{{metrics_per_sample.csv, metrics_summary.md, qualitative_montage.png}}")


if __name__ == "__main__":
    main()
