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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models import Generator
from utils.dataset import PhongDataset
from utils.metrics import all_metrics, to_uint8


def make_montage(pairs, path, max_rows=8, scale=2, diff_gain=4):
    """Build a 3-column montage (GT | Pred | |GT-Pred|×diff_gain) with row
    labels marking 'TYPICAL' (top half) vs 'HARD' (bottom half).
    """
    rows = []
    n = min(max_rows, len(pairs))
    half = n // 2
    for i, (gt, pred) in enumerate(pairs[:n]):
        diff = np.abs(gt.astype(np.int16) - pred.astype(np.int16)) * diff_gain
        diff = diff.clip(0, 255).astype(np.uint8)
        row = np.concatenate([gt, pred, diff], axis=1)
        row_pil = Image.fromarray(row).resize(
            (row.shape[1] * scale, row.shape[0] * scale), Image.NEAREST,
        )
        rows.append(np.array(row_pil))

    body = np.concatenate(rows, axis=0)
    h, w = body.shape[:2]
    cell = w // 3

    # Header with column labels
    header = Image.new("RGB", (w + 80, 28), (32, 32, 32))
    d = ImageDraw.Draw(header)
    d.text((80 + cell // 2 - 8, 7),                  "GT",                       fill=(255, 255, 255))
    d.text((80 + cell + cell // 2 - 16, 7),          "Pred",                     fill=(255, 255, 255))
    d.text((80 + 2 * cell + cell // 2 - 50, 7),      f"|GT - Pred|  (x{diff_gain})", fill=(255, 200, 200))

    # Side-strip with row-group labels
    side = Image.new("RGB", (80, h), (32, 32, 32))
    d = ImageDraw.Draw(side)
    cell_h = h // n
    d.text((4, half * cell_h // 2 - 6),                                   "TYPICAL", fill=(180, 220, 255))
    d.text((4, half * cell_h + (n - half) * cell_h // 2 - 6),             "HARD",    fill=(255, 200, 180))

    body_with_side = np.concatenate([np.array(side), body], axis=1)
    canvas = np.concatenate([np.array(header), body_with_side], axis=0)
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
    montage_candidates = []   # (FLIP, brightness, gt, pred)
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
                montage_candidates.append((m["FLIP"], float(gt.mean()), gt, pred))
                pbar.update(1)
    pbar.close()

    # Build a montage that shows both modes honestly:
    #   * top half: best half of test (low FLIP) — the model's typical accurate mode
    #   * bottom half: worst quarter — the failure cases (close-ups, off-position)
    montage_candidates.sort(key=lambda x: x[0])
    n_show = args.montage
    half = n_show // 2
    n = len(montage_candidates)
    good_pos = np.linspace(0, n // 2, half).astype(int)
    bad_pos = np.linspace(int(n * 0.75), n - 1, n_show - half).astype(int)
    montage_pairs = [(montage_candidates[p][2], montage_candidates[p][3])
                     for p in list(good_pos) + list(bad_pos)]

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
