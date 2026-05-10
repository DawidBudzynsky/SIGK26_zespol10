"""Mała poglądowa siatka 4x6 obrazów z wygenerowanego zbioru — pokazujemy
różnorodność scen (pozycje, kolory, połyskliwość)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset import PhongDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="lab3/data")
    ap.add_argument("--out",  default="lab3/results/dataset_preview.png")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    full = PhongDataset(args.data)
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(full), args.rows * args.cols, replace=False)

    cells = []
    for i in idx:
        _, img = full[int(i)]
        arr = (np.transpose(img.numpy(), (1, 2, 0)) * 255).astype(np.uint8)
        cells.append(arr)

    grid = np.zeros((args.rows * 128, args.cols * 128, 3), dtype=np.uint8)
    for k, c in enumerate(cells):
        r, col = divmod(k, args.cols)
        grid[r * 128:(r + 1) * 128, col * 128:(col + 1) * 128] = c

    canvas = Image.new("RGB", (args.cols * 128, args.rows * 128 + 22), (28, 28, 28))
    d = ImageDraw.Draw(canvas)
    d.text((6, 4), f"Wygenerowany zbior - losowych {args.rows * args.cols} z 3000 (renderer Phonga, 128x128)",
           fill=(220, 220, 220))
    canvas.paste(Image.fromarray(grid), (0, 22))
    canvas.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
