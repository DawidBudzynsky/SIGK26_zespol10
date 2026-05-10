from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Generator
from utils.dataset import PhongDataset


def load(ckpt_path: str, device):
    G = Generator().to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    G.load_state_dict(ck["generator"])
    G.eval()
    return G, ck["test_idx"]


def to_u8(t):
    return (np.clip(np.transpose(t.numpy(), (1, 2, 0)) * 255, 0, 255)).astype(np.uint8)


def main():
    "skyrpcik to porownania graficznego"
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_gan", default="lab3/checkpoints/cgan.pt")
    ap.add_argument("--ckpt_l1", default="lab3/checkpoints/l1_only.pt")
    ap.add_argument("--data", default="lab3/data")
    ap.add_argument("--out", default="lab3/results/compare_gan_vs_l1.png")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G_gan, test_idx = load(args.ckpt_gan, device)
    G_l1, _ = load(args.ckpt_l1, device)

    full = PhongDataset(args.data)
    brightness = sorted(
        ((idx, float(np.asarray(full[idx][1]).mean())) for idx in test_idx),
        key=lambda x: x[1],
    )
    picks_pos = np.linspace(0, len(brightness) - 1, args.n).astype(int)
    picks = [brightness[p][0] for p in picks_pos]

    rows = []
    for idx in picks:
        feats, gt = full[idx]
        feats = feats.unsqueeze(0).to(device)
        with torch.no_grad():
            gan_p = G_gan(feats).cpu()[0]
            l1_p = G_l1(feats).cpu()[0]
        gt_u8 = to_u8(gt)
        gan_u8 = to_u8(gan_p)
        l1_u8 = to_u8(l1_p)
        diff = (
            (np.abs(gan_u8.astype(int) - l1_u8.astype(int)) * 4)
            .clip(0, 255)
            .astype(np.uint8)
        )
        rows.append(np.concatenate([gt_u8, gan_u8, l1_u8, diff], axis=1))

    body = np.concatenate(rows, axis=0)
    header = Image.new("RGB", (body.shape[1], 22), (28, 28, 28))
    d = ImageDraw.Draw(header)
    cell = 128
    for i, txt in enumerate(["GT", "cGAN + L1", "L1 only", "|cGAN-L1| ×4"]):
        d.text((i * cell + 6, 4), txt, fill=(220, 220, 220))
    canvas = np.concatenate([np.array(header), body], axis=0)
    Image.fromarray(canvas).save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
