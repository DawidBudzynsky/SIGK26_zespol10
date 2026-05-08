"""Side-by-side qualitative comparison of GAN+L1 vs L1-only vs GT."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Generator
from utils.dataset import PhongDataset


def load(ckpt, device):
    G = Generator().to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    G.load_state_dict(state["generator"])
    G.eval()
    return G, state["test_idx"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G_gan, test_idx = load("lab3/checkpoints/neural_renderer.pt", device)
    G_l1, _ = load("lab3/checkpoints/l1_only.pt", device)

    full = PhongDataset("lab3/data")
    # Pick 8 samples spanning the brightness range — typical, large, and
    # close-up spheres — so the comparison shows behaviour on each regime.
    test_brightness = []
    for idx in test_idx:
        gt = np.asarray(full[idx][1])
        test_brightness.append((idx, float(gt.mean())))
    test_brightness.sort(key=lambda x: x[1])
    picks_pos = np.linspace(0, len(test_brightness) - 1, 8).astype(int)
    picks = [test_brightness[p][0] for p in picks_pos]

    rows = []
    title = Image.new("RGB", (4 * 128, 24), (32, 32, 32))
    d = ImageDraw.Draw(title)
    d.text((10, 5),               "GT", fill=(255, 255, 255))
    d.text((128 + 10, 5),         "GAN + L1", fill=(255, 255, 255))
    d.text((2 * 128 + 10, 5),     "L1 only", fill=(255, 255, 255))
    d.text((3 * 128 + 10, 5),     "|GAN - L1|", fill=(255, 255, 255))
    rows.append(np.array(title))

    for idx in picks:
        sample = full.samples[idx]
        feats = full[idx][0].unsqueeze(0).to(device)
        with torch.no_grad():
            gan_pred = G_gan(feats).cpu().numpy()[0]
            l1_pred = G_l1(feats).cpu().numpy()[0]
        gt = (np.transpose(full[idx][1].numpy(), (1, 2, 0)) * 255).astype(np.uint8)
        gan_pred = (np.transpose(gan_pred, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
        l1_pred = (np.transpose(l1_pred, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
        diff = np.abs(gan_pred.astype(np.int16) - l1_pred.astype(np.int16))
        diff = (diff * 4).clip(0, 255).astype(np.uint8)   # x4 to make it visible
        rows.append(np.concatenate([gt, gan_pred, l1_pred, diff], axis=1))

    canvas = np.concatenate(rows, axis=0)
    Image.fromarray(canvas).save("lab3/results/compare_gan_vs_l1.png")
    print("wrote lab3/results/compare_gan_vs_l1.png")


if __name__ == "__main__":
    main()
