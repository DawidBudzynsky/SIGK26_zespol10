import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
import lpips
import cv2

from src.upscaling.models import UpscaleNet
from src.upscaling.dataset import UpscaleDataset

DATA_DIR = "/root/sigk/data/div2k/DIV2K_train_HR"
TEST_DATA_DIR = "/root/sigk/data/div2k/DIV2K_valid_HR"
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
LOW_RES_SIZE = 32
HIGH_RES_SIZE = 256
NUM_TRAIN = 800
NUM_TEST = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lpips_model = lpips.LPIPS(net="alex").to(DEVICE)


def compute_metrics(pred, target):
    pred_tensor = pred.unsqueeze(0).to(DEVICE)
    target_tensor = target.unsqueeze(0).to(DEVICE)

    psnr = peak_signal_noise_ratio(pred_tensor, target_tensor)
    ssim = structural_similarity_index_measure(pred_tensor, target_tensor)
    lpips_val = lpips_model(pred_tensor, target_tensor)

    return psnr.item(), ssim.item(), lpips_val.item()


def bicubic_upscale(low_res_image, target_size):
    return cv2.resize(low_res_image, target_size, interpolation=cv2.INTER_CUBIC)


def train():
    print(f"Training UpscaleNet on {DEVICE}")
    print(
        f"Upscaling from {LOW_RES_SIZE}x{LOW_RES_SIZE} to {HIGH_RES_SIZE}x{HIGH_RES_SIZE}"
    )

    train_dataset = UpscaleDataset(
        root_dir=DATA_DIR, low_res_size=LOW_RES_SIZE, high_res_size=HIGH_RES_SIZE
    )
    train_dataset.low_images = train_dataset.low_images[:NUM_TRAIN]
    train_dataset.high_images = train_dataset.high_images[:NUM_TRAIN]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    upscale_factor = HIGH_RES_SIZE // LOW_RES_SIZE
    model = UpscaleNet(
        num_residual_blocks=8, channels=64, upscale_factor=upscale_factor
    ).to(DEVICE)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for low, high in train_loader:
            low = low.to(DEVICE)
            high = high.to(DEVICE)

            optimizer.zero_grad()
            output = model(low)
            loss = criterion(output, high)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    torch.save(model.state_dict(), "/root/sigk/upscale_model.pth")
    print("Model saved to upscale_model.pth")
    return model


def evaluate(model):
    """Evaluate model and compare with bicubic baseline."""
    print("\nEvaluating...")
    model.eval()

    test_dataset = UpscaleDataset(
        root_dir=TEST_DATA_DIR, low_res_size=LOW_RES_SIZE, high_res_size=HIGH_RES_SIZE
    )
    test_dataset.low_images = test_dataset.low_images[:NUM_TEST]
    test_dataset.high_images = test_dataset.high_images[:NUM_TEST]

    results = {
        "UpscaleNet": {"psnr": [], "ssim": [], "lpips": []},
        "Bicubic": {"psnr": [], "ssim": [], "lpips": []},
    }

    os.makedirs("/root/sigk/results_upscaling", exist_ok=True)

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            low, high = test_dataset[idx]
            low = low.to(DEVICE)
            high = high.to(DEVICE)

            upscaled = model(low.unsqueeze(0)).squeeze(0)

            low_np = low.permute(1, 2, 0).cpu().numpy()
            high_np = high.permute(1, 2, 0).cpu().numpy()

            bicubic_result = bicubic_upscale(low_np, (HIGH_RES_SIZE, HIGH_RES_SIZE))

            psnr, ssim, lpips_val = compute_metrics(upscaled, high)
            results["UpscaleNet"]["psnr"].append(psnr)
            results["UpscaleNet"]["ssim"].append(ssim)
            results["UpscaleNet"]["lpips"].append(lpips_val)

            bicubic_tensor = (
                torch.from_numpy(bicubic_result).permute(2, 0, 1).float().to(DEVICE)
            )
            psnr, ssim, lpips_val = compute_metrics(bicubic_tensor, high)
            results["Bicubic"]["psnr"].append(psnr)
            results["Bicubic"]["ssim"].append(ssim)
            results["Bicubic"]["lpips"].append(lpips_val)

            if idx < 5:
                create_comparison_image(
                    low_np,
                    high_np,
                    upscaled.permute(1, 2, 0).cpu().numpy(),
                    bicubic_result,
                    idx,
                )

    print("\n" + "=" * 60)
    print("RESULTS TABLE")
    print("=" * 60)
    print(f"{'Method':<20} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10}")
    print("-" * 60)

    for method, metrics in results.items():
        psnr = np.mean(metrics["psnr"])
        ssim = np.mean(metrics["ssim"])
        lpips = np.mean(metrics["lpips"])
        print(f"{method:<20} {psnr:>10.4f} {ssim:>10.4f} {lpips:>10.4f}")

    print("=" * 60)

    with open("/root/sigk/results_upscaling.txt", "w") as f:
        f.write("Method,PSNR,SSIM,LPIPS\n")
        for method, metrics in results.items():
            psnr = np.mean(metrics["psnr"])
            ssim = np.mean(metrics["ssim"])
            lpips = np.mean(metrics["lpips"])
            f.write(f"{method},{psnr:.4f},{ssim:.4f},{lpips:.4f}\n")

    print("\nResults saved to results_upscaling.txt")


def create_comparison_image(low, high, upscaled_model, upscaled_bicubic, idx):
    import matplotlib.pyplot as plt

    _, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(low)
    axes[0].set_title("Low-Res Input")
    axes[0].axis("off")

    axes[1].imshow(high)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(upscaled_model)
    axes[2].set_title("UpscaleNet")
    axes[2].axis("off")

    axes[3].imshow(upscaled_bicubic)
    axes[3].set_title("Bicubic")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(f"/root/sigk/results_upscaling/comparison_idx{idx}.png", dpi=150)
    plt.close()


def visualize_indices(model, indices):
    print("\nVisualizing specific images...")
    model.eval()

    test_dataset = UpscaleDataset(
        root_dir=TEST_DATA_DIR, low_res_size=LOW_RES_SIZE, high_res_size=HIGH_RES_SIZE
    )

    import matplotlib.pyplot as plt

    for idx in indices:
        if idx >= len(test_dataset.high_images):
            print(f"Index {idx} out of range, skipping")
            continue

        low, high = test_dataset[idx]
        low = low.to(DEVICE)
        high = high.to(DEVICE)

        with torch.no_grad():
            upscaled = model(low.unsqueeze(0)).squeeze(0)

        low_np = low.permute(1, 2, 0).cpu().numpy()
        high_np = high.permute(1, 2, 0).cpu().numpy()
        upscaled_np = upscaled.permute(1, 2, 0).cpu().numpy()

        bicubic_np = bicubic_upscale(low_np, (HIGH_RES_SIZE, HIGH_RES_SIZE))

        psnr_model, ssim_model, _ = compute_metrics(upscaled, high)
        psnr_bicubic, ssim_bicubic, _ = compute_metrics(
            torch.from_numpy(bicubic_np).permute(2, 0, 1).float().to(DEVICE), high
        )

        print(f"\nImage {idx}:")
        print(f"  UpscaleNet: PSNR={psnr_model:.2f}, SSIM={ssim_model:.4f}")
        print(f"  Bicubic:    PSNR={psnr_bicubic:.2f}, SSIM={ssim_bicubic:.4f}")

        _, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(low_np)
        axes[0].set_title(f"Low-Res ({LOW_RES_SIZE}x{LOW_RES_SIZE})")
        axes[0].axis("off")

        axes[1].imshow(high_np)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(upscaled_np)
        axes[2].set_title(f"UpscaleNet\nPSNR={psnr_model:.2f} SSIM={ssim_model:.4f}")
        axes[2].axis("off")

        axes[3].imshow(bicubic_np)
        axes[3].set_title(f"Bicubic\nPSNR={psnr_bicubic:.2f} SSIM={ssim_bicubic:.4f}")
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(f"/root/sigk/results_upscaling/visualize_idx{idx}.png", dpi=150)
        plt.close()
        print(f"  Saved to results_upscaling/visualize_idx{idx}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", type=int, nargs="+", default=[])
    args, _ = parser.parse_known_args()

    if args.visualize:
        model = UpscaleNet(
            num_residual_blocks=8,
            channels=64,
            upscale_factor=HIGH_RES_SIZE // LOW_RES_SIZE,
        ).to(DEVICE)
        model.load_state_dict(torch.load("/root/sigk/upscale_model.pth"))
        model.eval()
        visualize_indices(model, args.visualize)
    else:
        model = train()
        evaluate(model)
