import argparse
import os
import json
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
import matplotlib.pyplot as plt
from src.denoising.models.denoising_autodecoder import DenoisingModel
from src.denoising.dataset import NoisyImageDataset
from src.denoising.methods.bilateral import bilateral_denoise
from src.denoising.dataset import resize_stretch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "div2k", "DIV2K_train_HR")
TEST_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "div2k", "DIV2K_valid_HR")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "denoising")
MODEL_PATH = os.path.join(OUTPUT_DIR, "denoising_model.pth")

BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
NOISE_SIGMA = 0.02
NUM_TRAIN = 800
NUM_TEST = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lpips_model = lpips.LPIPS(net="alex").to(DEVICE)


def compute_metrics(pred, target):
    pred_tensor = pred.unsqueeze(0)
    target_tensor = target.unsqueeze(0)

    pred_tensor = pred_tensor.to(DEVICE)
    target_tensor = target_tensor.to(DEVICE)

    psnr = peak_signal_noise_ratio(pred_tensor, target_tensor)
    ssim = structural_similarity_index_measure(pred_tensor, target_tensor)
    lpips_val = lpips_model(pred_tensor, target_tensor)

    return psnr.item(), ssim.item(), lpips_val.item()


def train():
    print(f"using device: {DEVICE}")

    train_dataset = NoisyImageDataset(
        root_dir=DATA_DIR, resize=resize_stretch, sigma=NOISE_SIGMA
    )

    train_dataset.images = train_dataset.images[:NUM_TRAIN]
    train_dataset.noisy_images = train_dataset.noisy_images[:NUM_TRAIN]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DenoisingModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for noisy, clean in train_loader:
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def evaluate(model):
    """Evaluate model and compare with bilateral filter."""
    print("\nEvaluating...")
    model.eval()

    test_dataset = NoisyImageDataset(
        root_dir=TEST_DATA_DIR, resize=resize_stretch, sigma=NOISE_SIGMA
    )
    test_dataset.images = test_dataset.images[:NUM_TEST]
    test_dataset.noisy_images = test_dataset.noisy_images[:NUM_TEST]

    results = {
        "summary": {
            "DenoisingModel": {"psnr": [], "ssim": [], "lpips": []},
            "Bilateral": {"psnr": [], "ssim": [], "lpips": []},
        },
        "per_image": [],
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            noisy, clean = test_dataset[idx]
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            with torch.no_grad():
                noisy_batch = noisy.unsqueeze(0)
                denoised = model(noisy_batch).squeeze(0)

            noisy_np = noisy.permute(1, 2, 0).cpu().numpy()
            clean_np = clean.permute(1, 2, 0).cpu().numpy()
            denoised_np = denoised.permute(1, 2, 0).cpu().numpy()
            bilateral_result = bilateral_denoise(noisy_np)

            psnr_model, ssim_model, lpips_model_val = compute_metrics(denoised, clean)
            results["summary"]["DenoisingModel"]["psnr"].append(psnr_model)
            results["summary"]["DenoisingModel"]["ssim"].append(ssim_model)
            results["summary"]["DenoisingModel"]["lpips"].append(lpips_model_val)

            bilateral_tensor = (
                torch.from_numpy(bilateral_result).permute(2, 0, 1).float().to(DEVICE)
            )
            psnr_bilateral, ssim_bilateral, lpips_bilateral_val = compute_metrics(bilateral_tensor, clean)
            results["summary"]["Bilateral"]["psnr"].append(psnr_bilateral)
            results["summary"]["Bilateral"]["ssim"].append(ssim_bilateral)
            results["summary"]["Bilateral"]["lpips"].append(lpips_bilateral_val)

            results["per_image"].append({
                "image_idx": idx,
                "DenoisingModel": {"psnr": psnr_model, "ssim": ssim_model, "lpips": lpips_model_val},
                "Bilateral": {"psnr": psnr_bilateral, "ssim": ssim_bilateral, "lpips": lpips_bilateral_val},
            })

            if idx < 5:
                create_comparison_image(
                    noisy_np, clean_np, denoised_np, bilateral_result, idx
                )

    print("\n" + "=" * 60)
    print("RESULTS TABLE")
    print("=" * 60)
    print(f"{'Method':<20} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10}")
    print("-" * 60)

    for method, metrics in results["summary"].items():
        psnr = np.mean(metrics["psnr"])
        ssim = np.mean(metrics["ssim"])
        lpips_val = np.mean(metrics["lpips"])
        print(f"{method:<20} {psnr:>10.4f} {ssim:>10.4f} {lpips_val:>10.4f}")
        metrics["mean"] = {"psnr": psnr, "ssim": ssim, "lpips": lpips_val}

    print("=" * 60)

    results_file = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")


def create_comparison_image(noisy, clean, denoised_model, denoised_bilateral, idx):

    _, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(noisy)
    axes[0].set_title("Noisy Input")
    axes[0].axis("off")

    axes[1].imshow(clean)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(denoised_model)
    axes[2].set_title("DenoisingModel")
    axes[2].axis("off")

    axes[3].imshow(denoised_bilateral)
    axes[3].set_title("Bilateral")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"comparison_idx{idx}.png"), dpi=150)
    plt.close()


def visualize_indices(model, indices, use_train=False):
    """Visualize specific indices from validation set."""
    print("\nVisualizing specific images...")
    model.eval()

    data_dir = DATA_DIR if use_train else TEST_DATA_DIR
    dataset = NoisyImageDataset(
        root_dir=data_dir, resize=resize_stretch, sigma=NOISE_SIGMA
    )

    import matplotlib.pyplot as plt

    for idx in indices:
        if idx >= len(dataset.images):
            print(
                f"Index {idx} out of range (only {len(dataset.images)} images), skipping"
            )
            continue

        noisy, clean = dataset[idx]
        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        noisy_np = noisy.permute(1, 2, 0).cpu().numpy()
        clean_np = clean.permute(1, 2, 0).cpu().numpy()

        with torch.no_grad():
            noisy_batch = noisy.unsqueeze(0)
            denoised_t = model(noisy_batch).squeeze(0)  
            denoised = denoised_t.permute(1, 2, 0).cpu().numpy()  

        bilateral_result = bilateral_denoise(noisy_np)

        psnr_model, ssim_model, _ = compute_metrics(denoised_t, clean)
        psnr_bilateral, ssim_bilateral, _ = compute_metrics(
            torch.from_numpy(bilateral_result).permute(2, 0, 1).to(DEVICE), clean
        )

        print(f"\nImage {idx} ({'train' if use_train else 'valid'}):")
        print(f"  DenoisingModel: PSNR={psnr_model:.2f}, SSIM={ssim_model:.4f}")
        print(f"  Bilateral:     PSNR={psnr_bilateral:.2f}, SSIM={ssim_bilateral:.4f}")

        _, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(noisy_np)
        axes[0].set_title(f"Noisy\n(idx={idx})")
        axes[0].axis("off")

        axes[1].imshow(clean_np)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(denoised)
        axes[2].set_title(
            f"DenoisingModel\nPSNR={psnr_model:.2f} SSIM={ssim_model:.4f}"
        )
        axes[2].axis("off")

        axes[3].imshow(bilateral_result)
        axes[3].set_title(
            f"Bilateral\nPSNR={psnr_bilateral:.2f} SSIM={ssim_bilateral:.4f}"
        )
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"visualize_idx{idx}.png"), dpi=150)
        plt.close()
        print(f"  Saved to output/denoising/visualize_idx{idx}.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", type=int, nargs="+", default=[])
    parser.add_argument(
        "--train-set",
        action="store_true",
        help="Use training set instead of validation for visualization",
    )
    args, _ = parser.parse_known_args()

    if args.visualize:
        model = DenoisingModel().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        visualize_indices(model, args.visualize, use_train=args.train_set)
    else:
        model = train()
        evaluate(model)
