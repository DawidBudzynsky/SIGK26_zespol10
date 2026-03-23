# Image Denoising

Trained a neural denoising model on DIV2K dataset and compared against bilateral filter.

## Method

- **DenoisingModel**: Autoencoder with 3 encoder/decoder blocks (256 hidden units)
- **Baseline**: Bilateral filter (`skimage.restoration.denoise_bilateral`)

## Results

| Method | PSNR | SSIM | LPIPS |
|--------|------|------|-------|
| DenoisingModel | 28.87 | 0.986 | 0.046 |
| Bilateral | 23.08 | 0.943 | 0.223 |

Trained on 800 DIV2K images for 50 epochs.

## Examples

![Comparison](results/visualize_idx67.png)

Left to right: Noisy input, Ground truth, DenoisingModel, Bilateral

## Usage

```bash
python train_denoising.py                    # train + evaluate
python train_denoising.py --visualize 11 67   # visualize specific images
```
