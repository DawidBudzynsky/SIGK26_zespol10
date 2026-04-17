import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn
from tqdm import tqdm

from models.exposure_synthesizer import ExposureSynthesizer
from utils.hdr import (
    get_exif,
    measure_ev_range,
    read_exposure_time,
    read_hdr,
    tone_map_reinhard,
)

TEST_SCENES = ['C40', 'C41', 'C42', 'C43', 'C44', 'C45', 'C46']
IMG_SIZE = (512, 512)


def load_ldr_rgb(path: Path, size=IMG_SIZE) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    return img


def load_hdr_rgb(path: Path, size=IMG_SIZE) -> np.ndarray:
    hdr = read_hdr(str(path))
    return cv2.resize(hdr, size, interpolation=cv2.INTER_LINEAR)


def find_hdr_file(hdr_dir: Path, scene: str) -> Path | None:
    for name in (f'{scene}_HDR.hdr', f'{scene}.hdr'):
        p = hdr_dir / name
        if p.exists():
            return p
    return None


def synth_under_over(model, base_rgb_u8: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(base_rgb_u8).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x.to(device)
    with torch.no_grad():
        y = model(x)[0].cpu().numpy()
    y = np.clip(y, 0, 1).transpose(1, 2, 0)
    under = (y[:, :, :3] * 255).astype(np.uint8)
    over = (y[:, :, 3:] * 255).astype(np.uint8)
    return under, over


def merge_debevec_rgb(rgb_images_u8: list[np.ndarray], times: np.ndarray) -> np.ndarray:
    bgr_list = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in rgb_images_u8]
    merged_bgr = cv2.createMergeDebevec().process(bgr_list, times=times.astype(np.float32))
    merged = cv2.cvtColor(merged_bgr, cv2.COLOR_BGR2RGB)
    return merged.astype(np.float32)


def tonemap_reinhard_u8(hdr_rgb: np.ndarray) -> np.ndarray:
    ldr = tone_map_reinhard(hdr_rgb)
    ldr = np.nan_to_num(ldr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(ldr * 255.0, 0, 255).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray, max_val: float) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 0:
        return float('inf')
    return 20.0 * np.log10(max_val / np.sqrt(mse))


def mu_law(x: np.ndarray, max_val: float, mu: float = 5000.0) -> np.ndarray:
    x = np.clip(x / max(max_val, 1e-8), 0, 1)
    return np.log1p(mu * x) / np.log1p(mu)


def compute_metrics(gt_hdr_rgb: np.ndarray, pred_hdr_rgb: np.ndarray) -> dict:
    max_val = float(max(gt_hdr_rgb.max(), pred_hdr_rgb.max(), 1e-8))

    psnr_linear = psnr(gt_hdr_rgb, pred_hdr_rgb, max_val=max_val)

    gt_mu = mu_law(gt_hdr_rgb, max_val)
    pred_mu = mu_law(pred_hdr_rgb, max_val)
    psnr_mu = psnr(gt_mu, pred_mu, max_val=1.0)

    gt_tm = tonemap_reinhard_u8(gt_hdr_rgb)
    pred_tm = tonemap_reinhard_u8(pred_hdr_rgb)
    psnr_tm = psnr(gt_tm, pred_tm, max_val=255.0)
    ssim_tm = float(ssim_fn(gt_tm, pred_tm, channel_axis=-1, data_range=255))

    rmse_linear = float(np.sqrt(np.mean(
        (gt_hdr_rgb.astype(np.float64) - pred_hdr_rgb.astype(np.float64)) ** 2
    )))

    dr_gt = float(measure_ev_range(gt_hdr_rgb))
    dr_pred = float(measure_ev_range(pred_hdr_rgb))

    return {
        'psnr_linear_db': float(psnr_linear),
        'psnr_mu_db': float(psnr_mu),
        'psnr_tonemapped_db': float(psnr_tm),
        'ssim_tonemapped': ssim_tm,
        'rmse_linear': rmse_linear,
        'dynamic_range_gt_ev': dr_gt,
        'dynamic_range_pred_ev': dr_pred,
    }


def save_side_by_side(scene: str, gt_hdr: np.ndarray, pred_hdr: np.ndarray,
                       under: np.ndarray, base: np.ndarray, over: np.ndarray,
                       out_dir: Path) -> None:
    gt_tm = tonemap_reinhard_u8(gt_hdr)
    pred_tm = tonemap_reinhard_u8(pred_hdr)

    top = np.concatenate([under, base, over], axis=1)
    bottom = np.concatenate([gt_tm, pred_tm, np.zeros_like(pred_tm)], axis=1)
    montage = np.concatenate([top, bottom], axis=0)
    montage_bgr = cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / f'{scene}_merge_compare.png'), montage_bgr)

    pred_bgr = cv2.cvtColor(pred_hdr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / f'{scene}_pred.hdr'), pred_bgr)


def process_scene(scene: str, bracketed_dir: Path, hdr_dir: Path,
                   model, device, ev: float, out_dir: Path) -> dict | None:
    scene_path = bracketed_dir / scene
    ldr_files = sorted(scene_path.glob('*.JPG*'))
    if not ldr_files:
        print(f'[{scene}] brak plikow LDR, pomijam')
        return None

    base_file = ldr_files[len(ldr_files) // 2]
    try:
        exif = get_exif(str(base_file))
        t_base = read_exposure_time(exif)
    except Exception as e:
        print(f'[{scene}] brak EXIF ({e}), pomijam')
        return None

    base_rgb = load_ldr_rgb(base_file)
    under_rgb, over_rgb = synth_under_over(model, base_rgb, device)

    factor = 2.0 ** ev
    times = np.array([t_base / factor, t_base, t_base * factor], dtype=np.float32)

    pred_hdr = merge_debevec_rgb([under_rgb, base_rgb, over_rgb], times)

    gt_path = find_hdr_file(hdr_dir, scene)
    if gt_path is None:
        print(f'[{scene}] brak pliku HDR, pomijam porownanie')
        return None
    gt_hdr = load_hdr_rgb(gt_path)

    metrics = compute_metrics(gt_hdr, pred_hdr)
    metrics['scene'] = scene
    metrics['base_frame'] = base_file.name
    metrics['base_exposure_s'] = float(t_base)

    save_side_by_side(scene, gt_hdr, pred_hdr, under_rgb, base_rgb, over_rgb, out_dir)
    return metrics


def print_table(rows: list[dict]) -> None:
    cols = ['scene', 'psnr_linear_db', 'psnr_mu_db', 'psnr_tonemapped_db',
            'ssim_tonemapped', 'rmse_linear',
            'dynamic_range_gt_ev', 'dynamic_range_pred_ev']
    headers = ['scene', 'PSNR-L', 'PSNR-mu', 'PSNR-TM', 'SSIM-TM',
               'RMSE', 'DR-GT', 'DR-Pred']
    widths = [8, 9, 9, 9, 9, 10, 8, 9]

    line = ' '.join(f'{h:<{w}}' for h, w in zip(headers, widths))
    print(line)
    print('-' * len(line))
    for r in rows:
        vals = [r[c] for c in cols]
        fmt = [f'{vals[0]:<{widths[0]}}']
        for v, w in zip(vals[1:], widths[1:]):
            fmt.append(f'{v:<{w}.4f}')
        print(' '.join(fmt))

    if not rows:
        return
    numeric = cols[1:]
    means = {c: float(np.mean([r[c] for r in rows])) for c in numeric}
    fmt = [f'{"MEAN":<{widths[0]}}']
    for c, w in zip(numeric, widths[1:]):
        fmt.append(f'{means[c]:<{w}.4f}')
    print('-' * len(line))
    print(' '.join(fmt))


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = ExposureSynthesizer().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    data_dir = Path(args.data_dir)
    bracketed_dir = data_dir / 'images' / 'Bracketed_images'
    hdr_dir = data_dir / 'images' / 'HDR'

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scene in tqdm(TEST_SCENES, desc='Scenes'):
        res = process_scene(scene, bracketed_dir, hdr_dir, model,
                             device, args.ev, out_dir)
        if res is not None:
            rows.append(res)

    print('\n=== HDR Reconstruction vs. Ground Truth ===')
    print_table(rows)

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'\nZapisano: {out_dir}/metrics.json')
    print(f'Wizualizacje i pliki .hdr: {out_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Katalog HDR-Eye (z images/Bracketed_images i images/HDR)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Plik .pt z wagami ExposureSynthesizer')
    parser.add_argument('--output_dir', type=str, default='results/merge_compare')
    parser.add_argument('--ev', type=float, default=2.7,
                        help='EV uzyte do treningu (domyslnie 2.7)')
    args = parser.parse_args()

    torch.set_num_threads(1)
    main(args)
