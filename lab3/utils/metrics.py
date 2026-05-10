from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity as sk_ssim


def to_uint8(img: np.ndarray) -> np.ndarray:
    """CHW/HWC, float[0,1] lub uint8 -> HWC uint8."""
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


_flip = None
def flip_score(pred: np.ndarray, gt: np.ndarray) -> float:
    global _flip
    if _flip is None:
        import flip_evaluator
        _flip = flip_evaluator
    _, mean, _ = _flip.evaluate(to_uint8(gt), to_uint8(pred), "LDR")
    return float(mean)


_lpips_model = None
_lpips_dev = None
def lpips_score(pred: np.ndarray, gt: np.ndarray, device: str = "cuda") -> float:
    global _lpips_model, _lpips_dev
    if _lpips_model is None or _lpips_dev != device:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex").to(device).eval()
        _lpips_dev = device

    def _t(img):
        x = to_uint8(img).astype(np.float32) / 255.0
        x = x * 2.0 - 1.0
        return torch.from_numpy(np.transpose(x, (2, 0, 1))[None]).to(device)

    with torch.no_grad():
        return float(_lpips_model(_t(pred), _t(gt)).item())


def ssim_score(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(sk_ssim(to_uint8(gt), to_uint8(pred), channel_axis=2, data_range=255))


def _canny_points(img_u8: np.ndarray) -> np.ndarray:
    import cv2
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.argwhere(edges > 0).astype(np.float32)


def hausdorff_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Symetryczna odległość Hausdorffa na krawędziach Cannego.

    Jeśli któryś obraz nie ma żadnych krawędzi (np. czarna predykcja),
    zwracamy przekątną kadru, żeby metryka pozostała skończona —
    `inf` zatruwałoby średnie. To miejsce trzeba mieć z tyłu głowy
    przy interpretacji wyników (omówione w README).
    """
    p, g = to_uint8(pred), to_uint8(gt)
    ep = _canny_points(p); eg = _canny_points(g)
    h, w = p.shape[:2]
    diag = float(np.hypot(h, w))
    if len(ep) == 0 or len(eg) == 0:
        return diag
    return float(max(directed_hausdorff(ep, eg)[0], directed_hausdorff(eg, ep)[0]))


def all_metrics(pred: np.ndarray, gt: np.ndarray, device: str = "cuda") -> dict:
    return {
        "FLIP":      flip_score(pred, gt),
        "LPIPS":     lpips_score(pred, gt, device=device),
        "SSIM":      ssim_score(pred, gt),
        "Hausdorff": hausdorff_score(pred, gt),
    }
