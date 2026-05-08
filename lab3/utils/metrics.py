"""Evaluation metrics for the neural renderer (Projekt 3, sec. 3).

Implemented:
  - FLIP (Andersson et al. 2020) via the official `flip_evaluator` package
  - LPIPS (Zhang 2018)             via `lpips` (AlexNet backbone)
  - SSIM (Wang et al. 2004)        via scikit-image
  - Hausdorff distance on the Canny edge map of both images
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity as sk_ssim


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a CHW / HWC float-or-uint8 image to HWC uint8."""
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


# ---------- FLIP --------------------------------------------------------------
_flip_module = None
def flip_score(pred: np.ndarray, gt: np.ndarray) -> float:
    global _flip_module
    if _flip_module is None:
        import flip_evaluator
        _flip_module = flip_evaluator
    pred = to_uint8(pred)
    gt = to_uint8(gt)
    _, mean, _ = _flip_module.evaluate(gt, pred, "LDR")
    return float(mean)


# ---------- LPIPS -------------------------------------------------------------
_lpips_model = None
_lpips_device = None
def lpips_score(pred: np.ndarray, gt: np.ndarray, device: str = "cuda") -> float:
    global _lpips_model, _lpips_device
    if _lpips_model is None or _lpips_device != device:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex").to(device).eval()
        _lpips_device = device

    def _to_tensor(img):
        img = to_uint8(img).astype(np.float32) / 255.0   # [0,1]
        img = img * 2.0 - 1.0                            # [-1,1]
        img = np.transpose(img, (2, 0, 1))[None]
        return torch.from_numpy(img).to(device)

    with torch.no_grad():
        d = _lpips_model(_to_tensor(pred), _to_tensor(gt))
    return float(d.item())


# ---------- SSIM --------------------------------------------------------------
def ssim_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = to_uint8(pred)
    gt = to_uint8(gt)
    return float(sk_ssim(gt, pred, channel_axis=2, data_range=255))


# ---------- Hausdorff on Canny edges -----------------------------------------
def _canny_points(img_uint8: np.ndarray) -> np.ndarray:
    import cv2
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    pts = np.argwhere(edges > 0).astype(np.float32)   # (N, 2) of (y, x)
    return pts


def hausdorff_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Symmetric Hausdorff distance between the two images' Canny edge sets.

    If one of the two has no edges (pure-black render), we return the diagonal
    of the image as a worst-case distance so the metric stays finite.
    """
    pred = to_uint8(pred)
    gt = to_uint8(gt)
    p = _canny_points(pred)
    g = _canny_points(gt)
    h, w = pred.shape[:2]
    diag = float(np.hypot(h, w))
    if len(p) == 0 or len(g) == 0:
        return diag
    d1 = directed_hausdorff(p, g)[0]
    d2 = directed_hausdorff(g, p)[0]
    return float(max(d1, d2))


def all_metrics(pred: np.ndarray, gt: np.ndarray, device: str = "cuda") -> dict:
    return {
        "FLIP":     flip_score(pred, gt),
        "LPIPS":    lpips_score(pred, gt, device=device),
        "SSIM":     ssim_score(pred, gt),
        "Hausdorff": hausdorff_score(pred, gt),
    }
