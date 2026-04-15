import torch
import numpy as np
from numpy import ndarray


def compute_psnr(img1: ndarray, img2: ndarray, max_value: float = 255.0) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_value / np.sqrt(mse))


def compute_lpips_tensor(img1: torch.Tensor, img2: torch.Tensor, lpips_model, device: str = 'cuda') -> float:
    img1 = img1.to(device)
    img2 = img2.to(device)
    with torch.no_grad():
        dist = lpips_model(img1, img2)
    return dist.item()


def tensor_to_numpy(tensor: torch.Tensor) -> ndarray:
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = tensor.cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return (arr * 255).astype(np.uint8)


def numpy_to_tensor(array: ndarray, device: str = 'cuda') -> torch.Tensor:
    array = array.astype(np.float32) / 255.0
    if array.ndim == 3:
        array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0).to(device)
