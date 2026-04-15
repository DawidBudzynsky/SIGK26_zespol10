import cv2
import numpy as np
from numpy import ndarray
from PIL import Image
from PIL.ExifTags import TAGS


def read_hdr(image_path: str) -> ndarray:
    hdr_image = cv2.imread(
        filename=image_path,
        flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR
    )
    hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
    return hdr_image


def get_exif(image_path: str) -> dict:
    image = Image.open(image_path)
    info = image._getexif()
    if info is None:
        raise ValueError(f"Missing metadata for {image_path}")
    exif_data = {}
    for tag, value in info.items():
        decoded = TAGS.get(tag)
        if not decoded:
            continue
        exif_data[decoded] = value
    return exif_data


def _calculate_luminance(hdr_image: ndarray) -> ndarray:
    return (0.2126 * hdr_image[:, :, 0] +
            0.7152 * hdr_image[:, :, 1] +
            0.0722 * hdr_image[:, :, 2])


def _filter_pixels(luminance: ndarray, epsilon: float) -> ndarray:
    return luminance[luminance > epsilon]


def measure_ev_range(hdr_image: ndarray, epsilon: float = 1e-6) -> float:
    valid_pixels = _filter_pixels(
        luminance=_calculate_luminance(hdr_image),
        epsilon=epsilon
    )
    luminance_max = np.max(valid_pixels)
    luminance_min = np.min(valid_pixels)
    return np.log2(luminance_max / luminance_min)


def tone_map_reinhard(hdr_image: ndarray) -> ndarray:
    tonemap_operator = cv2.createTonemapReinhard(
        gamma=2.2,
        intensity=0.0,
        light_adapt=0.0,
        color_adapt=0.0
    )
    return tonemap_operator.process(src=hdr_image)


def read_exposure_time(exif_data: dict) -> float:
    exposure = exif_data.get('ExposureTime')
    if exposure is None:
        raise ValueError("No ExposureTime in EXIF")
    if isinstance(exposure, tuple):
        return exposure[0] / exposure[1] if exposure[1] != 0 else 1.0
    return float(exposure)


def apply_exposure_adjustment(image: ndarray, ev: float) -> ndarray:
    factor = 2 ** ev
    adjusted = image.astype(np.float32) * factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)
