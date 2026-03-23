import numpy as np


def nlm_denoise(image, h=0.1, search_window=10, patch_window=3):
    if len(image.shape) == 4:
        image = image.squeeze(0)

    if isinstance(image, np.ndarray):
        if image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))

        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = _nlm_filter_single_channel(
                image[:, :, c], h, search_window, patch_window
            )

        if result.shape[2] in [1, 3]:
            result = np.transpose(result, (2, 0, 1))
        return result
    return image


def _nlm_filter_single_channel(img, h, search_window, patch_window):
    rows, cols = img.shape
    result = np.zeros_like(img)

    half_search = search_window // 2
    half_patch = patch_window // 2

    pad_width = half_search + half_patch
    padded_img = np.pad(img, pad_width, mode="edge")

    for i in range(rows):
        for j in range(cols):
            i_pad = i + pad_width
            j_pad = j + pad_width

            central_patch = padded_img[
                i_pad - half_patch : i_pad + half_patch + 1,
                j_pad - half_patch : j_pad + half_patch + 1,
            ]

            weights_sum = 0
            weighted_sum = 0

            for di in range(-half_search, half_search + 1):
                for dj in range(-half_search, half_search + 1):
                    ni = i_pad + di
                    nj = j_pad + dj

                    neighbor_patch = padded_img[
                        ni - half_patch : ni + half_patch + 1,
                        nj - half_patch : nj + half_patch + 1,
                    ]

                    diff = central_patch - neighbor_patch
                    weight = np.exp(-np.sum(diff**2) / (h**2 * patch_window**2))

                    weighted_sum += weight * padded_img[ni, nj]
                    weights_sum += weight

            result[i, j] = weighted_sum / (weights_sum + 1e-10)

    return result
