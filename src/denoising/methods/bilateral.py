import numpy as np
from skimage.restoration import denoise_bilateral


def bilateral_denoise(image, spatial_sigma=1.5, intensity_sigma=0.1):
    if isinstance(image, np.ndarray):
        if image.ndim == 4:
            image = image.squeeze(0)

        input_was_chw = image.shape[0] in [1, 3]

        if input_was_chw:
            image = np.transpose(image, (1, 2, 0))

        if image.max() <= 1.0:
            result = denoise_bilateral(
                image,
                sigma_color=intensity_sigma,
                sigma_spatial=spatial_sigma,
                channel_axis=-1,
            )
        else:
            result = (
                denoise_bilateral(
                    image / 255.0,
                    sigma_color=intensity_sigma,
                    sigma_spatial=spatial_sigma,
                    channel_axis=-1,
                )
                * 255.0
            )

        if input_was_chw and result.shape[-1] in [1, 3]:
            result = np.transpose(result, (2, 0, 1))

        return result.astype(np.float32)
    return image
