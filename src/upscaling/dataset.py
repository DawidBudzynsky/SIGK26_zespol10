import os
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


def resize_cv2(image, size, interpolation=cv2.INTER_CUBIC):
    if isinstance(size, int):
        size = (size, size)
    image_np = np.array(image)
    resized = cv2.resize(image_np, size, interpolation=interpolation)
    return Image.fromarray(resized)


class UpscaleDataset(Dataset):

    def __init__(
        self,
        root_dir,
        low_res_size=32,
        high_res_size=256,
        low_res_interpolation=cv2.INTER_CUBIC,
    ):
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        self.low_res_interpolation = low_res_interpolation

        self.transform = transforms.ToTensor()

        self.low_images = []
        self.high_images = []

        root_dir = os.path.expanduser(root_dir)
        for fname in sorted(os.listdir(root_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root_dir, fname)
                image = Image.open(image_path).convert("RGB")

                image_high = image.resize(
                    (high_res_size, high_res_size), Image.Resampling.LANCZOS
                )

                image_low = image.resize(
                    (low_res_size, low_res_size),
                    Image.Resampling.BICUBIC,
                )

                self.low_images.append(np.array(image_low))
                self.high_images.append(np.array(image_high))

    def __len__(self):
        return len(self.high_images)

    def __getitem__(self, idx):
        low = Image.fromarray(self.low_images[idx])
        high = Image.fromarray(self.high_images[idx])

        low = self.transform(low)
        high = self.transform(high)

        return low, high
