import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.util import random_noise

SIGMA_GAUSS=0.01
SIZE=256
SIGMA=0.02

def apply_gaussian_noise(image, sigma=SIGMA_GAUSS):
    image_np = np.array(image) / 255.0
    noisy = random_noise(image_np, mode="gaussian", var=sigma**2)
    return (noisy * 255).astype(np.uint8)


def resize_stretch(image, size=SIZE):
    return image.resize((size, size), Image.Resampling.LANCZOS)


class NoisyImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, resize=resize_stretch, sigma=SIGMA):
        self.sigma = sigma
        self.resize = resize if resize else (lambda x: x)
        self.images = []
        self.noisy_images = []

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        root_dir = os.path.expanduser(root_dir)
        for fname in sorted(os.listdir(root_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root_dir, fname)
                image = Image.open(image_path).convert("RGB")
                image = self.resize(image)
                noisy = apply_gaussian_noise(image, sigma=self.sigma)
                self.images.append(np.array(image))
                self.noisy_images.append(noisy)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        clean = Image.fromarray(self.images[idx])
        noisy = Image.fromarray(self.noisy_images[idx])

        noisy = self.transform(noisy)
        clean = self.transform(clean)

        return noisy, clean
