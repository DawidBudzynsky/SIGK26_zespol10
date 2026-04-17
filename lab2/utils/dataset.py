import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from utils.hdr import get_exif


def check_exif(path):
    try:
        get_exif(str(path))
        return True
    except:
        return False


class HDREyeDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', resize=(512, 512)):
        self.data_dir = Path(data_dir)
        self.split = split
        self.resize = resize
        self.bracketed_dir = self.data_dir / 'images' / 'Bracketed_images'
        self.hdr_dir = self.data_dir / 'images' / 'HDR'
        
        # Test scenes are C40-C46
        self.test_scenes = set(['C40', 'C41', 'C42', 'C43', 'C44', 'C45', 'C46'])
        
        # Find scenes with EXIF
        self.good_scenes = []
        for scene_dir in sorted(self.bracketed_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if not scene_dir.name.startswith('C'):
                continue
            files = list(scene_dir.glob('*.JPG*'))
            if files and check_exif(files[0]):
                self.good_scenes.append(scene_dir.name)
        
        if split == 'test':
            self.samples = self._get_test_samples()
        else:
            # Exclude C40-C46 from training
            train_scenes = [s for s in self.good_scenes if s not in self.test_scenes]
            self.samples = self._get_train_samples(train_scenes)
        
        print(f'{split}: {len(self.samples)} samples from {len(self.good_scenes)} scenes (EXIF good)')
    
    def _get_train_samples(self, allowed_scenes):
        samples = []
        for scene_dir in sorted(self.bracketed_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if scene_dir.name not in allowed_scenes:
                continue
            for ldr_file in sorted(scene_dir.glob('*.JPG*')):
                samples.append((ldr_file, None))
        return samples
    
    def _get_test_samples(self):
        samples = []
        for scene in sorted(self.test_scenes):
            scene_path = self.bracketed_dir / scene
            if scene_path.exists():
                for ldr_file in sorted(scene_path.glob('*.JPG*')):
                    hdr_path = self.hdr_dir / f'{scene}_HDR.hdr'
                    samples.append((ldr_file, hdr_path))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ldr_path, hdr_path = self.samples[idx]
        
        ldr_img = Image.open(ldr_path).convert('RGB')
        if self.resize:
            ldr_img = ldr_img.resize(self.resize, Image.LANCZOS)
        ldr_array = np.array(ldr_img).astype(np.float32) / 255.0
        ldr_tensor = torch.from_numpy(ldr_array).permute(2, 0, 1)
        
        if self.split == 'test' and hdr_path:
            hdr = cv2.imread(str(hdr_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
            hdr_tensor = torch.from_numpy(hdr).permute(2, 0, 1)
            return ldr_tensor, hdr_tensor, str(ldr_path)
        
        return ldr_tensor, ldr_path.name
