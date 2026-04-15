import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from pathlib import Path


class HDREyeDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', 
                 target_ev: float = 2.7, transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_ev = target_ev
        self.transform = transform
        
        self.ldr_dir = self.data_dir / 'Bracketed images' / 'LDR'
        self.hdr_dir = self.data_dir / 'Bracketed images' / 'HDR'
        
        self.test_scenes = ['C40', 'C41', 'C42', 'C43', 'C44', 'C45', 'C46']
        
        if split == 'test':
            self.samples = self._get_test_samples()
        else:
            self.samples = self._get_train_samples()
    
    def _get_train_samples(self):
        samples = []
        for scene_dir in self.ldr_dir.iterdir():
            if scene_dir.is_dir() and scene_dir.name not in self.test_scenes:
                for ldr_file in sorted(scene_dir.glob('*.jpg')):
                    samples.append((ldr_file, None))
        return samples
    
    def _get_test_samples(self):
        samples = []
        for scene in self.test_scenes:
            scene_path = self.ldr_dir / scene
            if scene_path.exists():
                for ldr_file in sorted(scene_path.glob('*.jpg')):
                    hdr_path = self.hdr_dir / f"{scene}.hdr"
                    samples.append((ldr_file, hdr_path))
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        ldr_path, hdr_path = self.samples[idx]
        
        ldr_img = Image.open(ldr_path).convert('RGB')
        ldr_array = np.array(ldr_img).astype(np.float32) / 255.0
        
        if self.transform:
            ldr_array = self.transform(ldr_array)
        
        ldr_tensor = torch.from_numpy(ldr_array).permute(2, 0, 1)
        
        if self.split == 'test' and hdr_path:
            hdr_array = self._read_hdr(hdr_path)
            hdr_tensor = torch.from_numpy(hdr_array).permute(2, 0, 1)
            return ldr_tensor, hdr_tensor, str(ldr_path)
        
        return ldr_tensor, ldr_path.name
    
    def _read_hdr(self, hdr_path):
        hdr = cv2.imread(str(hdr_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
        return hdr
