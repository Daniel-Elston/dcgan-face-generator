from __future__ import annotations

import logging
from pathlib import Path

from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

from config.data import DataParams


class CelebDataModule:
    def __init__(
        self, dataroot: Path,
        params: DataParams
    ):
        self.dataroot = dataroot
        self.params = params
    
    def setup(self):
        transform = transforms.Compose([
            transforms.Resize(self.params.img_size),
            transforms.CenterCrop(self.params.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.dataset = dset.ImageFolder(root=self.dataroot, transform=transform)
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers
        )
