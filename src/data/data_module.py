from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

from config.data import DataParams


class CelebDataModule(pl.LightningDataModule):
    def __init__(
        self, dataroot: Path,
        params: DataParams
    ):
        super().__init__()
        self.dataroot = dataroot
        self.params = params
        self.dataset = None
    
    def setup(self, stage=None):
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
