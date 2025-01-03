from __future__ import annotations

import logging
from pathlib import Path
from pprint import pformat

from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


class CelebDataModule:
    def __init__(
        self, dataroot: Path,
        img_size: int,
        batch_size: int,
        num_workers: int
    ):
        self.dataroot = dataroot
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self):
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.dataset = dset.ImageFolder(root=self.dataroot, transform=transform)
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def view_dataloader(self):
        loader = self.train_dataloader()
        batch = next(iter(loader))
        logging.debug(pformat(batch))
        
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(
                    batch[0].to(device="cpu")[:64],
                    padding=2,
                    normalize=True).cpu(), (1,2,0)
            )
        )
        plt.savefig("reports/figures/training_images.png")
        plt.show()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    dataroot = Path("data/raw")
    img_size = 64
    batch_size = 128
    num_workers = 2
    datamodule = CelebDataModule(dataroot, img_size, batch_size, num_workers)
    datamodule.setup()
    datamodule.view_dataloader()