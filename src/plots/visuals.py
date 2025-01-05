from __future__ import annotations

import logging
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np

import torchvision.utils as vutils


class ViewTrainLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
    
    def view_dataloader(self):
        batch = next(iter(self.dataloader))
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
        plt.savefig("reports/figures/training_images2.png")
        plt.show()
