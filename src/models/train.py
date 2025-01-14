from __future__ import annotations

import torch
import pytorch_lightning as pl

from config.state_init import StateManager
from config.model import HyperParams
from src.models.dcgan import DCGAN
from src.data.data_module import CelebDataModule
from src.models.callbacks import GenerateCallback


class TrainDCGAN:
    def __init__(
        self, state: StateManager,
        hyperparams: HyperParams,
        data_module: CelebDataModule,
        model: DCGAN
    ):
        self.state = state
        self.hyperparams = hyperparams
        self.model = model
        self.data_module = data_module

    def __call__(self):
        trainer = pl.Trainer(
            max_epochs=self.hyperparams.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            deterministic=True,
            callbacks=[GenerateCallback(self.model.sample_noise)]
        )
        trainer.fit(model=self.model, datamodule=self.data_module)
