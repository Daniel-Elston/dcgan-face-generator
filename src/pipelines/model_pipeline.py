from __future__ import annotations

from config.data import DataParams
from config.model import HyperParams
from config.state_init import StateManager
from src.data.data_module import CelebDataModule
from src.models.dcgan import DCGAN
from src.models.train import TrainDCGAN
from utils.execution import TaskExecutor


class ModelPipeline:
    """
    Model training pipeline:
        - Train model running ``main.py``
        - View results by running ``gen_imgs_from_cp.py``
        - View logs via tensorboard
        - Generated imgs saved to: ``reports/results``
    """
    def __init__(
        self, state: StateManager,
        exe: TaskExecutor
    ):
        self.state = state
        self.exe = exe

        self.params = DataParams()
        self.hyperparams = HyperParams()

        self.data = CelebDataModule(
            dataroot=self.state.paths.get_path("raw"),
            params=self.params
        )

        self.model = DCGAN(self.hyperparams)

    def __call__(self):
        steps = [
            TrainDCGAN(
                self.state,
                self.hyperparams,
                self.data,
                self.model
            )
        ]
        self.exe._execute_steps(steps, stage="parent")
