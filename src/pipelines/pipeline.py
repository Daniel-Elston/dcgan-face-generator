from __future__ import annotations

from config.state_init import StateManager
from utils.execution import TaskExecutor
from src.data.data_module import CelebDataModule
from config.data import DataParams
from config.model import HyperParams

from src.plots.visuals import ViewTrainLoader
from src.models.dcgan import DCGAN
from src.models.train import TrainDCGAN

class Pipeline:
    def __init__(
        self, state: StateManager,
        exe: TaskExecutor
    ):
        self.state = state
        self.exe = exe
        
        self.params = DataParams()
        self.hyperparams = HyperParams()

        self.data = CelebDataModule(
            dataroot = self.state.paths.get_path("raw"),
            params = self.params
        )
        # self.dataset = self.data.setup()
        # self.dataloader = self.data.train_dataloader()
        
        self.model = DCGAN(self.hyperparams)


    def __call__(self):
        steps = [
            # ViewTrainLoader(self.dataloader).view_dataloader,
            TrainDCGAN(
                self.state,
                self.hyperparams,
                self.data,
                self.model
            )
        ]
        self.exe._execute_steps(steps, stage="parent")
