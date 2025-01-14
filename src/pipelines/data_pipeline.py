from __future__ import annotations

from config.data import DataParams
from config.model import HyperParams
from config.state_init import StateManager
from src.data.data_module import CelebDataModule
from src.plots.visuals import ViewTrainLoader
from utils.execution import TaskExecutor


class DataPipeline:
    """
    Data pipeline:
        - Visualise raw img data and train loader
        - Switch ``config/config.yaml`` -> "console_level: DEBUG" to view tensors
        - Train imgs saved to: ``reports/figures``
    """

    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe

        self.params = DataParams()
        self.hyperparams = HyperParams()

        self.data = CelebDataModule(
            dataroot=self.state.paths.get_path("raw"),
            params=self.params
        )
        self.dataset = self.data.setup()
        self.dataloader = self.data.train_dataloader()

    def __call__(self):
        steps = [
            ViewTrainLoader(self.dataloader).view_dataloader,
        ]
        self.exe._execute_steps(steps, stage="parent")
