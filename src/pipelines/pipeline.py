from __future__ import annotations

from config.state_init import StateManager
from utils.execution import TaskExecutor
from src.data.data_module import CelebDataModule
from config.data import DataParams

from src.plots.visuals import ViewTrainLoader


class Pipeline:
    def __init__(
        self, state: StateManager,
        exe: TaskExecutor
    ):
        self.state = state
        self.exe = exe
        
        self.params = DataParams()

        self.data = CelebDataModule(
            dataroot = self.state.paths.get_path("raw"),
            params = self.params
        )
        self.dataset = self.data.setup()
        self.dataloader = self.data.train_dataloader()


    def __call__(self):
        steps = [
            ViewTrainLoader(self.dataloader).view_dataloader
        ]
        self.exe._execute_steps(steps, stage="parent")
