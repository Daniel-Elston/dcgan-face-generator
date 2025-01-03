from __future__ import annotations

from config.state_init import StateManager
from utils.execution import TaskExecutor
from src.data.data_module import CelebDataModule

class Pipeline:
    def __init__(
        self, state: StateManager,
        exe: TaskExecutor
    ):
        self.state = state
        self.exe = exe

        self.data = CelebDataModule(
            dataroot = self.state.paths.get_path("raw"),
            img_size = 64,
            batch_size = 128,
            num_workers = 2
        )


    def __call__(self):
        steps = [
        ]
        self.exe._execute_steps(steps, stage="parent")