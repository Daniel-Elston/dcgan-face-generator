from __future__ import annotations

from dataclasses import dataclass, field

from config.request import RequestConfig
from config.model import HyperParams, ModelState
from config.data import DataParams, DataState
from config.db import DatabaseConfig, DatabaseConnManager
from config.paths import PathsConfig

@dataclass
class StateManager:
    paths: PathsConfig = field(default_factory=PathsConfig)
    api_config: RequestConfig = field(default_factory=RequestConfig)
    data_config: DataParams = field(default_factory=DataParams)
    data_state: DataState = field(default_factory=DataState)
    # db_config: DatabaseConfig = field(default_factory=DatabaseConfig)
    model_config: HyperParams = field(default_factory=HyperParams)
    model_state: ModelState = field(default_factory=ModelState)

    # db_manager: DatabaseConnManager = field(init=False)

    # def __post_init__(self):
    #     self.initialize_database()

    # def initialize_database(self):
    #     self.db_manager = DatabaseConnManager(self.db_config)
