from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
from pprint import pformat
from src.data.data_dict import DataDictionary
from utils.file_access import FileAccess

class DataModule:
    def __init__(
        self, data_path: Path,
        sdo_path: Path,
        data_dict: DataDictionary
    ):
        self.data_path = data_path
        self.sdo_path = sdo_path
        self.dd = data_dict

    def load(self):
        logging.debug(
            f"Running DataModule for {self.data_path}.\n{pformat(self.dd.data)}")
        self.prepare_data()
        df = self.load_data()
        df = self.apply_data_dict(df)
        self.to_parquet(df)
        return df
    
    def prepare_data(self):
        """Download/Extract data"""
        pass

    def load_data(self):
        """Load data to memory"""
        return FileAccess.load_file(self.data_path)

    def apply_data_dict(self, df):
        """Apply data dictionary transforms"""
        transforms = self.dd.transforms_store()
        for func in transforms.values():
            df = func(df)
        return df
    
    def to_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        FileAccess.save_file(df, self.sdo_path)