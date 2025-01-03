from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


class FileAccess:
    @staticmethod
    def extract_suffix(path: Path):
        return path.suffix

    @staticmethod
    def load_file(path: Path):
        path = Path(path)
        suffix = FileAccess.extract_suffix(path)
        logging.debug(f"Reading file: ``{path}``")
        if suffix == ".parquet":
            return pd.read_parquet(path)
        elif suffix == ".csv":
            return pd.read_csv(path)
        elif suffix == ".xlsx":
            return pd.read_excel(path)
        elif suffix == ".json":
            return pd.read_json(path)
        else:
            raise ValueError(f"Unknown file type: {suffix}")

    @staticmethod
    def save_file(df: pd.DataFrame, path: Path):
        suffix = FileAccess.extract_suffix(path)
        if suffix == ".parquet":
            return df.to_parquet(path, index=False)
        elif suffix == ".csv":
            return df.to_csv(path, index=False)
        elif suffix == ".xlsx":
            return df.to_excel(path, index=False)
        elif suffix == ".json":
            return df.to_json(path)
        else:
            raise ValueError(f"Unknown file type: {path} {suffix}")

    @staticmethod
    def load_json(path):
        with open(path, "r") as file:
            return json.load(file)

    @staticmethod
    def save_json(data, path, overwrite=False):
        if overwrite is False and Path(path).exists():
            logging.warning(f"File already exists: ``{path}``")
        else:
            logging.debug(f"Saving json to ``{path}``")
            with open(path, "w") as file:
                json.dump(data, file)
