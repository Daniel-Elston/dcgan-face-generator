from __future__ import annotations

import numpy as np


class DataDictionary:
    def __init__(self):
        self.data = {
        "dtypes": {
            "age": int,
            "sex": int,
            "cp": int,
            "trtbps": int,
            "output":int
        },
        "use_cols": [
            "age",
            "sex",
            "cp",
            "trtbps",
            "output"
        ],
        "rename_mapping": {
            "age": "a",
            "sex": "s",
            "cp": "c",
            "trtbps": "t"
        },
        "na_values": [
            "?",
            "NA"
        ],
    }
    
    def apply_dtypes(self, df):
        for col, dtype in self.data["dtypes"].items():
            df[col] = df[col].astype(dtype)
        return df
    
    def apply_use_cols(self, df):
        return df[self.data["use_cols"]]
    
    def apply_rename_mapping(self, df):
        return df.rename(columns=self.data["rename_mapping"])
    
    def apply_na_values(self, df):
        for v in self.data["na_values"]:
            df = df.replace(v, np.nan)
        return df

    def transforms_store(self):
        return {
            "apply_dtypes": self.apply_dtypes,
            "apply_use_cols": self.apply_use_cols,
            "apply_rename_mapping": self.apply_rename_mapping,
            "apply_na_values": self.apply_na_values
        }
