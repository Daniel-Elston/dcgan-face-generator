from __future__ import annotations

import attr
import logging
from pprint import pformat


@attr.s
class DataParams:
    overwrite: bool = attr.ib(default=True)
    save_fig: bool = attr.ib(default=True)
    subset: bool = attr.ib(default=True)
    shuffle: bool = attr.ib(default=True)
    img_size: int = attr.ib(default=64)
    batch_size: int = attr.ib(default=128)
    num_workers: int = attr.ib(default=2)

    def __attrs_post_init__(self):
        attr_dict = attr.asdict(self)
        logging.debug(f"DataConfig:\n{pformat(attr_dict)}\n")


class DataState:
    def __init__(self):
        self._state = {}

    def set(self, key, value):
        """Store a value in the state"""
        self._state[key] = value

    def get(self, key):
        """Retrieve a value from the state"""
        return self._state.get(key)

    def clear(self):
        """Clear all state"""
        self._state = {}
