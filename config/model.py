from __future__ import annotations

import logging
from pprint import pformat
import torch
import attr


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@attr.s
class HyperParams:
    device: str = attr.ib(factory=get_device)
    num_classes: int = attr.ib(default=10)


    def __attrs_post_init__(self):
        attr_dict = attr.asdict(self)
        logging.debug(f"ModelConfig:\n{pformat(attr_dict)}\n")


class ModelState:
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
