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
    epochs: int = attr.ib(default=5)
    lr: float = attr.ib(default=0.0002)
    n_channels: int = attr.ib(default=3)
    latent_vec_dim: int = attr.ib(default=100) # Gen input
    n_gen_fm: int = attr.ib(default=64) # Gen feature map
    n_disc_fm: int = attr.ib(default=64) # Disc feature map
    beta1: float = attr.ib(default=0.5)
    ngpu: int = attr.ib(default=0)
    
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
