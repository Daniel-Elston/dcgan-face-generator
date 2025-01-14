from __future__ import annotations

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(
        self, n_channels=3,
        n_disc_fm=64
    ):
        """
        n_channels: N of channels in input image
        n_disc_fm: N of discriminator feature maps
        """
        super().__init__()
        self.main = nn.Sequential(
            # input is (n_channels) x 64 x 64
            nn.Conv2d(n_channels, n_disc_fm, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (n_disc_fm) x 32 x 32
            nn.Conv2d(n_disc_fm, n_disc_fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_disc_fm * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (n_disc_fm*2) x 16 x 16
            nn.Conv2d(n_disc_fm * 2, n_disc_fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_disc_fm * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (n_disc_fm*4) x 8 x 8
            nn.Conv2d(n_disc_fm * 4, n_disc_fm * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_disc_fm * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (n_disc_fm*8) x 4 x 4
            nn.Conv2d(n_disc_fm * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # output is scalar probability
        )

    def forward(self, x):
        return self.main(x)