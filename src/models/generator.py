from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self, latent_vec_dim,
        n_gen_fm,
        n_channels
    ):
        """
        latent_vec_dim: latent vector size
        n_gen_fm: N of generator feature maps
        n_channels: N of channels in output image
        """
        super().__init__()
        self.main = nn.Sequential(
            # input is latent_vec_dim, going into a convolution
            nn.ConvTranspose2d(latent_vec_dim, n_gen_fm * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_gen_fm * 8),
            nn.ReLU(True),
            # state size: (n_gen_fm*8) x 4 x 4
            nn.ConvTranspose2d(n_gen_fm * 8, n_gen_fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_gen_fm * 4),
            nn.ReLU(True),
            # state size: (n_gen_fm*4) x 8 x 8
            nn.ConvTranspose2d(n_gen_fm * 4, n_gen_fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_gen_fm * 2),
            nn.ReLU(True),
            # state size: (n_gen_fm*2) x 16 x 16
            nn.ConvTranspose2d(n_gen_fm * 2, n_gen_fm, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_gen_fm),
            nn.ReLU(True),
            # state size: (n_gen_fm) x 32 x 32
            nn.ConvTranspose2d(n_gen_fm, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # output is state size: (n_channels) x 64 x 64
        )
        
    def forward(self, input):
        return self.main(input)