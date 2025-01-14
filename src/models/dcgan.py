from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.models.generator import Generator
from src.models.discriminator import Discriminator
from config.model import HyperParams

import torchvision.utils as vutils
import matplotlib.pyplot as plt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN(pl.LightningModule):
    def __init__(
        self, hyperparams: HyperParams
    ):
        """
        The DCGAN model
        hyperparams: HyperParams
        """
        super().__init__()
        self.automatic_optimization = False
        self.hyperparams = hyperparams
        # self.save_hyperparameters(ignore=['hyperparams'])
        # self.hyperparams.update(vars(hyperparams))
        # self.save_hyperparameters()
        self.save_hyperparameters(vars(hyperparams))
        self.sample_noise = torch.randn(64, self.hyperparams.latent_vec_dim, 1, 1)
        
        # ------------------
        # Instantiate networks
        # ------------------
        self.generator = Generator(
            latent_vec_dim=self.hyperparams.latent_vec_dim,
            n_gen_fm=self.hyperparams.n_gen_fm,
            n_channels=self.hyperparams.n_channels
        )
        self.discriminator = Discriminator(
            n_disc_fm=self.hyperparams.n_disc_fm,
            n_channels=self.hyperparams.n_channels
        )
        
        # Apply weight initialization
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
        # loss function
        self.criterion = nn.BCELoss()

    def forward(self, z):
        """
        For generation/inference: just pass noise to the generator
        """
        return self.generator(z)

    def configure_optimizers(self):
        # Two optimizers: D & G
        lr = self.hyperparams.lr
        betas = (self.hyperparams.beta1, 0.999)
        
        opt_D = optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=betas
        )
        opt_G = optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=betas
        )
        return [opt_D, opt_G], []

    def training_step(self, batch, batch_idx):
        """
        Manual optimization with multiple optimizers in a single step.
        We'll do:
          1) Discriminator update
          2) Generator update
        """
        images, _ = batch
        batch_size = images.size(0)
        device = images.device

        # valid = 1, fake = 0
        valid_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # -----------------------------------------------------
        # (1) Train Discriminator
        # -----------------------------------------------------
        # Get optimizer references
        optD, optG = self.optimizers()

        # ---- Train on real images
        real_preds = self.discriminator(images)
        d_real_loss = self.criterion(real_preds.view(-1), valid_labels.view(-1))

        # ---- Train on fake images
        noise = torch.randn(batch_size, self.hyperparams.latent_vec_dim, 1, 1, device=device)
        fake_images = self.generator(noise)
        fake_preds = self.discriminator(fake_images.detach())
        d_fake_loss = self.criterion(fake_preds.view(-1), fake_labels.view(-1))

        # Combine D losses
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Backprop and update D
        self.manual_backward(d_loss)  # instead of d_loss.backward()
        optD.step()
        optD.zero_grad()

        # -----------------------------------------------------
        # (2) Train Generator
        # -----------------------------------------------------
        # We do *not* get a new batch here, we reuse same noise
        # or generate a new one
        noise = torch.randn(batch_size, self.hyperparams.latent_vec_dim, 1, 1, device=device)
        fake_images = self.generator(noise)
        fake_preds = self.discriminator(fake_images)

        # Our generator wants the discriminator to guess "valid"
        g_loss = self.criterion(fake_preds.view(-1), valid_labels.view(-1))

        self.manual_backward(g_loss)
        optG.step()
        optG.zero_grad()

        # Logging
        self.log("D_loss", d_loss, prog_bar=True)
        self.log("G_loss", g_loss, prog_bar=True)

        return {"loss": (d_loss + g_loss) / 2.0}
    
    def on_epoch_end(self):
        noise = self.sample_noise.to(self.device)

        fake_images = self.generator(noise)
        grid = vutils.make_grid(
            fake_images,
            normalize=True,
            range=(1,1)
        )

        np_grid = grid.cpu().numpy().transpose(1,2,0)
        plt.figure(figsize=(8,8))
        plt.imshow(np_grid)
        plt.title("Generated Images")
        plt.axis("off")
        plt.savefig(f"reports/results/generated_e{self.current_epoch}.png")
        plt.close()
