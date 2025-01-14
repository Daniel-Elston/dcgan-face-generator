from __future__ import annotations

import pytorch_lightning as pl
import torchvision.utils as vutils

class GenerateCallback(pl.Callback):
    def __init__(self, sample_noise):
        super().__init__()
        self.sample_noise = sample_noise

    def on_epoch_end(self, trainer, pl_module):
        noise = self.sample_noise.to(pl_module.device)

        fake_images = pl_module.generator(noise)
        grid = vutils.make_grid(
            fake_images,
            normalize=True,
            value_range=(-1, 1)
        )

        trainer.logger.experiment.add_image(
            "Generated Images",
            grid,
            global_step=trainer.current_epoch
        )
        # trainer.logger.experiment.flush()