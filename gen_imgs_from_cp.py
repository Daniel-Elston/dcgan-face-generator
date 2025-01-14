#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

from config.model import HyperParams
from src.models.dcgan import DCGAN


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from a DCGAN checkpoint.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the .ckpt file from PyTorch Lightning.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/results",
        help="Directory to save generated image(s). Defaults to 'reports/results'.",
    )
    parser.add_argument(
        "--num_images", type=int, default=64, help="Number of images to generate in a grid."
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="generated.png",
        help="Output PNG filename. Defaults to 'generated.png'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    my_hparams = HyperParams(
        latent_vec_dim=100,
        n_gen_fm=64,
        n_disc_fm=64,
        n_channels=3,
        lr=0.0002,
        beta1=0.999,
        epochs=5,
        device="cuda",
        ngpu=1,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = DCGAN.load_from_checkpoint(
        args.checkpoint_path,
        hyperparams=my_hparams,
    )
    model.eval()
    model.freeze()

    # 3) Generate noise and produce fake images
    latent_dim = model.hyperparams.latent_vec_dim
    noise = torch.randn(args.num_images, latent_dim, 1, 1, device="cuda")

    with torch.no_grad():
        fake_images = model.generator(noise)

    grid = vutils.make_grid(fake_images, normalize=True, value_range=(-1, 1))

    np_grid = grid.cpu().numpy().transpose((1, 2, 0))

    plt.figure(figsize=(8, 8))
    plt.imshow(np_grid)
    plt.title("Generated Images")
    plt.axis("off")

    save_path = os.path.join(args.output_dir, args.save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved generated image grid to: {save_path}")


if __name__ == "__main__":
    main()
