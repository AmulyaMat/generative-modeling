import argparse
import torch
from cleanfid import fid
from matplotlib import pyplot as plt


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    ##################################################################
    # TODO: 1.2: Generate and save out latent space interpolations.
    # 1. Generate 100 samples of 128-dim vectors. Do so by linearly
    # interpolating for 10 steps across each of the first two
    # dimensions between -1 and 1. Keep the rest of the z vector for
    # the samples to be some fixed value (e.g. 0).
    # 2. Forward the samples through the generator.
    # 3. Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.
    ##################################################################
    linspace = torch.linspace(-1, 1, 10)
    grid_x, grid_y = torch.meshgrid(linspace, linspace)
    grid_x, grid_y = grid_x.flatten(), grid_y.flatten()

    # Initialize the rest of the z vector (dim-2 dimensions) with zeros or any fixed value
    fixed_values = torch.zeros((10**2, 128 - 2))

    # Concatenate the interpolated values with the fixed part of the z vector
    z = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1), fixed_values], dim=1).to(gen.device)

    # Generate images
    images = gen.forward_given_samples(z)  # Assuming gen returns images in the range [-1, 1]
    images = (images + 1) / 2  # Rescale images to [0, 1]

    # Save the images in a grid
    save_image(images, path, nrow=10)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()
    return args
