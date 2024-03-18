import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO 1.3: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ####
    # discrim_real and discrim_fake are each a tensor with values of the predictions
    # discrim_real is a tensor with predictions when REAL images are input into the discriminator --> ideally 1
    # discrim_fake is a tensor with predictions when FAKE images are output from the generator and 
    # passed into the discriminator --> ideally 0
    # 0 means fake, 1 means real  --> raw logits

    # for discriminator, the total loss shows the overall ability of the discriminator to differentiate
    # between fake and real images
    # real loss is to compare predictions of real images with tensor of ones (ideal)
    # fake loss is to compare predictions of fake images with tensor of zeros (ideal)
    ##################################################################

    real_loss = F.binary_cross_entropy_with_logits(discrim_real, torch.ones_like(discrim_real))
    fake_loss = F.binary_cross_entropy_with_logits(discrim_fake, torch.zeros_like(discrim_fake))
    loss = real_loss + fake_loss
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.3: Implement GAN loss for the generator.
    ##################################################################
    # For generator, aim is to ge the MINIMUM loss between fake and real images
    # loss evaluated predictions of fake images and ideally real images (tensor with all ones)
    ######
    
    loss = F.binary_cross_entropy_with_logits(discrim_fake, torch.ones_like(discrim_fake))
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
