# libraries 
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

# libraries for importing data
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

# libraries for visualizing the image
from torchvision.utils import make_grid

import matplotlib.pyplot as plt




class Encoder(nn.Module):
    """Encoder Class"""
    def __init__(self, im_chan=1, output_chan=32, hidden_dim=16):
        """
        initialize Encoder model

        Parameters
        ----------
        im_chan : int
            channel number of input image
        output_chan : int
            channel number of encoded output (latent space)
        hidden_dim : int
            channel number of output of first hidden block layer (a measure of model capacity)
        """
        super(Encoder, self).__init__()
        self.z_dim = output_chan
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, output_chan, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        """
        Function to return a sequence of operations corresponding to a encoder block of the VAE, 
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation
        
        Parameters
        ----------
        input_channels : int
            how many channels the input feature representation has
        output_channels : int 
            how many channels the output feature representation should have
        kernel_size : int 
            the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride : int 
            the stride of the convolution
        final_layer : bool 
            whether we're on the final layer (affects activation and batchnorm)
        """       
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        """
        Function for completing a forward pass of the Encoder: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.

        Parameters
        ----------
        image: torch.Tensor
            a flattened image tensor with dimension (im_dim)

        Returns
        -------
        torch.Tensor
            encoded result (latent space)

        """
        disc_pred = self.disc(image)
        encoding = disc_pred.view(len(disc_pred), -1)
        return encoding


class Decoder(nn.Module):
    """Decoder Class"""
    def __init__(self, z_dim=32, im_chan=1, hidden_dim=64):
        """
        initialize Decoder model
        
        Parameters
        ----------
        z_dim:  int
            the dimension of the noise vector
        im_chan: int
            channel number of the output image, MNIST is black-and-white, so that's our default
        hidden_dim : int
            channel number of input of last hidden block layer (a measure of model capacity)
        """
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        """
        Function to return a sequence of operations corresponding to a Decoder block of the VAE, 
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation
        
        Parameters
        ----------
        input_channels : int
            how many channels the input feature representation has
        output_channels : int
            how many channels the output feature representation should have
        kernel_size : int
            the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride : int
            the stride of the convolution
        final_layer : bool
            whether we're on the final layer (affects activation and batchnorm)
        """
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Sigmoid(),
            )

    def forward(self, noise):
        """
        Function for completing a forward pass of the Decoder: Given a noise vector, 
        returns a generated image.

        Parameters
        ----------
        noise: torch.Tensor
            a noise tensor with dimensions (batch_size, z_dim)

        Returns
        -------
        torch.Tensor
            recostructed result (decoded image)
        """
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)



class AutoEncoder(nn.Module):
    """
    Auto-Encoder Class
    """
    def __init__(self, z_dim=32, im_chan=1, hidden_dim=64):
        """
        initialize Auto-Encoder model
        
        Parameters
        ----------
        z_dim:  int
            the dimension of the noise vector
        im_chan: int
            channel number of the output image, MNIST is black-and-white, so that's our default
        hidden_dim : int
            channel number of hidden block layer (a measure of model capacity)
        """
        super(AutoEncoder, self).__init__()
        self.z_dim = z_dim
        self.encode = Encoder(im_chan, z_dim)
        self.decode = Decoder(z_dim, im_chan)

    def forward(self, images):
        """
        Function for completing a forward pass of the Decoder: Given a noise vector, 
        returns a generated image.
        
        Parameters
        ----------
        images : torch.Tensor 
            an image tensor with dimensions (batch_size, im_chan, im_height, im_width)

        Returns
        -------
        decoding : torch.Tensor
            the reconstructed image
        z_sample : torch.Tensor
            encoded result
        """
        z_sample = self.encode(images)
        decoding = self.decode(z_sample)
        return decoding, z_sample

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    """
    Function for visualizing images

    Parameters
    ----------
    image_tensor : torch.Tensor
        batch of images to visualize
    num_images : int
        number of images
    size : tuple
        size of images
    """
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())



def kl_divergence_loss(q_dist):
    """
    to calculate kl divergence distance for loss
    """
    return kl_divergence(
        q_dist, Normal(torch.zeros_like(q_dist.mean), torch.ones_like(q_dist.stddev))
    ).sum(-1)


