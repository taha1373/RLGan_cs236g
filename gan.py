import torch
import torch.nn as nn
from spectral import SpectralNorm
import numpy as np


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation):
        """
        initialize self-attention layer
        initializes key, query and value layers, gamma coefficient and softmax layer

        Parameters
        ----------
        in_dim : int
            channel number of input
        activation : str
            activation type (e.g. 'relu')
        """
        super(Self_Attn, self).__init__()
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        calculate key, query and value for each couple in input (number of couples N x N)

        Parameters
        ----------
        x : torch.Tensor
            input feature maps (shape: B X C X W X H)

        Returns
        -------
        out : torch.Tensor
            self attention value + input feature (shape: B X C X W X H)
        attention: torch.Tensor
            attention coefficients (shape: B X N X N) (N is W * H)
        """
        batch, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch, -1, width * height).permute(0, 2, 1)  # shape: B x N x (C // 8)
        proj_key = self.key_conv(x).view(batch, -1, width * height)  # shape: B x (C // 8) x N
        energy = torch.bmm(proj_query, proj_key)  # shape: B x N x N
        attention = self.softmax(energy)  # shape: B x N x N
        proj_value = self.value_conv(x).view(batch, -1, width * height)  # shape: B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # shape: B x C x N
        out = out.view(batch, C, width, height)  # shape: B x C x W x H

        # calculate attention + input
        out = self.gamma * out + x  # shape: B x C x W x H
        return out, attention


class Generator(nn.Module):
    """Generator."""
    def __init__(self, out_size=64, z_dim=100, conv_dim=64):
        """
        initialize Generator model based on self-attention gan (https://arxiv.org/abs/1805.08318)

        Parameters
        ----------
        out_size : int
            size of output of model
        z_dim : int
            size of input (latent) of model
        conv_dim : int
            channel number of attention layer
        """
        super(Generator, self).__init__()
        self.outsize = out_size

        # layers use spectral norm based on (https://arxiv.org/abs/1805.08318) with deconvolution
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        # layer1  output channel number
        out_dim = conv_dim * (2 ** 2)

        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, out_dim, 4)))
        layer1.append(nn.BatchNorm2d(out_dim))
        layer1.append(nn.ReLU())

        layer2.append(SpectralNorm(nn.ConvTranspose2d(out_dim, int(out_dim / 2), 3, 2, 2)))
        layer2.append(nn.BatchNorm2d(int(out_dim / 2)))
        layer2.append(nn.ReLU())

        # layer2 output channel number
        out_dim = int(out_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(out_dim, int(out_dim / 2), 3, 2, 2)))
        layer3.append(nn.BatchNorm2d(int(out_dim / 2)))
        layer3.append(nn.ReLU())

        # layer3 output channel number
        out_dim = int(out_dim / 2)

        # creating layers
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        # creating self-attention layer
        self.attn = Self_Attn(out_dim, 'relu')

        # creating final layer with tanh activation
        last.append(nn.ConvTranspose2d(out_dim, 1, 2, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        # get output with a fully connected layer
        self.input1d2d = nn.ConvTranspose1d(144, self.outsize, 1)

    def forward(self, z):
        """
        generate from latent variable

        Parameters
        ----------
        z : torch.Tensor
            input of size B x z_dim

        Returns
        -------
        out : torch.Tensor
            out of size B x 1 x 1 x out_size

        p1: torch.Tensor
            attention coefficients (shape: B X 49 X 49)
        """
        # Taha changed:
        try:
            z = z.view(z.size(0), z.size(1), 1, 1)  # shape B x z_dim x 1 x 1
        except:
            # for batch size : 1
            z = z.view(1, z.size(0), 1, 1)
        out = self.l1(z)  # B x z_dim x 1 x 1 ---> B x (conv_dim * 4) x 4 x 4
        out = self.l2(out)  # ---> B x (conv_dim * 4) x 7 x 7 ---> B x (conv_dim * 2) x 5 x 5
        out = self.l3(out)  # ---> B x (conv_dim * 2) x 7 x 7 ---> B x conv_dim x 7 x 7
        out, p1 = self.attn(out)  # ---> B x conv_dim x 7 x 7 ---> B x conv_dim x 7 x 7
        out = self.last(out)  # B x conv_dim x 7 x 7 ---> B x 1 x 12 x 12

        out = out.view(-1, 1, 144)  # B x 1 x 12 x 12 ---> B x 1 x 1 x 144
        out = out.transpose(1, 2)  # B x 1 x 1 x 144 ---> B x 1 x 144 x 1
        out = self.input1d2d(out)  # B x 1 x 144 x 1 ---> B x 1 x out_size x 1
        out = out.transpose(2, 1)  # B x 1 x out_size x 1 ---> B x 1 x 1 x out_size

        out = out.view(-1, 1, 1, 32) # B x 1 x 1 x out_size ---> B x 1 x 1 x out_size
        return out, p1  # output, attention coefficients


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""
    def __init__(self, input_size=64, conv_dim=64):
        """
        initialize Discriminator model based on self-attention gan (add ref)

        Parameters
        ----------
        input_size : int
            channel number of input (generator output)
        conv_dim : int
            channel number of attention layer
        """

        super(Discriminator, self).__init__()

        self.inputsize = input_size

        # convert input channel number to suitable number (144) with a fully connected layer
        self.input1d2d = nn.ConvTranspose1d(self.inputsize, 144, 1)

        # layers use spectral norm based on (add ref) with deconvolution
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(1, conv_dim, 3, 2, 2)))
        layer1.append(nn.LeakyReLU(0.1))

        # layer1  output channel number
        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 3, 2, 2)))  # 4,2,1
        layer2.append(nn.LeakyReLU(0.1))

        # layer2  output channel number
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 3, 2, 2)))
        layer3.append(nn.LeakyReLU(0.1))

        # layer3  output channel number
        curr_dim = curr_dim * 2

        # creating layers
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        # creating self-attention layer
        self.attn = Self_Attn(curr_dim, 'relu')

        # creating final layer
        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.Tensor
            input of size B x 1 x 1 x input_size

        Returns
        -------
        out : torch.Tensor
            input of size B x 1 x 1 x 1
        p1: torch.Tensor
            attention coefficients (shape: B X 49 X 49)
        """
        x = x.squeeze(1)  # B x 1 x 1 x input_size ---> B x 1 x input_size
        x = x.transpose(1, 2)  # B x 1 x input_size ---> B x input_size x 1
        x = self.input1d2d(x)  # B x input_size x 1 ---> B x 144 x 1
        x = x.transpose(2, 1)  # B x 144 x 1 ---> B x 1 x 144
        x = x.view(-1, 1, 12, 12)  # B x 1 x 144 ---> B x 1 x 12 x 12
        out = self.l1(x)  # B x 1 x 12 x 12 ---> B x conv_dim x 7 x 7
        out = self.l2(out)  # B x conv_dim x 7 x 7 ---> B x (conv_dim * 2) x 5 x 5
        out = self.l3(out)  # B x (conv_dim * 2) x 5 x 5 ---> B x (conv_dim * 4) x 4 x 4
        out, p1 = self.attn(out)  # B x (conv_dim * 4) x 4 x 4 ---> B x (conv_dim * 4) x 4 x 4
        out = self.last(out)  # B x (conv_dim * 4) x 4 x 4 ---> B x 1 x 1 x 1

        return out.squeeze(), p1   # output (classifier), attention coefficients


if __name__ == '__main__':
    from utils import tensor2var
    g = Generator(out_size=32, z_dim=1, conv_dim=16)
    d = Discriminator(input_size=32, conv_dim=16)
    e, g1 = g(tensor2var(torch.FloatTensor([[1.]])))
    w, d1 = d(e)
    print(w.dim())
    print('finished')
