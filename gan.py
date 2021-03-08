import torch
import torch.nn as nn
from spectral import SpectralNorm
import numpy as np


# from models.layers import *


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class Generator(nn.Module):
    """Generator."""
    def __init__(self, out_size=64, z_dim=100, conv_dim=64):
        """
        :param out_size: output size of generator
        :param z_dim: input dimension of generator
        :param conv_dim: attention layer dimension
        """
        super(Generator, self).__init__()
        self.outsize = out_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        out_dim = conv_dim * (2 ** 2)

        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, out_dim, 4)))
        layer1.append(nn.BatchNorm2d(out_dim))
        layer1.append(nn.ReLU())

        layer2.append(SpectralNorm(nn.ConvTranspose2d(out_dim, int(out_dim / 2), 3, 2, 2)))
        layer2.append(nn.BatchNorm2d(int(out_dim / 2)))
        layer2.append(nn.ReLU())

        out_dim = int(out_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(out_dim, int(out_dim / 2), 3, 2, 2)))
        layer3.append(nn.BatchNorm2d(int(out_dim / 2)))
        layer3.append(nn.ReLU())

        out_dim = int(out_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        self.attn = Self_Attn(out_dim, 'relu')

        last.append(nn.ConvTranspose2d(out_dim, 1, 2, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.input1d2d = nn.ConvTranspose1d(144, self.outsize, 1)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.l1(z)  # ---> 4 x 4    #---> 4 x 4
        out = self.l2(out)  # ---> 8 x 8 # #---> 5 x 5
        out = self.l3(out)  # ---> 16 x 16 # #---> 7 x 7
        out, p1 = self.attn(out)  # ---> 16 x 16 #---> 7 x 7
        out = self.last(out)  # ---> 32 x 32

        out = out.view(-1, 1, 144)
        out = out.transpose(1, 2)
        out = self.input1d2d(out)
        out = out.transpose(2, 1)

        out = out.view(-1, 1, 1, 32)
        return out, p1  # , p2


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, input_size=64, conv_dim=64):
        """
        :param input_size: input size of discriminator
        :param conv_dim: attention dimension
        """
        super(Discriminator, self).__init__()
        self.inputsize = input_size

        self.input1d2d = nn.ConvTranspose1d(self.inputsize, 144, 1)  # SpectralNorm(nn.ConvTranspose1d(128,144,1))

        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(1, conv_dim, 3, 2, 2)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 3, 2, 2)))  # 4,2,1
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 3, 2, 2)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        self.attn = Self_Attn(curr_dim, 'relu')

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)



    def forward(self, x):
        x = x.squeeze(1)
        x = x.transpose(1, 2)
        x = self.input1d2d(x)
        x = x.transpose(2, 1)
        x = x.view(-1, 1, 12, 12)
        out = self.l1(x)  # ----> 16 x 16    ---> 7 x 7
        out = self.l2(out)  # ----> 8 x 8    -----> 5 x 5
        out = self.l3(out)  # ----> 4 x 4   -----> 4 x4
        out, p1 = self.attn(out)  # ----> 4 x 4  -----> 4 x4
        out = self.last(out)  # 1 x 1

        return out.squeeze(), p1  # , p2


if __name__ == '__main__':
    from utils import tensor2var

    g = Generator(out_size=32, z_dim=1, conv_dim=16)
    d = Discriminator(input_size=32, conv_dim=16)
    e, g1 = g(tensor2var(torch.FloatTensor([[1.]])))
    # e = tensor2var(torch.FloatTensor(np.zeros((10, 1, 1, 32))))
    # print(e.size())
    w, d1 = d(e)
    print(w.dim())
    print('finished')
