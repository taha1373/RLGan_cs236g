import torch
import os
import numpy as np
import time
import datetime
from torch.autograd import Variable
import torch.nn as nn
from torchvision.utils import save_image
from gan import Generator, Discriminator
from collections import OrderedDict
from utils import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class Tester(object):
    def __init__(self, args, model_decoder, visualizer, input_loader=None):

        # decoder settings
        self.device = args.device

        self.model_decoder = model_decoder
        self.vis = visualizer

        # Data loader
        if input_loader is not None:
            self.input_loader = iter(input_loader)
        else:
            self.input_loader = None

        # Model hyper-parameters
        self.l_size = args.l_size
        self.z_dim = args.z_dim
        self.g_conv_dim = args.g_conv_dim

        self.batch_size = args.batch_size
        self.pretrained_path = args.pretrained_path

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.build_model()
        self.load_pretrained_model(self.pretrained_path)

    def evaluate(self):
        out_number = 0
        while True:

            if self.input_loader is not None:
                print('input loader') 
                try:
                    input_z = next(self.input_loader)[0]
                except:
                    break
            else:
                input_z = tensor2var(torch.randn(self.batch_size, self.z_dim))

            # ================== Run G ================== #
            latent, _ = self.G(input_z)
            encoded = latent.contiguous().view(self.batch_size, self.l_size)

            img = self.model_decoder(encoded)
            self.vis(img, self.batch_size)
            plt.title("GAN result")
            plt.savefig(os.path.join(self.save_dir, str(out_number)))
            out_number += 1
            plt.show()

            yield img, latent


    def build_model(self):
        self.G = Generator(self.l_size, self.z_dim, self.g_conv_dim).to(self.device)
        self.G.eval()
        print(self.G)
 

    def load_pretrained_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path + ' not found')
        self.G.load_state_dict(torch.load(model_path))
        print('loaded trained model')