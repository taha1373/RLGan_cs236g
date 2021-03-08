import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision.utils import save_image
from gan import Generator, Discriminator
from utils import *
import matplotlib.pyplot as plt


class Tester(object):
    """Generator tester"""
    def __init__(self, args, model_decoder, visualizer, input_loader=None):
        """
        initialize Generator tester

        Parameters
        ----------
        args : dict
            args dictionary look at :func: '~gan_main.parse_args'
        model_decoder : torch.nn.Module
            decoder model used in auto-decoder
        visualizer : any
            method for visualizing results
        input_loader : iterator
            generator input loader
        """

        # decoder settings

        # setting device
        self.device = args.device

        # setting model decoder and visualizer
        self.model_decoder = model_decoder
        self.vis = visualizer

        # Data loader if no data loader is present random data are generated
        if input_loader is not None:
            self.input_loader = iter(input_loader)
        else:
            self.input_loader = None

        # Model hyper-parameters l_size: latent size (output of generator), z_dim: input size of generator,
        # g_conv_dim: channel size of attention layer in generator (a measure for model capacity)
        self.l_size = args.l_size
        self.z_dim = args.z_dim
        self.g_conv_dim = args.g_conv_dim

        # setting batchsize and pretrained model path
        self.batch_size = args.batch_size
        self.pretrained_path = args.pretrained_path

        # directory to save results in
        self.result_path = os.path.join(args.result_dir, 'GAN_eval')
        os.makedirs(self.result_path, exist_ok=True)

        self.build_model()
        self.load_pretrained_model(self.pretrained_path)

    def evaluate(self):
        """
        a generator (python iterator) that creates latent results and saves them

        Yields
        ------
        img : torch.Tensor
            output result of decoder with input latent (output of generator model)
        latent : torch.Tensor
            output of generator model

        Examples
        --------

        >>> tester = Tester(args, model_decoder, show_tensor_images)
        >>> print('start')
        >>> evaluater = tester.evaluate()
        >>> for i in range(10):
            ... print('evaluate')
            ... next(evaluater)
        """

        # name of result for saving
        out_number = 0
        while True:

            # if loader present create until data finished, else random generation of input data
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

            # run decoder to get image results
            img = self.model_decoder(encoded)

            # visualize result
            self.vis(img, self.batch_size)
            plt.title("GAN result")
            # save result
            plt.savefig(os.path.join(self.result_path, str(out_number)))
            out_number += 1
            plt.show()
            yield img, latent

    def build_model(self):
        """
        builds generator model for evaulation
        """
        self.G = Generator(self.l_size, self.z_dim, self.g_conv_dim).to(self.device)
        self.G.eval()
        print(self.G)

    def load_pretrained_model(self, model_path):
        """
        load pretrained model

        Parameters
        ----------
        model_path : str
            model path (e.g. G.th)

        Raises
        ------
        FileNotFoundError
            if no model dict is present
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path + ' not found')
        self.G.load_state_dict(torch.load(model_path))
        print('loaded trained model')