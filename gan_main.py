from torch.backends import cudnn

import argparse
import sys
import torch
from gan_trainer import Trainer
from gan_tester import Tester
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from AE import AutoEncoder, show_tensor_images


class toLatentTsfm(object):
    """
    class for transforming the images to the latent space
    """
    def __init__(self, ae_model, device):
        """
        initialize transform to encode mnist dataset

        Parameters
        ----------
        ae_model : torch.nn.Module
            auto-encoder model
        device : torch.device
            device to load model to (cpu, gpu)
        """
        # the vae model for transofrmation
        self.ae = ae_model
        # set the vae to the eval mode
        self.ae.eval()
        self.device = device

    def __call__(self, sample):
        image = sample
        image = image.reshape((1,) + tuple(image.shape)).to(self.device)
        reconImage, z_sample = self.ae(image)
        return z_sample


def str2bool(v):
    """
    Parameters
    ----------
    v : str
        string of False or True 
    """
    return v.lower() in ('true')


def parse_args(args):
    """
    parse args

    Parameters
    ----------
    args : 
        passed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_path', default='G.pth')


    # pretrained_path: path to pretrained model dict
    # adv_loss: type of loss for gan trainer
    # l_size: latent size of auto-encoder
    # z_dim: input size of generator
    # g_conv_dim, d_conv_dim: channel number of attention layer for generator and discriminator
    # lambda_gp: gradient penalty
    parser.add_argument('--adv_loss', default='hinge', type=str, choices=['wgan-gp', 'hinge'])  #
    parser.add_argument('--l_size', default=32, type=int)
    
    parser.add_argument('--z_dim', type=int, default=1)

    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--lambda_gp', type=float, default=10, help='gradient penalty')

    # GPU settings
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu ids: e.g. 0, 1. -1 is no GPU')

    # Training setting
    parser.add_argument('--total_step', type=int, default=1000000, help='how many times to update the generator')
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', type=float, default=0.0)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)

    # model step to load and continue model training
    parser.add_argument('--pretrained_num', type=int, default=None)

    # if test: ---train False
    parser.add_argument('--train', type=str2bool, default=True)
    
    # Paths
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--result_dir', type=str, default='./checkPoints')

    # Step size
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--save_step', type=float, default=1.0)

    return parser.parse_args(args)


if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)

    """ GPU training """
    cudnn.benchmark = True
    args.device = torch.device(
        "cuda:%d" % args.gpu_id if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu_id)

    # latent_loader auto-encoder used to create data loader for latent space of mnist
    ae = AutoEncoder()
    ae.load_state_dict(torch.load('models/ae.pth'))
    ae.to(args.device)
    transform = transforms.Compose([transforms.ToTensor(), toLatentTsfm(ae, args.device)])
    mnistEncoded = datasets.MNIST('.', train=True, transform=transform)
    latent_loader = DataLoader(mnistEncoded, shuffle=True, batch_size=args.batch_size)

    # only used for visualization
    model_decoder = ae.decode

    if args.train:
        trainer = Trainer(args, latent_loader, model_decoder, show_tensor_images)
        trainer.train()
    else:
        tester = Tester(args, model_decoder, show_tensor_images)
        evaluater = tester.evaluate()
        for i in range(10):
            next(evaluater)
