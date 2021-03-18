import argparse
import sys
import torch
import torch.nn.parallel
from utils import get_n_params, toLatentTsfm

from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from RL_trainer import Trainer

from AE import AutoEncoder
from gan import Generator, Discriminator
from MNISTClassifier import Classifier


def str2bool(v):
    """
    Parameters
    ----------
    v : str
        string of False or True
    """
    return v.lower() in 'true'


def parse_args(args):
    """
    parse args

    Parameters
    ----------
    args :
        passed arguments
    """
    parser = argparse.ArgumentParser(description='RL Agent Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # arguments for Saving Models
    parser.add_argument('--save_path', default='./checkPoints', help='Path to Checkpoints')

    # Arguments for Torch Data Loader
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('-w', '--workers', type=int, default=8, help='Set the number of workers')

    # Hyper parameters for RL
    parser.add_argument('--attempts', default=5, type=int)  # Number of tries to give to RL Agent
    parser.add_argument("--state_dim", default=32, type=int)  # State Dimesnions
    parser.add_argument("--max_action", default=10, type=int)  # For Normal Distribution 2.5 is feasible ?

    parser.add_argument("--weight", default=10, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--start_timesteps", default=50,  # 1e4
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--save_models", default=True)  # Save Pytorch Models?
    parser.add_argument("--batch_size_actor", default=50, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--max_episodes_steps", default=5, type=int)  # Frequency of delayed policy updates

    parser.add_argument("--d_reward_coeff", default=5, type=float)
    parser.add_argument("--cl_reward_coeff", default=30, type=float)

    # Model Hype-Parameter
    parser.add_argument('--l_size', default=32, type=int)
    parser.add_argument('--z_dim', type=int, default=2)
    parser.add_argument('--g_conv_dim', type=int, default=16)
    parser.add_argument('--d_conv_dim', type=int, default=16)

    # GPU settings
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu ids: e.g. 0, 1. -1 is no GPU')

    # if test: ---train False
    parser.add_argument('--train', type=str2bool, default=True)

    parser.add_argument('--load_model', help='Use saved model.', action='store_true')

    return parser.parse_args(args)


def main(args):
    """ Transforms/ Data Augmentation Tec """
    ae = AutoEncoder()
    ae.load_state_dict(torch.load('models/AE_train/AE_final.pth', map_location=args.device))
    ae.to(args.device)
    transform = transforms.Compose([transforms.ToTensor(), toLatentTsfm(ae, args.device)])
    """-----------------------------------------------Data Loader----------------------------------------------------"""
    # Train datasets
    train_valid_dataset = datasets.MNIST('.', train=True, transform=transform, download=True)
    train_valid_size = len(train_valid_dataset)
    valid_size = int(0.2 * train_valid_size)
    train_size = train_valid_size - valid_size
    train_dataset, val_dataset = random_split(train_valid_dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    valid_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size)

    # Test datasets
    test_dataset = datasets.MNIST('.', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)

    """----------------Model Settings-----------------------------------------------"""

    cl = Classifier()
    cl.load_state_dict(torch.load('models/MNISTclassifier_train/MNISTclassifier_final.pth', map_location=args.device))
    cl.to(args.device)
    cl.eval()

    g_model = Generator(args.l_size, args.z_dim, args.g_conv_dim)
    g_model.load_state_dict(torch.load('models/GAN_train/final_G.pth', map_location=args.device))
    g_model.to(args.device)
    g_model.eval()

    d_model = Discriminator(args.l_size, args.d_conv_dim)
    d_model.load_state_dict(torch.load('models/GAN_train/final_D.pth', map_location=args.device))
    d_model.to(args.device)
    d_model.eval()

    params = get_n_params(ae.encode)
    print('| Number of Encoder parameters [' + str(params) + ']...')

    params = get_n_params(ae.decode)
    print('| Number of Decoder parameters [' + str(params) + ']...')

    if args.train:
        trainer = Trainer(args, train_loader, valid_loader, test_loader, ae.encode, ae.decode, g_model, d_model, cl)
        trainer.train()
    else:
        raise NotImplementedError('test not yet implemented')


if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    args.device = torch.device("cuda:%d" % args.gpu_id if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    print(args)

    main(args)
