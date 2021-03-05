import argparse
import sys
import torch
import os 

from VAE_trainer import Trainer
from VAE_c import VAE, show_tensor_images
import matplotlib.pyplot as plt

# libraries for importing data
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms


def str2bool(v):
    return v.lower() in ('true')


def parse_args(args):
    parser = argparse.ArgumentParser()


    # learning parameters
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--numEpochs', default=10, type=int)

    # Misc
    parser.add_argument('--train', default=True, type=str2bool)

    # save path:s
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--train_checkPoints', type=str, default='./checkPoints/AE_train')
    parser.add_argument('--eval_checkPoints', type=str, default='./checkPoints/AE_eval')

    return parser.parse_args(args)


if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)

    transform=transforms.Compose([transforms.ToTensor(),])

    # Train datasets 
    mnist_dataset = datasets.MNIST('.', train=True, transform=transform,download=True)
    train_dataloader = DataLoader(mnist_dataset, shuffle=True, batch_size=args.batch_size)

    # Test datasets
    mnist_dataset = datasets.MNIST('.', train=False, transform=transform,download=True)
    test_dataloader = DataLoader(mnist_dataset, shuffle=True, batch_size=1)


    if(args.train):
        trainer = Trainer(args,train_dataloader)
        trainer.train()
    else:
        # For evaluating the model
        vae = VAE().to(args.device)
        vae.load_state_dict(torch.load(os.path.join(args.model_save_path, 'vae.pth'), map_location=torch.device(args.device)))
        sample = iter(test_dataloader).next()[0][0].reshape(1,1,28,28).to(args.device)
        plt.figure()
        show_tensor_images(sample)
        plt.savefig(os.path.join(args.eval_checkPoints, "sample"))
        vae.eval()
        recon_images, q_dist, _ = vae(sample)
        for i in range(5):
            z_sample = q_dist.rsample()
            decoded= vae.decode(z_sample)
            plt.figure()
            show_tensor_images(decoded)
            plt.savefig(os.path.join(args.eval_checkPoints, "decode_{}".format(i)))

