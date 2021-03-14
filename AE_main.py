import argparse
import sys
import torch
import os 

from AE_trainer import Trainer
from AE import AutoEncoder, show_tensor_images
import matplotlib.pyplot as plt

# libraries for importing data
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms


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


    # learning parameters lr: learning rate
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--numEpochs', default=10, type=int)

    # if test: ---train False
    parser.add_argument('--train', default=True, type=str2bool)

<<<<<<< HEAD
    # save paths
=======
    # Paths
>>>>>>> 3f51161fb6afda5b00f3231c7345b34e6ce5f53e
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
    test_dataloader = DataLoader(mnist_dataset, shuffle=True, batch_size=args.batch_size)


    if(args.train):
        trainer = Trainer(args,train_dataloader)
        trainer.train()
    else:
        # For evaluating the model
        ae = AutoEncoder().to(args.device)
        ae.load_state_dict(torch.load(os.path.join(args.model_save_path, 'ae.pth'), map_location=torch.device(args.device)))
        images = iter(test_dataloader).next()[0].to(args.device)
        plt.figure()
        ae.eval()
        
        # outputs of model: reconstructed image, latent space (encoded)
        recon_images, z_samples= ae(images)
        
        # visualizing results
        plt.figure()
        plt.subplot(1,2,1)
        show_tensor_images(images)
        plt.title("True")
        plt.subplot(1,2,2)
        show_tensor_images(recon_images)
        plt.title("Reconstructed")
        plt.savefig(os.path.join(args.eval_checkPoints, "eval"))
        plt.show()

