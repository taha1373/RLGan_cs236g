import argparse
import sys
import torch
import os 

from MNISTClassifier_trainer import Trainer
from MNISTClassifier import Classifier, show_tensor_images
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


    # learning parameters
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta_1', type=float, default=0.5)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--numEpochs', default=10, type=int)
    parser.add_argument('--numLabels', default=10, type=int)

    # if test: ---train False
    parser.add_argument('--train', default=True, type=str2bool)

    # Path
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--train_checkPoints', type=str, default='./checkPoints/MNISTClassifier_train')
    parser.add_argument('--eval_checkPoints', type=str, default='./checkPoints/MNISTClassifier_eval')

    return parser.parse_args(args)


if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)

    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

    # Train datasets 
    mnist_dataset = datasets.MNIST('.', train=True, transform=transform,download=True)
    train_dataloader = DataLoader(mnist_dataset, shuffle=True, batch_size=args.batch_size)

    # Test datasets
    mnist_dataset = datasets.MNIST('.', train=False, transform=transform,download=True)
    test_dataloader = DataLoader(mnist_dataset, shuffle=True, batch_size=args.batch_size)


    if(args.train):
        trainer = Trainer(args,train_dataloader)
        trainer.train()
        trainer.test(test_dataloader)
    else:
        pass
        # # For evaluating the model
        # ae = AutoEncoder().to(args.device)
        # ae.load_state_dict(torch.load(os.path.join(args.model_save_path, 'ae.pth'), map_location=torch.device(args.device)))
        # images = iter(test_dataloader).next()[0].to(args.device)
        # plt.figure()
        # ae.eval()
        # recon_images, z_samples= ae(images)
        # plt.figure()
        # plt.subplot(1,2,1)
        # show_tensor_images(images)
        # plt.title("True")
        # plt.subplot(1,2,2)
        # show_tensor_images(recon_images)
        # plt.title("Reconstructed")
        # plt.savefig(os.path.join(args.eval_checkPoints, "eval"))
        # plt.show()

