# libraries 
import torch
import torch.nn as nn
import os

from VAE_c import VAE, kl_divergence_loss, show_tensor_images

# libraries for importing data
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

# libraries for visualizing the image
from tqdm import tqdm
import time

import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, args):

        self.lr = args.lr
        self.batch_size = args.batch_size
        self.device = args.device
        self.model_save_path = args.model_save_path
        self.numEpochs = args.numEpochs

        self.buildModel()

    def train(self):
        transform=transforms.Compose([transforms.ToTensor(),])
        mnist_dataset = datasets.MNIST('.', train=True, transform=transform,download=True)
        train_dataloader = DataLoader(mnist_dataset, shuffle=True, batch_size=self.batch_size)
        for epoch in range(self.numEpochs):
            print(f"Epoch {epoch}")
            time.sleep(0.5)
            for images, _ in tqdm(train_dataloader):
                images = images.to(self.device)
                self.vae_opt.zero_grad() # Clear out the gradients
                recon_images, encoding, _ = self.vae(images)
                loss = self.reconstruction_loss(recon_images, images) + kl_divergence_loss(encoding).sum()
                loss.backward()
                self.vae_opt.step()

            plt.subplot(1,2,1)
            show_tensor_images(images)
            plt.title("True")
            plt.subplot(1,2,2)
            show_tensor_images(recon_images)
            plt.title("Reconstructed")
            plt.show()

        torch.save(self.vae.state_dict(),os.path.join(self.model_save_path, 'vae.pth'))


    def buildModel(self):
        self.vae = VAE().to(self.device)
        self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=self.lr)
        self.reconstruction_loss = nn.BCELoss(reduction='sum')


