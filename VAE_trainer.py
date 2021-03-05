# libraries 
import torch
import torch.nn as nn
import os

from VAE_c import VAE, kl_divergence_loss, show_tensor_images


# libraries for visualizing the image
from tqdm import tqdm
import time

import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, args,dataLoader):

        self.lr = args.lr
        self.device = args.device
        self.model_save_path = args.model_save_path
        self.train_checkPoints = args.train_checkPoints
        self.numEpochs = args.numEpochs
        self.dataLoader = dataLoader


        self.buildModel()

    def train(self):

        for epoch in range(self.numEpochs):
            print(f"Epoch {epoch}")
            time.sleep(0.5)
            for images, _ in tqdm(self.dataLoader):
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
            # plt.savefig("epoch_{}.pdf".format(epoch))
            plt.savefig(os.path.join(self.train_checkPoints, "epoch_{}".format(epoch)))
            plt.show()

        torch.save(self.vae.state_dict(),os.path.join(self.model_save_path, 'vae.pth'))


    def buildModel(self):
        self.vae = VAE().to(self.device)
        self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=self.lr)
        self.reconstruction_loss = nn.BCELoss(reduction='sum')


