# libraries 
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import os
import numpy as np


# libraries for visualizing the image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class Env(nn.Module):
    def __init__(self, args, model_G, model_classifier, model_decoder):
        super(Env, self).__init__()

        # generator model
        self.generator = model_G
        # classifier model for caluclating the reward
        self.classifier = model_classifier
        # decoder model
        self.decoder = model_decoder

        self.device = args.device

        self.save_path = os.path.join(args.save_path, 'RL_train')
        os.makedirs(self.save_path, exist_ok=True)

    def reset(self):
        self.count =0
        pass

    def agent_input(self, input):
        with torch.no_grad():
            input = input.cuda(non_blocking=True)
            input_var = Variable(input, requires_grad=True)
            out = input_var.detach().cpu().numpy().squeeze()
        return out

    def forward(self,state ,action, episodeTarget,saveFig = False,t=None):

        # episodeTarget: the number that the RL agent is trying to find

        with torch.no_grad():

            # action
            z = Variable(action, requires_grad=True).cuda().squeeze()

            genOut,_ = self.generator(z)
            genImage = self.decoder(genOut)
            classification = self.classifier(genImage)

        # batch size:
        batchSize = len(episodeTarget)

        # reward based on the classifier
        reward = 10 * np.exp(classification[0:1:batchSize,episodeTarget].cpu().data.numpy().squeeze())
        # the nextState 
        nextState = genOut.detach().cpu().data.numpy().squeeze()

        done = True

        if(saveFig):
            plt.figure()
            show_tensor_images(genImage)
            plt.title("reward: {}".format(reward))
            plt.savefig(os.path.join(self.save_path, "time_{}_number_{}".format(t,self.count)))
            plt.show()

            self.count += 1


        return nextState, reward, done






def show_tensor_images(image_tensor, num_images=1, size=(1, 28, 28)):
    """
    Function for visualizing images

    Parameters
    ----------
    image_tensor : torch.Tensor
        batch of images to visualize
    num_images : int
        number of images
    size : tuple
        size of images
    """
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=1)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())


