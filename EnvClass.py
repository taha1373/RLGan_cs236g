# libraries 
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import os
import numpy as np

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

    def reset(self):
        pass

    def agent_input(self, input):
        with torch.no_grad():
            input = input.cuda(non_blocking=True)
            input_var = Variable(input, requires_grad=True)
            out = input_var.detach().cpu().numpy().squeeze()
        return out

    def forward(self,state ,action, episodeTarget):

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

        return nextState, reward, done









