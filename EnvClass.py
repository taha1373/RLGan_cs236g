# libraries 
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import os


class Env(object):
    def __init__(self, args, model_G, model_classifier, model_decoder):
        super(Env, self).__init__()

        # generator model
        self.generator = model_G
        # classifier model for caluclating the reward
        self.classifier = model_classifier
        # decoder model
        self.decoder = model_decoder

        # weight of the reward
        self.weight = args.weight

        self.device = args.device

    def reset(self):
        pass

    def agent_input(self, input):
        with torch.no_grad():
            input = input.cuda(async=True)
            input_var = Variable(input, requires_grad=True)
            out = input_var.detach().cpu().numpy().squeeze()
        return out

    def forward(self, action, episodeTarget):

        # episodeTarget: the number that the RL agent is trying to find

        with torch.no_grad():

            # action
            z = Variable(action, requires_grad=True).cuda()

            genOut = self.generator(z)
            genImage = self.decoder(genOut)
            classification = self.classifier(genImage)

 
        # reward based on the classifier
        reward = self.weight * classification[:,episodeTarget]
        # the nextState 
        nextState = genOut

        return nextState, reward









