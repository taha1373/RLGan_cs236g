# libraries
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import os
import numpy as np
from utils import show_tensor_images

# libraries for visualizing the image
import matplotlib.pyplot as plt


class Env(nn.Module):
    def __init__(self, args, model_G, model_D, model_classifier, model_decoder):
        """
        environment model of RL

        Parameters
        ----------
        args : dict
            args dictionary look at :func: '~gan_main.parse_args'
        model_G : torch.nn.Module
            generator model
        model_D : torch.nn.Module
            discriminator model
        model_classifier : torch.nn.Module
            classifier model
        model_decoder : torch.nn.Module
            decoder model
        """
        super(Env, self).__init__()

        # generator model
        self.generator = model_G

        # disciminator model
        self.disciminator = model_D

        # classifier model for caluclating the reward
        self.classifier = model_classifier
        # decoder model
        self.decoder = model_decoder

        self.device = args.device

        # for calculating the discriminator reward
        self.hinge = torch.nn.HingeEmbeddingLoss()

        self.count = 0
        self.save_path = os.path.join(args.save_path, 'RL_train')
        os.makedirs(self.save_path, exist_ok=True)

    def reset(self):
        """
        reset env by setting count to zero
        """
        self.count = 0

    def agent_input(self, input):
        """
        detach input from tensor to numpy (get numpy state)
        Parameters
        ----------
        input : torch.Tensor
            tensor of input state

        Returns
        -------
        numpy.ndarray
            numpy array of state
        """
        return input.detach().cpu().numpy().squeeze()

    def forward(self, action, episode_target, save_fig=False, t=None):
        """
        calling environment to step and give next state and reward

        Parameters
        ----------
        action : torch.Tensor
            tensor of action array
        episode_target : int
            target label agent must try to produce
        save_fig : bool
            to save image
        t : int
            time step

        Returns
        -------
        tuple
            next_state, reward, is_episode_finished
        """
        # episodeTarget: the number that the RL agent is trying to find

        with torch.no_grad():
            # action
            z = Variable(action, requires_grad=True).to(self.device).squeeze()

            gen_out, _ = self.generator(z)
            dis_judge, _ = self.disciminator(gen_out)
            gen_image = self.decoder(gen_out)
            classification = self.classifier(gen_image)

        batch_size = len(episode_target)

        # reward based on the classifier
        reward_cl = 30 * np.exp(classification[0:1:batch_size, episode_target].cpu().data.numpy().squeeze())
        reward_d = -10 * self.hinge(dis_judge, -1 * torch.ones_like(dis_judge)).cpu().data.numpy().squeeze()

        reward = reward_cl + reward_d

        # the nextState
        next_state = gen_out.detach().cpu().data.numpy().squeeze()

        done = True

        if save_fig:
            plt.figure(num=1, figsize=(5, 5))
            show_tensor_images(gen_image)
            plt.title("target: {}, reward: {}".format(episode_target, reward))
            plt.savefig(os.path.join(self.save_path, "time_{}_number_{}".format(t, self.count)))
            # plt.show()

            f = open(os.path.join(self.save_path, "time_{}_number_{}.txt".format(t, self.count)), "w")
            f.write("reward_cl: {}   reward_d:{}".format(reward_cl, reward_d))
            f.close()

            self.count += 1

        return next_state, reward, done