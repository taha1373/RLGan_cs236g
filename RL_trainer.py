import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import time

import models
from EnvClass import Env
from collections import OrderedDict

import numpy as np
import os

from torch.autograd.variable import Variable
from utils import ReplayBuffer
from RL import TD3
from torch.utils.tensorboard import SummaryWriter

np.random.seed(5)
# torch.manual_seed(5)


class RL_dataloader:
    """data loader for RL training"""
    def __init__(self, dataloader):
        """
        initialize RL data loader

        Parameters
        ----------
        dataloader : torch.utils.data.dataloader
            torch data loader
        """
        self.loader = dataloader
        self.loader_iter = iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def next_data(self):
        """
        get a state and corresponding target value from dataset

        Returns
        -------
        tuple
            state and target value
        """
        try:
            data, label = next(self.loader_iter)
        except:
            self.loader_iter = iter(self.loader)
            data, label = next(self.loader_iter)
        return data, (label + 1) % 10


def evaluate_policy(policy, dataloader, env, eval_episodes=6, save_fig=False, t=None):
    """
    evaluate policy based on dataloader dataset

    Parameters
    ----------
    policy : TD3
        policy model
    dataloader : RL_dataloader
        data loader
    env : Env
        environment class
    eval_episodes : int
        number of episodes to validate with
    save_fig : bool
        to save figure of result
    t : int
        time step

    Returns
    -------
    float
        average reward
    """
    avg_reward = 0.
    env.reset()

    for i in range(0, eval_episodes):
        input, episodeTarget = dataloader.next_data()
        obs = env.agent_input(input)
        done = False

        while not done:
            # Action By Agent and collect reward
            action = policy.select_action(np.array(obs))
            action = torch.tensor(action).unsqueeze(dim=0)
            new_state, reward, done = env(action, episodeTarget, save_fig, t)
            avg_reward += reward

        if i + 1 >= eval_episodes:
            break

    avg_reward /= eval_episodes

    return avg_reward


class Trainer(object):
    """RL trainer"""

    def __init__(self, args, train_loader, valid_loader, test_loader, model_encoder, model_decoder, model_g, model_d,
                 model_classifier):
        """
        initialize RL trainer

        Parameters
        ----------
        args : dict
            args dictionary look at :func: '~gan_main.parse_args'
        train_loader : torch.utils.data.dataloader
            torch data loader of train dataset
        valid_loader : torch.utils.data.dataloader
            torch data loader of validation dataset
        test_loader : torch.utils.data.dataloader
            torch data loader of test dataset
        model_encoder : torch.nn.Module
            encoder model
        model_decoder : torch.nn.Module
            decoder model
        model_g : torch.nn.Module
            generator model
        model_d : torch.nn.Module
            discriminator model
        """
        self.device = args.device

        self.train_loader = RL_dataloader(train_loader)
        self.valid_loader = RL_dataloader(valid_loader)
        self.test_loader = RL_dataloader(test_loader)

        self.epoch_size = len(self.valid_loader)
        self.max_timesteps = args.max_timesteps

        self.batch_size = args.batch_size
        self.batch_size_actor = args.batch_size_actor
        self.eval_freq = args.eval_freq
        self.save_models = args.save_models
        self.start_timesteps = args.start_timesteps
        self.max_episodes_steps = args.max_episodes_steps

        self.z_dim = args.z_dim
        self.max_action = args.max_action
        self.expl_noise = args.expl_noise

        self.encoder = model_encoder
        self.decoder = model_decoder
        self.G = model_g
        self.D = model_d
        self.model_classifier = model_classifier

        self.env = Env(args, self.G, self.D, self.model_classifier, self.decoder)

        self.state_dim = args.state_dim
        self.action_dim = args.z_dim
        self.max_action = args.max_action

        self.policy = TD3(args.device, self.state_dim, self.action_dim, self.max_action, self.batch_size_actor,
                          args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)

        self.model_path = os.path.join(args.model_dir, 'RL_train')
        os.makedirs(self.model_path, exist_ok=True)
        if args.load_model:
            self.policy.load(args.model_name, directory=self.model_path)

        self.log_path = os.path.join(args.log_dir, 'RL_train')
        os.makedirs(self.log_path, exist_ok=True)

        # writing log for training
        self.writer = SummaryWriter(self.log_path)

        self.evaluations = [evaluate_policy(self.policy, self.valid_loader, self.env)]

        self.replay_buffer = ReplayBuffer()

    def train(self):
        """
        train RL
        """
        sum_return = 0
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        # get state and corresponding target value
        state_t, episode_target = self.train_loader.next_data()
        state = self.env.agent_input(state_t)

        done = False
        self.env.reset()

        for t in range(int(self.max_timesteps)):

            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < self.start_timesteps:
                # action_t = torch.FloatTensor(self.batch_size, self.z_dim).uniform_(-self.max_action, self.max_action)
                action_t = torch.randn(self.batch_size, self.z_dim)
                action = action_t.detach().cpu().numpy().squeeze(0)
            else:
                action = (
                        self.policy.select_action(state)
                        + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
                ).clip(-self.max_action, self.max_action)
                # action = self.policy.select_action(state)
                action = np.float32(action)
                action_t = torch.tensor(action).to(self.device).unsqueeze(dim=0)

            # Perform action
            next_state, reward, done = self.env(action_t, episode_target)

            # Store data in replay buffer
            self.replay_buffer.add((state, next_state, action, reward, done))

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= self.start_timesteps:
                self.policy.train(self.replay_buffer)

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                # print(
                #     f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state_t, episode_target = self.train_loader.next_data()
                state = self.env.agent_input(state_t)

                done = False
                self.env.reset()

                self.writer.add_scalar("episode_reward", episode_reward, t + 1)
                sum_return += episode_reward
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % self.eval_freq == 0:
                episode_result = "Total T: {} Episode Num: {} Average Reward: {}".format(t + 1, episode_num,
                                                                                         sum_return / episode_num)
                print(episode_result)
                # save some of environment buffer seen so far
                self.replay_buffer.save(len(self.replay_buffer) * 0.01, self.decoder)

                self.evaluations.append(evaluate_policy(self.policy, self.valid_loader, self.env))
                eval_avg_reward = evaluate_policy(self.policy, self.test_loader, self.env, save_fig=True, t=t)
                eval_result = "Evaluation over 6 episodes: {}".format(eval_avg_reward)
                print("---------------------------------------")
                print(eval_result)
                print("---------------------------------------")

                self.writer.add_scalar("average_episode_reward", sum_return / episode_num, t + 1)
                self.writer.add_scalar("test_episode_reward", eval_avg_reward, t + 1)
                self.writer.flush()

                with open(os.path.join(self.log_path, "logs.txt"), "a") as f:
                    f.write(episode_result + '\n' + eval_result + '\n\n')
                    f.close()
                if self.save_models:
                    self.policy.save('RL_{}'.format(t + 1), directory=self.model_path)
