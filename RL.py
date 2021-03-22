import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    """
    actor model
    """
    def __init__(self, state_dim, action_dim, max_action):
        """
        initialize actor model

        Parameters
        ----------
        state_dim : int
            dimension of state
        action_dim : int
            dimension of action
        max_action : int, float
            maximum of absolute value possible for action
        """
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        """
        forward pass of actor model

        Parameters
        ----------
        state : torch.Tensor
            state tensor

        Returns
        -------
        torch.Tensor
            action tensor in range (-max_action, max_action)
        """
        a = F.relu(self.l1(state))  # B x state_dim ---> B x 256
        a = F.relu(self.l2(a))  # B x 256 --->  B x 256
        a = self.max_action * torch.tanh(self.l3(a))  # B x 256 --->  B x action_dim
        return a  # (values in range [-max_action, max_action])


class Critic(nn.Module):
    """
    critic model
    """
    def __init__(self, state_dim, action_dim):
        """
        initialize critic model

        Parameters
        ----------
        state_dim :  int
            dimension of state
        action_dim :  int
            dimension of action
        """
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        """
        forward pass of critic model

        Parameters
        ----------
        state : torch.Tensor
            state tensor
        action : torch.Tensor
            action tensor

        Returns
        -------
        tuple
            tensor of Q1, tensor of Q2 (estimates of value function Q)

        """
        sa = torch.cat([state, action], 1)  # (B x state_dim, B x action_dim) ---> B x (state_dim + action_dim)

        q1 = F.relu(self.l1(sa))  # B x (state_dim + action_dim) ---> B x 256
        q1 = F.relu(self.l2(q1))  # B x 256 ---> B x 256
        q1 = self.l3(q1)  # B x 256 ---> B x 1

        q2 = F.relu(self.l4(sa))  # B x (state_dim + action_dim) ---> B x 256
        q2 = F.relu(self.l5(q2))  # B x 256 ---> B x 256
        q2 = self.l6(q2)  # B x 256 ---> B x 1
        return q1, q2  # B x 1, B x 1

    def Q1(self, state, action):
        """
        get estimate of value function

        Parameters
        ----------
        state : torch.Tensor
            state tensor
        action : torch.Tensor
            action tensor

        Returns
        -------
        torch.Tensor
            tensor of Q1(estimate of value function Q)
        """
        sa = torch.cat([state, action], 1)  # (B x state_dim, B x action_dim) ---> B x (state_dim + action_dim)

        q1 = F.relu(self.l1(sa))  # B x (state_dim + action_dim) ---> B x 256
        q1 = F.relu(self.l2(q1))  # B x 256 ---> B x 256
        q1 = self.l3(q1)  # B x 256 ---> B x 1
        return q1


# class TD3(object):
#     def __init__(self, state_dim, action_dim, max_action, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
#                  noise_clip=0.5, policy_freq=2):
#
#         self.batch_size = batch_size
#         self.discount = discount
#         self.tau = tau
#         self.policy_noise = policy_noise
#         self.noise_clip = noise_clip
#         self.policy_freq = policy_freq
#
#         self.actor = Actor(state_dim, action_dim, max_action).to(device)
#         self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
#         self.actor_target.load_state_dict(self.actor.state_dict())
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
#
#         self.critic = Critic(state_dim, action_dim).to(device)
#         self.critic_target = Critic(state_dim, action_dim).to(device)
#         self.critic_target.load_state_dict(self.critic.state_dict())
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
#
#         self.max_action = max_action
#
#     def select_action(self, state):
#         # Taha Changed:
#         #### why did he do this ??????????
#         # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
#         state = torch.FloatTensor(state).to(device)
#         # Taha Changed:
#         # return self.actor(state).cpu().data.numpy().flatten()
#         return self.actor(state).cpu().data.numpy()
#
#     def train(self, replay_buffer, iterations):
#
#         for it in range(iterations):
#
#             # Sample replay buffer
#             x, y, u, r, d = replay_buffer.sample(self.batch_size)
#             state = torch.FloatTensor(x).to(device)
#             action = torch.FloatTensor(u).to(device)
#             next_state = torch.FloatTensor(y).to(device)
#             done = torch.FloatTensor(1 - d).to(device)
#             reward = torch.FloatTensor(r).to(device)
#
#             # Select action according to policy and add clipped noise
#             noise = torch.FloatTensor(u).data.normal_(0, self.policy_noise).to(device)
#             noise = noise.clamp(-self.noise_clip, self.noise_clip)
#             next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
#             next_action = next_action.clamp(-self.max_action, self.max_action)
#
#             # Compute the target Q value
#             target_Q1, target_Q2 = self.critic_target(next_state, next_action)
#             target_Q = torch.min(target_Q1, target_Q2)
#             target_Q = reward + (done * self.discount * target_Q).detach()
#
#             # Get current Q estimates
#             current_Q1, current_Q2 = self.critic(state, action)
#
#             # Compute critic loss
#             critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
#
#             # Optimize the critic
#             self.critic_optimizer.zero_grad()
#             critic_loss.backward()
#             self.critic_optimizer.step()
#
#             # Delayed policy updates
#             if it % self.policy_freq == 0:
#
#                 # Compute actor loss
#                 actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
#
#                 # Optimize the actor
#                 self.actor_optimizer.zero_grad()
#                 actor_loss.backward()
#                 self.actor_optimizer.step()
#
#                 # Update the frozen target models
#                 for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
#                     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
#
#                 for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
#                     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
#
#     def save(self, filename, directory):
#         torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
#         torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
#
#     def load(self, filename, directory):
#         self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
#         self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


class TD3(object):
    """
    Twin Delayed Deep Deterministic Policy Gradients model
    """
    def __init__(self, device, state_dim, action_dim, max_action, lr, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_freq=2):
        """
        initialize TD3

        Parameters
        ----------
        device : torch.device
            device to run model
        state_dim : int
            dimension of state
        action_dim : int
            dimension of action
        max_action : int, float
            maximum value possible for action
        batch_size : int
            batch size of steps to train model on
        discount : float
            discount in RL reward calculation
        tau : float
            coefficient to update target policy
        policy_noise : float
            noise in policy
        noise_clip : int, float
            maximum value of noise in policy
        policy_freq : int
            frequency to update actor (update once every policy_freq times)
        """
        self.device = device
        self.batch_size = batch_size
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.total_it = 0

    def select_action(self, state):
        """
        select action from actor model

        Parameters
        ----------
        state : torch.Tensor
            tensor of state

        Returns
        -------
        numpy.ndarray
            numpy array of action chosen by actor
        """
        state = torch.FloatTensor(state).to(self.device)
        return self.actor(state).cpu().data.numpy()

    def train(self, replay_buffer):
        """
        train policy based of TD3 paper (actor-critic method with double Q-learning)

        Parameters
        ----------
        replay_buffer :
            buffer storing all seen environment samples
        """
        self.total_it += 1

        # Sample replay buffer
        s, n_s, a, r, not_d = replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(s).to(self.device)
        next_state = torch.FloatTensor(n_s).to(self.device)
        action = torch.FloatTensor(a).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)
        done = torch.FloatTensor(1 - not_d).to(self.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename, directory):
        """
        save models and optimizers

        Parameters
        ----------
        filename : str
            file name
        directory : str
            directory names

        """
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))

        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))

    def load(self, filename, directory):
        """
        load models and optimizers

        Parameters
        ----------
        filename : str
            file name
        directory : str
            directory names

        """
        critic_path = '%s/%s_critic.pth' % (directory, filename)
        critic_optimizer_path = '%s/%s_critic_optimizer.pth' % (directory, filename)
        actor_path = '%s/%s_actor.pth' % (directory, filename)
        actor_optimizer_path = '%s/%s_actor_optimizer.pth' % (directory, filename)
        if not os.path.exists(critic_path):
            raise FileNotFoundError('critic model not found')
        if not os.path.exists(critic_optimizer_path):
            raise FileNotFoundError('critic optimizer model not found')
        if not os.path.exists(actor_path):
            raise FileNotFoundError('actor model not found')
        if not os.path.exists(actor_optimizer_path):
            raise FileNotFoundError('actor optimizer model not found')

        self.critic.load_state_dict(torch.load(critic_path))
        self.critic_optimizer.load_state_dict(torch.load(critic_optimizer_path))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(actor_path))
        self.actor_optimizer.load_state_dict(torch.load(actor_optimizer_path))
        self.actor_target = copy.deepcopy(self.actor)


if __name__ == "__main__":
    actor = Actor(state_dim=32, action_dim=2, max_action=10)
    critic = Critic(state_dim=32, action_dim=2)
    s = torch.FloatTensor(np.random.rand(5, 32))
    a = actor(s)
    n_s = critic(s, a)
    print('finished')
