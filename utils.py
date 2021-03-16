import os
import torch
from torch.autograd import Variable
import numpy as np


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


# class ReplayBuffer(object):
#     def __init__(self, state_dim, action_dim, max_size=int(1e6)):
#         self.max_size = max_size
#         self.ptr = 0
#         self.size = 0
#
#         self.state = np.zeros((max_size, state_dim))
#         self.action = np.zeros((max_size, action_dim))
#         self.next_state = np.zeros((max_size, state_dim))
#         self.reward = np.zeros((max_size, 1))
#         self.not_done = np.zeros((max_size, 1))
#
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     def add(self, state, action, next_state, reward, done):
#         self.state[self.ptr] = state
#         self.action[self.ptr] = action
#         self.next_state[self.ptr] = next_state
#         self.reward[self.ptr] = reward
#         self.not_done[self.ptr] = 1. - done
#
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
#
#     def sample(self, batch_size):
#         ind = np.random.randint(0, self.size, size=batch_size)
#
#         return (
#             torch.FloatTensor(self.state[ind]).to(self.device),
#             torch.FloatTensor(self.action[ind]).to(self.device),
#             torch.FloatTensor(self.next_state[ind]).to(self.device),
#             torch.FloatTensor(self.reward[ind]).to(self.device),
#             torch.FloatTensor(self.not_done[ind]).to(self.device)
#         )


# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self):
        self.storage = []

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size=100):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


def tensor2var(x, grad=False):
    """
    tensor to variable

    Parameters
    ----------
    x : 
        variable
    grad : 
        set if trainable or not
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)


def var2tensor(x):
    """
    get variable data

    Parameters
    ----------
    x : 
        variable
    """
    return x.data.cpu()


def var2numpy(x):
    """
    get numpy result from variable

    Parameters
    ----------
    x : 
        variable
    """
    return x.data.cpu().numpy()
