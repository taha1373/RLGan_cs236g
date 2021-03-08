import os
import torch
from torch.autograd import Variable

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