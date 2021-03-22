import os
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# libraries for visualizing the image
from torchvision.utils import make_grid
from math import ceil, sqrt
import pickle
import os, fnmatch


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


class toLatentTsfm(object):
    """
    class for transforming the images to the latent space
    """
    def __init__(self, ae_model, device):
        """
        initialize transform to encode mnist dataset

        Parameters
        ----------
        ae_model : torch.nn.Module
            auto-encoder model
        device : torch.device
            device to load model to (cpu, gpu)
        """
        # the vae model for transofrmation
        self.ae = ae_model
        # set the vae to the eval mode
        self.ae.eval()
        self.device = device

    def __call__(self, sample):
        image = sample
        image = image.reshape((1,) + tuple(image.shape)).to(self.device)
        reconImage, z_sample = self.ae(image)
        return z_sample


def get_n_params(model):
    """
    get parameter size of model

    Parameters
    ----------
    model : torch.nn.Module

    Returns
    -------
    int
        parameter size of model

    """
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


def display_env(state_img, next_state_img, action, reward, save_path, target=None):
    np.set_printoptions(precision=3)
    title = 'action: {}, reward: {}'.format(action.squeeze(), reward.squeeze())
    if target is not None:
        title += ', target: {}'.format(target.squeeze())
    show_tensor_images(torch.cat((state_img, next_state_img), dim=0), 2)
    plt.title(title)
    plt.savefig(save_path)


class ReplayBuffer(object):
    def __init__(self):
        """
        buffer to store environment samples seen so far
        """
        self.storage = []
        self._saved = []
        self._sample_ind = None
        self._ind_to_save = 0

    def add(self, data):
        """
        add sample of environment to buffer

        Parameters
        ----------
        data : tuple
            (state, next_state, action, reward, is_episode_finished)

        """
        self.storage.append(data)
        self._saved.append(False)

    def sample(self, batch_size=100):
        """
        get sample from buffer containing all seen environment informaiton

        Parameters
        ----------
        batch_size : int
            batch number of sample

        Returns
        -------
        tuple
            batch of (state, next_state, action, reward, is_episode_finished)

        """
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        self._sample_ind = ind
        return self[ind]

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, items):
        if hasattr(items, '__iter__'):
            items_iter = items
        else:
            items_iter = [items]

        x, y, u, r, d = [], [], [], [], []
        for i in items_iter:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

    def save(self, file_dir='./replay'):
        if not os.path.exists(os.path.splitext(file_dir)[0]):
            os.makedirs(os.path.splitext(file_dir)[0], exist_ok=True)
        file_path = os.path.join(os.path.splitext(file_dir)[0], 'replay_{:07}.pkl'.format(len(self)))
        with open(file_path, 'wb') as f:
            pickle.dump(self.storage[self._ind_to_save:], f, -1)
        self._ind_to_save = len(self.storage)

    def load(self, file_dir='./replay'):
        file_paths = sorted(find('*.pkl', file_dir))
        for p in file_paths:
            with open(p, 'rb') as f:
                self.storage += pickle.load(f)

    def save_samples(self, sample_num, model_decoder, shuffle=False, save_path='./replay', max_num=10):
        """
        save some samples from buffer
        if shuffle is false indexes 0:sample_num are saved

        Parameters
        ----------
        sample_num : int
            number of samples to save
        model_decoder : torch.nn.Module
            decoder model
        shuffle : bool
            get random sample or first elements in buffer
        save_path : str
            saving folder
        max_num : int
            maximum number of samples to save
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        buffer_length = len(self) - 1

        num = min(int(sample_num), max_num)
        for i in range(buffer_length, buffer_length - num, -1):
            if shuffle:
                x, y, u, r, d = self.sample(1)
                ind = self._sample_ind[0]
            else:
                x, y, u, r, d = self[i]
                ind = i
            if self._saved[ind]:
                return
            device = next(model_decoder.parameters()).device
            x_tensor = torch.tensor(x).to(device)
            y_tensor = torch.tensor(y).to(device)
            with torch.no_grad():
                x_img = model_decoder(x_tensor)
                y_img = model_decoder(y_tensor)
            display_env(x_img, y_img, u, r, os.path.join(save_path, "img_{}".format(ind + 1)))
            self._saved[ind] = True


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


def show_tensor_images(image_tensor, num_images=25):
    """
    Function for visualizing images

    Parameters
    ----------
    image_tensor : torch.Tensor
        batch of images to visualize
    num_images : int
        number of images
    """
    row_num = min(ceil(sqrt(num_images)), num_images)
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=row_num)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())