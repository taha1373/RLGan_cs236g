# libraries 
import torch
import torch.nn as nn

# libraries for visualizing the image
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

import torch.nn.functional as F

class Classifier(nn.Module):
    """Classifier class"""
    def __init__(self):
        """
        initialize Classifier model

        simple classifier for one channel images (e.g. mnist dataset)
        """
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Function for completing a forward pass of the classifier
        
        Parameters
        ----------
        x : torch.Tensor 
            an image tensor with dimensions (batch_size, 1, im_height, im_width)

        Returns
        -------
        torch.Tensor
            class output
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)



def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
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
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())



