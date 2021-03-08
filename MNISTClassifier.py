# libraries 
import torch
import torch.nn as nn

# libraries for visualizing the image
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

import torch.nn.functional as F

# class Classifier(nn.Module):
#     '''
#     Classifier Class
#     Values:
#         im_chan: the number of channels in the images, fitted for the dataset used, a scalar
#               (MNIST is not rgb, so 1 is our default)
#         n_classes: the total number of classes in the dataset, an integer scalar
#         hidden_dim: the inner dimension, a scalar
#     '''
#     def __init__(self, im_chan=1, n_classes=10, hidden_dim=32):
#         super(Classifier, self).__init__()
#         self.classifier = nn.Sequential(
#             self.make_classifier_block(im_chan, hidden_dim),
#             self.make_classifier_block(hidden_dim, hidden_dim * 2),
#             self.make_classifier_block(hidden_dim * 2, hidden_dim * 4, stride=3),
#             self.make_classifier_block(hidden_dim * 4, n_classes, final_layer=True),
#         )

#     def make_classifier_block(self, input_channels, output_channels, kernel_size=2, stride=2, final_layer=False):
#         '''
#         Function to return a sequence of operations corresponding to a classifier block; 
#         a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
#         Parameters:
#             input_channels: how many channels the input feature representation has
#             output_channels: how many channels the output feature representation should have
#             kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
#             stride: the stride of the convolution
#             final_layer: a boolean, true if it is the final layer and false otherwise 
#                       (affects activation and batchnorm)
#         '''
#         if final_layer:
#             return nn.Sequential(
#                 nn.Conv2d(input_channels, output_channels, kernel_size, stride),
#             )
#         else:
#             return nn.Sequential(
#                 nn.Conv2d(input_channels, output_channels, kernel_size, stride),
#                 nn.BatchNorm2d(output_channels),
#                 nn.LeakyReLU(0.2, inplace=True),
#             )

#     def forward(self, image):
#         '''
#         Function for completing a forward pass of the classifier: Given an image tensor, 
#         returns an n_classes-dimension tensor representing fake/real.
#         Parameters:
#             image: a flattened image tensor with im_chan channels
#         '''
#         class_pred = self.classifier(image)
#         return class_pred.view(len(class_pred), -1)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)



def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())



