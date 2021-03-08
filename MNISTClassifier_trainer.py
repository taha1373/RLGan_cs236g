# libraries 
import torch
import torch.nn as nn
import os
from torch.nn.functional import one_hot
import seaborn as sns


from MNISTClassifier import Classifier, show_tensor_images


# libraries for visualizing the image
from tqdm import tqdm
import time

import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, args,dataLoader):

        self.lr = args.lr
        self.beta_1 = args.beta_1
        self.beta_2 = args.beta_2
        self.numLabels = args.numLabels
        self.device = args.device
        self.model_save_path = args.model_save_path
        self.train_checkPoints = args.train_checkPoints
        self.numEpochs = args.numEpochs
        self.dataLoader = dataLoader

        self.train_losses = []
        self.train_counter = []


        self.buildModel()

    def train(self):
        cur_step = 0
        display_step = 20
        classifier_losses = []
        for epoch in range(self.numEpochs):
            for real, labels in tqdm(self.dataLoader):
                real = real.to(self.device)
                labels = one_hot(labels.to(self.device)).float()

                self.class_opt.zero_grad()
                class_pred = self.classifier(real)
                class_loss = self.criterion(class_pred, labels)
                class_loss.backward() # Calculate the gradients
                self.class_opt.step() # Update the weights
                classifier_losses += [class_loss.item()] # Keep track of the average classifier loss

                ## Visualization code ##
                if cur_step % display_step == 0 and cur_step > 0:
                    class_mean = sum(classifier_losses[-display_step:]) / display_step
                    print(f"Step {cur_step}: Classifier loss: {class_mean}")
                    # step_bins = 20
                    # x_axis = sorted([i * step_bins for i in range(len(classifier_losses) // step_bins)] * step_bins)
                    # sns.lineplot(x_axis, classifier_losses[:len(x_axis)], label="Classifier Loss")
    #                 plt.legend()
                    # plt.show()
                    # torch.save({"classifier": self.classifier.state_dict()}, filename)
                cur_step += 1

        # torch.save(self.ae.state_dict(),os.path.join(self.model_save_path, 'MNISTclassifier.pth'))

    def test(self,test_loader):
        self.classifier.eval()
        # test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.classifier(data)
                # test_loss += nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        # test_loss /= len(test_loader.dataset)
        # test_losses.append(test_loss)
        print(100. * correct / len(test_loader.dataset))
        # print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        # test_loss, correct, len(test_loader.dataset),
        # 100. * correct / len(test_loader.dataset)))


    def buildModel(self):
        self.classifier = Classifier().to(self.device)
        self.class_opt = torch.optim.Adam(self.classifier.parameters(), lr=self.lr, betas=(self.beta_1,self.beta_2))
        self.criterion = nn.BCEWithLogitsLoss()

