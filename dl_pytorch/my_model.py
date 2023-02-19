import torch
import torch.nn as nn
import torch.nn.functional as F

#######################################################################
# TODO: Design your own neural network
# You can define utility functions/classes here
#######################################################################
pass
#######################################################################
# End of your code
#######################################################################


class MyNeuralNetwork(nn.Module):
    def __init__(self, do_batchnorm=False, p_dropout=0.0):
        super().__init__()
        self.do_batchnorm = do_batchnorm
        self.p_dropout = p_dropout

        #######################################################################
        # TODO: Design your own neural network
        #######################################################################
        self.conv1 = torch.nn.Conv2d(3, 20, 3, 1, 1)
        if do_batchnorm:
            self.bn1 = torch.nn.BatchNorm2d(20)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.AvgPool2d(2,2)

        self.conv2 = torch.nn.Conv2d(20, 80, 3, 1, 1)
        if do_batchnorm:
            self.bn2 = torch.nn.BatchNorm2d(80)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2,2)

        self.conv3 = torch.nn.Conv2d(80, 150, 4, 2, 1)
        if do_batchnorm:
            self.bn3 = torch.nn.BatchNorm2d(150)
        self.relu3 = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(2400, 200)
        self.relu4 = torch.nn.ReLU()
        if p_dropout > 0.0:
            self.drop = torch.nn.Dropout(p_dropout)
        self.fc2 = torch.nn.Linear(200, 100)
        #######################################################################
        # End of your code
        #######################################################################

    def forward(self, x):
        #######################################################################
        # TODO: Design your own neural network
        #######################################################################
        x = self.conv1(x)  # [bsz, 16, 32, 32]
        if self.do_batchnorm:
            x = self.bn1(x)
        x = self.pool1(self.relu1(x))  # [bsz, 16, 16, 16]

        x = self.conv2(x)  # [bsz, 32, 16, 16]
        if self.do_batchnorm:
            x = self.bn2(x)
        x = self.pool2(self.relu2(x))  # [bsz, 32, 8, 8]

        x = self.conv3(x)  # [bsz, 64, 4, 4]
        if self.do_batchnorm:
            x = self.bn3(x)
        x = self.relu3(x)

        x = torch.flatten(x, 1)  # [bsz, 1024]
        x = self.relu4(self.fc1(x))  # [bsz, 256]
        if self.p_dropout > 0.0:
            x = self.drop(x)
        x = self.fc2(x)  # [bsz, 100]
        return x

        #######################################################################
        # End of your code
        #######################################################################
