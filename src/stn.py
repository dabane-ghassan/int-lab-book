#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A spatial transformer network, a LeNet network (taken from the generic
what pathway as the exact same structure for comparaison of performance) + 
A spatial transformer module.

@author: Ghassan Dabane
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_transformer import SpatialTransformer

class STN(nn.Module):

    def __init__(self: object) -> None:
        super(STN, self).__init__()
        
        ## The what pathway
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

        ## The spatial transformer
        self.transformer_module = SpatialTransformer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # transform the input
        x, _ = self.transformer_module.stn(x)

        # Perform the usual forward pass
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class STN_128x128(nn.Module):

    def __init__(self):
        super(STN_128x128, self).__init__()

        ##  The what pathway
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 100, kernel_size=5)
        self.fc1 = nn.Linear(100*29*29, 128)
        self.fc2 = nn.Linear(128, 10)
        
        ## The spatial transformer
        self.localization = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self: object, x: torch.Tensor) -> torch.Tensor:
        """
        The Spatial Transformer module's forward function, pass through
        the localization network, predict transformation parameters theta,
        generate a grid and apply the transformation parameters theta on it
        and finally sample the grid using an interpolation.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output (transformed) tensor.
        """

        xs = self.localization(x)
        xs = xs.view(-1, 128 * 4 * 4)
        
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        #grid_size = torch.Size([x.size()[0], x.size()[1], 28, 28])
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x, theta

    def forward(self, x):
        # transform the input
        x, _ = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 100 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x