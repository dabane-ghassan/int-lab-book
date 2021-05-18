#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 08:23:14 2021

@author: ghassan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionTransNet(nn.Module):
    def __init__(self):
        super(AttentionTransNet, self).__init__()

        ## The what pathway
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

        ## The spatial transformer
        self.localization = nn.Sequential(nn.Conv2d(1, 16, kernel_size=7),
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
                                          nn.ReLU(True))

        self.fc_loc = nn.Sequential(nn.Linear(128 * 4 * 4, 32), nn.ReLU(True),
                                    nn.Linear(32, 3))

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([0.2, 0, 0], dtype=torch.float)) #set scaling to 0.2 (~28/128)


    def stn(self: object, x: torch.Tensor) -> torch.Tensor:

        xs = self.localization(x)
        xs = xs.view(-1, 128 * 4 * 4)

        theta = self.fc_loc(xs)

        translation = theta[:, 1:].unsqueeze(2)
        scale = theta[:, 0].unsqueeze(1)
        scale_mat = torch.cat((scale, scale), 1)
        theta = torch.cat((torch.diag_embed(scale_mat), translation), 2)

        grid_size = torch.Size([x.size()[0], x.size()[1], 28, 28])
        grid = F.affine_grid(theta, grid_size)
        x = F.grid_sample(x, grid)

        return x, theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # transform the input
        x, _ = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x