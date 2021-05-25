#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:31:12 2021

@author: ghassan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Polo_AttentionTransNet(nn.Module):

    def __init__(self):
        super(Polo_AttentionTransNet, self).__init__()

        ##  The what pathway
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

        self.fc_loc = nn.Sequential(nn.Linear(8 * 6 * 16, 32),
                                    nn.ReLU(True),
                                    nn.Linear(32, 2))

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.zero_()

        self.downscale = nn.Parameter(torch.tensor([[0.2, 0], [0, 0.2]],
                                                   dtype=torch.float),
                                      requires_grad=False)

    def stn(self: object, x: torch.Tensor, x_polo: torch.Tensor) -> Tuple[torch.Tensor]:
    
        xs = x_polo.view(-1, 8 * 6 * 16)
        theta = self.fc_loc(xs)

        theta = torch.cat((self.downscale.unsqueeze(0).repeat(
            theta.size(0), 1, 1), theta.unsqueeze(2)),
                          dim=2)
        
        #theta = theta.view(-1, 2, 3)
        
        grid_size = torch.Size([x.size()[0], x.size()[1], 28, 28])
        grid = F.affine_grid(theta, grid_size)
        x = F.grid_sample(x, grid)

        return x, theta

    def forward(self, x, x_polo):
        # transform the input
        x, _ = self.stn(x, x_polo)

        # Perform the usual forward pass
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
