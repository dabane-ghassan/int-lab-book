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
        return x #F.log_softmax(x, dim=1)
