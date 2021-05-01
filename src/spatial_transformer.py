#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An implementation of Deepmind's Spatial Transformer Module.

@author: Ghassan Dabane
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    """A spatial transformer plug and play module for the 28x28 pixels dataset.
    
    Attributes
    ----------
    self.localization: nn.Sequential
        The localization network of the spatial transformer.
        
    self.fc_loc: nn.Sequential
        The regressor for the transformation parameters theta, fully connected
        layers.

    """
    def __init__(self: object) -> None:
        """Class constructor.

        Returns
        -------
        None.

        """
        super(SpatialTransformer, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0],
                                                   dtype=torch.float))

    def stn(self: object, x: torch.Tensor) -> torch.Tensor:
        """The Spatial Transformer module's forward function, pass through
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
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        # resizing theta
        theta = theta.view(-1, 2, 3)
        # grid generator => transformation on parameter 
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x, theta
