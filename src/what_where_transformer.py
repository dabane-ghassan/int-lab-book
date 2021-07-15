#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script for the new what/where based spatial transformer architecture.

@author: Ghassan Dabane
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Saccader(nn.Module):
    
    def __init__(self):
        super(Saccader, self).__init__()

    def forward(
            self,
            x: torch.Tensor,
            x_polo: torch.Tensor
    ) -> Tuple[torch.Tensor]:

        pass

class Pooler(nn.Module):

    def __init__(self):
        super(Pooler, self).__init__()

    def forward(
            self,
            x: torch.Tensor,
            x_polo: torch.Tensor
    ) -> Tuple[torch.Tensor]:

        pass

class WhatNet(nn.Module):

    def __init__(self):
        super(WhatNet, self).__init__()

    def forward(
            self,
            x: torch.Tensor,
            x_polo: torch.Tensor
    ) -> Tuple[torch.Tensor]:

        pass

class WhatWhereTransformer(nn.Module):

    def __init__(self):
        super(WhatWhereTransformer, self).__init__()

    def forward(
            self,
            x: torch.Tensor,
            x_polo: torch.Tensor
    ) -> Tuple[torch.Tensor]:

        pass
