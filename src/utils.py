#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 19:25:20 2021

@author: ghassan
"""
import torch
import matplotlib.pyplot as plt

def view_dataset(loader: torch.utils.data.DataLoader, n: int=10) -> plt.Figure:
    """This function views a couple of examples from a given DataLoader object.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        The datalaoder.
    n : int, optional
        The number of examples to show. The default is 10.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The resulting figure, can be saved with savefig() method.
    """
    data, label = next(iter(loader))
    fig, axs = plt.subplots(1, n, figsize = (21, 5))
    for i_ax, ax in enumerate(axs):
        ax.imshow(data[i_ax, 0, :, :], cmap=plt.gray())
        ax.set_title("Label = %d"%(label[i_ax].item()))
        ax.set_xticks([])
        ax.set_yticks([])
    return fig