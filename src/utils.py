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
    
def train(epoch, loader):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device, dtype=torch.float), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, args.epochs, batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(loader):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in loader:
            data, target = data.to(device,
                                   dtype=torch.float), target.to(device)
            output = model(data)

            # sum up batch loss
            #test_loss += F.nll_loss(output, target, size_average=False).item()
            test_loss += loss_func(output, target).item()
            # get the index of the max log-probability
            #pred = output.max(1, keepdim=True)[1]
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
              format(test_loss, correct, len(test_loader.dataset),
                     100. * correct / len(test_loader.dataset)))
        return correct / len(test_loader.dataset)
