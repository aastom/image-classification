# TODO: Import your dependencies.
# For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import boto3

# TODO: Import dependencies for Debugging andd Profiling
from torch.autograd import profiler


def test(model, test_loader):
    """
    Function to test the accuracy and loss of a pre-trained model on a given dataset.
    Parameters:
    - model (nn.Module): The pre-trained model to be tested
    - test_loader (torch.utils.data.DataLoader): The data loader for the test dataset

    Returns:
    - test_loss (float): average loss on the test dataset
    - test_acc (float): accuracy on the test dataset
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, pred = torch.max(output.data, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
    test_loss /= total
    test_acc = 100.0 * correct / total
    print("Test Loss: {:.4f}".format(test_loss))
    print("Test Accuracy: {:.2f}%".format(test_acc))
    return test_loss, test_acc


def train(model, train_loader, criterion, optimizer):
    """
    Function to train a model on a given dataset.

    Parameters:
    - model (nn.Module): The model to be trained
    - train_loader (torch.utils.data.DataLoader): The data loader for the training dataset
    - criterion (nn.Module): The loss function to be used
    - optimizer (torch.optim.Optimizer): The optimizer to be used

    Returns:
    - model (nn.Module): The trained model
    """
    model.train()
    for epoch in range(args.num_epochs):
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if i % args.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        i * len(data),
                        len(train_loader.dataset),
                        100.0 * i / len(train_loader),
                        loss.item(),
                    )
                )
    return model
