# Lets import some dependencies.
# For instance, below are some dependencies we might need if we are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import transforms, datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader

import argparse
import sys
import logging
import json
import os


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion):
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
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    total = len(test_loader.dataset)
    test_loss /= total
    test_acc = 100.0 * correct / total
    logger.info(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}")
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
    for epoch in range(args.epochs):
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            print(output, target)
            try:
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    logger.info(
                        f"Train Epoch: {epoch} [{i * len(data)}/{len(train_loader.dataset)} ({100.0 * i / len(train_loader):.0f}%)] Loss: {loss.item():.6f}".format(
                            epoch,
                            i * len(data),
                            len(train_loader.dataset),
                            100.0 * i / len(train_loader),
                            loss.item(),
                        )
                    )
            except Exception as error:
                print(error)
    return model


def net():
    """
    Initializes the model and loads a pretrained model
    """
    model = models.resnet18()
    # Freeze all layers
    for param in model.parameters():
        param.requiresGrad = False
    # Replace last fully connected layer with a new one
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    return model


def create_data_loaders(data_dir, batch_size):
    # Define the transformations for the train and test sets
    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + "/train", transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + "/test", transform=test_transforms)

    # define the dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader


def main(args):
    """
    Main function that trains and tests the model
    """

    # Create data loaders
    train_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)

    # Initialize model
    model = net()

    """
    Create loss and optimizer
    """
    # loss_criterion = nn.CrossEntropyLoss()
    loss_criterion = F.nll_loss
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    """
    Call the train function to start training the model
    """
    model = train(model, train_loader, loss_criterion, optimizer)

    """
    Test the model to see its accuracy
    """
    test(model, test_loader, loss_criterion)

    """
    Save the trained model
    """
    logger.info("Saving the model.")
    torch.save(model, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    Specify all the hyperparameters needed to for training the model.
    """
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="s3://sagemaker-us-east-1-553938774894/",
        metavar="M",
        help="The path for saving the model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dogImages/",
        metavar="M",
        help="The path to the data",
    )

    args = parser.parse_args()

    main(args)
