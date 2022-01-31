import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
from IPython import display
from matplotlib import pyplot as plt
import time
import numpy as np


def batch_norm(f_x, f_gamma, f_beta, f_moving_mean, f_moving_var, f_eps, f_momentum):
    if not torch.is_grad_enabled():
        # If it is prediction mode, directly use the mean and variance
        f_x_hat = (f_x - f_moving_mean) / torch.sqrt(f_moving_var + f_eps)
    else:
        assert len(f_x.shape) in (2, 4)
        if len(f_x.shape) == 2:
            # When using a fully-connected layer(linear layer), calculate the mean and variance on the feature dimension
            mean = f_x.mean(dim=0)
            var = ((f_x - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the mean and variance on the channel dimension
            # Maintain the shape of 'f_x', so that the broadcasting operation can be carried out later
            mean = f_x.mean(dim=(0, 2, 3), keepdim=True)
            var = ((f_x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the standardization
        f_x_hat = (f_x - mean) / torch.sqrt(var + f_eps)
        # Update the mean and variance using moving average and moving variance
        f_moving_mean = f_momentum * f_moving_mean + (1.0 - f_momentum) * mean
        f_moving_var = f_momentum * f_moving_var + (1.0 - f_momentum) * var
    f_y = f_gamma * f_x_hat + f_beta
    return f_y, f_moving_mean.data, f_moving_var.data


class BatchNorm(nn.Module):
    # num_features means the number of outputs for a fully-connected layer or the number of output channels for a
    # convolutional layer.
    # num_dims is 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, f_num_features, f_num_dims):
        super().__init__()
        if f_num_dims == 2:
            shape = (1, f_num_features)
        else:
            shape = (1, f_num_features, 1, 1)
        # gamma and beta are model parameters
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # moving_mean and moving_var are not  model parameters
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, f_x):
        if self.moving_mean.device != f_x.device:
            self.moving_mean = self.moving_mean.to(f_x.device)
            self.moving_var = self.moving_var.to(f_x.device)
        f_y, self.moving_mean, self.moving_var = batch_norm(
            f_x, self.gamma, self.beta, self.moving_mean, self.moving_var, f_eps=1e-5, f_momentum=0.9
        )
        return f_y


net = nn.Sequential(
    nn.Conv2d(1, 6, (5, 5)), BatchNorm(6, 4), nn.Sigmoid(),
    nn.AvgPool2d((2, 2), 2),
    nn.Conv2d(6, 16, (5, 5)), BatchNorm(16, 4), nn.Sigmoid(),
    nn.AvgPool2d((2, 2), 2),
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120), BatchNorm(120, 2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, 2), nn.Sigmoid(),
    nn.Linear(84, 10)
)

tensor = torch.tensor([[1.0, 2], [3, 4]])
print("tensor.mean(1) = ", tensor.mean(1))
print("tensor.mean(0) = ", tensor.mean(0))
# tensor.mean(1) =  tensor([1.5000, 3.5000])
# tensor.mean(0) =  tensor([2., 3.])
# The dimension after mean function will be reduced by 1
tensor = torch.arange(16.0)
tensor = tensor.reshape(2, 2, 2, 2)
print("tensor = ", tensor)
print("tensor.mean((0), keepdim=True) = ", tensor.mean((0), True))
print("tensor.mean((0, 3), keepdim=True) = ", tensor.mean((0, 3), True))
print("tensor.mean((0, 2, 3), keepdim=True) = ", tensor.mean((0, 2, 3), True))
