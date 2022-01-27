import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
from IPython import display
from matplotlib import pyplot as plt
import time
import numpy as np


# Since the data set is F-MNIST, the input channel is 1 and the final out put is 10
net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=(11, 11), stride=(4, 4), padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=2),
    nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=2),
    nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)


# # check the net definition
# x = torch.rand((1, 1, 224, 224), dtype=torch.float32)
# for layer in net:
#     x = layer(x)
#     print(layer.__class__.__name__, "output shape: \t", x.shape)
# # Conv2d output shape:     torch.Size([1, 96, 54, 54])
# # ReLU output shape:       torch.Size([1, 96, 54, 54])
# # MaxPool2d output shape:          torch.Size([1, 96, 26, 26])
# # Conv2d output shape:     torch.Size([1, 256, 26, 26])
# # ReLU output shape:       torch.Size([1, 256, 26, 26])
# # MaxPool2d output shape:          torch.Size([1, 256, 12, 12])
# # Conv2d output shape:     torch.Size([1, 384, 12, 12])
# # ReLU output shape:       torch.Size([1, 384, 12, 12])
# # Conv2d output shape:     torch.Size([1, 384, 12, 12])
# # ReLU output shape:       torch.Size([1, 384, 12, 12])
# # Conv2d output shape:     torch.Size([1, 256, 12, 12])
# # ReLU output shape:       torch.Size([1, 256, 12, 12])
# # MaxPool2d output shape:          torch.Size([1, 256, 5, 5])
# # Flatten output shape:    torch.Size([1, 6400])
# # Linear output shape:     torch.Size([1, 4096])
# # ReLU output shape:       torch.Size([1, 4096])
# # Dropout output shape:    torch.Size([1, 4096])
# # Linear output shape:     torch.Size([1, 4096])
# # ReLU output shape:       torch.Size([1, 4096])
# # Dropout output shape:    torch.Size([1, 4096])
# # Linear output shape:     torch.Size([1, 10])
