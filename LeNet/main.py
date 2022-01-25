import torch
from torch import nn


class Reshape(nn.Module):
    def forward(self, f_x):
        return f_x.view(-1, 1, 28, 28)



