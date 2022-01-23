import torch
from torch import nn


def comp_conv2d(f_conv2d, f_x):
    # single channel
    f_x = f_x.reshape((1, 1) + f_x.shape)
    f_y = f_conv2d(f_x)
    return f_y.reshape(f_y.shape[2:])


conv2d = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=1)
# For pytorch padding = 1 means that the feature will be padded with 1 every side
x = torch.rand(size=(8, 8))
y = comp_conv2d(conv2d, x)
print("y.shape = ", y.shape)
# y.shape = torch.Size([8, 8])

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
y = comp_conv2d(conv2d, x)
print("y.shape = ", y.shape)
# y.shape = torch.Size([8, 8])

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
x = torch.rand(size=(7, 9))
y = comp_conv2d(conv2d, x)
print("y.shape = ", y.shape)
# y.shape = torch.Size([4, 4])

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
x = torch.rand(size=(8, 8))
y = comp_conv2d(conv2d, x)
print("y.shape = ", y.shape)
# y.shape = torch.Size([2, 2])
