import torch
from torch import nn


class Reshape(nn.Module):
    def forward(self, f_x):
        return f_x.view(-1, 1, 28, 28)
        # reshape tries to return a view if possible, otherwise copies to data to a
        # contiguous tensor and returns the view on it


net = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, (5, 5), padding=(2, 2)), nn.Sigmoid(),
    # torch.Size([-1, 6, 28, 28])
    nn.AvgPool2d((2, 2), (2, 2)),
    # torch.Size([-1, 6, 14, 14])
    nn.Conv2d(6, 16, (5, 5)), nn.Sigmoid(),
    # torch.Size([-1, 16, 10, 10])
    nn.AvgPool2d((2, 2), (2, 2)),
    # torch.Size([-1, 16, 5, 5])
    nn.Flatten(),
    # torch.Size([-1, 16 * 5 * 5])
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)


# input = torch.randn(32, 1, 5, 5)
# # input.shape = torch.Size([32, 1, 5, 5])
# # The first number means batch size
# # The second number means the number of input channels
# print("input.shape = ", input.shape)
# m = nn.Sequential(
#     nn.Conv2d(1, 31, 5, 1, 1),
#     # Number of convolution kernel is 31 * 1 = 31
#     # After convolution the result is torch.Size([32, 31, 3, 3])
#     nn.Flatten()
#     # The default dimension is 1
# )
# output = m(input)
# print("output.shape = ", output.shape)
# # torch.Size([32, 279])
# # 279 = 31 * 3 * 3


# # check the net definition
# x = torch.rand((1, 1, 28, 28), dtype=torch.float32)
# for layer in net:
#     x = layer(x)
#     print(layer.__class__.__name__, "output shape: \t", x.shape)
