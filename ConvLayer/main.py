import torch
from torch import nn


def corr2d(f_x, f_k):
    # Realize 2d convolution
    f_h, f_w = f_k.shape
    f_y = torch.zeros(f_x.shape[0] - f_h + 1, f_x.shape[1] - f_w + 1)
    for i in range(f_y.shape[0]):
        for j in range(f_y.shape[1]):
            f_y[i, j] = (f_x[i: i + f_h, j: j + f_w] * f_k).sum()
    return f_y


# test_x = torch.arange(9)
# test_x = test_x.reshape(3, 3)
# test_k = torch.arange(4)
# test_k = test_k.reshape(2, 2)
# print("test_x = ", test_x)
# print("test_k = ", test_k)
# print("corr2d(test_x, test_k) = ", corr2d(test_x, test_k))
# # The result of corr2d is automatically converted into float


class Conv2D(nn.Module):
    # Realize 2D Convolution layer
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, f_x):
        return corr2d(f_x, self.weight) + self.bias


# To realize a simple image color scope detection
x = torch.ones(6, 8)
x[:, 2:6] = 0
k = torch.tensor([[1.0, -1]])
y = corr2d(x, k)
# print("x = ", x)
# print("k = ", k)
# print("y = ", y)
# x = x.t()
# y = corr2d(x, k)
# print("After transpose x = ", x)
# print("k = ", k)
# print("y = ", y)
# # The kernel can only be used to detect vertical line


conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
# The number of input channel and output channel are both 1

x = x.reshape(1, 1, 6, 8)
y = y.reshape(1, 1, 6, 7)
# The first 1 represents the number of batch size
# The second 1 represents the number of channel

for i in range(10):
    y_hat = conv2d(x)
    loss = (y_hat - y) ** 2
    conv2d.zero_grad()
    loss.sum().backward()
    learn_rate = 3e-2
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f"batch {i + 1}, loss {loss.sum(): .3f}")


print("trained weight = ", conv2d.weight.data)