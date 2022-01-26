import torch
from torch import nn


def pool2d(f_x, f_pool_size, f_mode='max'):
    f_p_h, f_p_w = f_pool_size
    f_y = torch.zeros(f_x.shape[0] - f_p_h + 1, f_x.shape[1] - f_p_w + 1)
    for i in range(f_y.shape[0]):
        for j in range(f_y.shape[1]):
            if f_mode == 'max':
                f_y[i, j] = f_x[i:i + f_p_h, j:j + f_p_w].max()
            elif f_mode == 'avg':
                f_y[i, j] = f_x[i:i + f_p_h, j:j + f_p_w].mean()
    return f_y


x = torch.arange(9.0)
# torch.range will be deleted in later version
x = x.reshape(3, 3)
print("max pooling = ", pool2d(x, (2, 2)))
print("average pooling = ", pool2d(x, (2, 2), 'avg'))

x = torch.arange(16.0).reshape(1, 1, 4, 4)
print("x = ", x)
print("nn.MaxPool2d() = ", nn.MaxPool2d(3))
# nn.MaxPool2d() =  MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
print("nn.MaxPool2d(3) = ", nn.MaxPool2d(3)(x))
# The dimension after pooling will not change
# In pytorch, the default kernel size is the same as the stride
# pool2d_torch = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3))

x = torch.cat((x, x + 1), 1)
# In concat() function the tensors are concatenated along the existing axis
# whereas in stack() function the tensors are concatenated along a new axis that does not exist for the individual tensors.
print("nn.MaxPool2d(3, padding=1, stride=2)(x) = ", nn.MaxPool2d(3, padding=1, stride=2)(x))
