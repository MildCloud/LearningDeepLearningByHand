import torch
from torch import nn
from torch.utils import data
from torch.nn import functional
import torchvision
from torchvision import transforms
from IPython import display
from matplotlib import pyplot as plt
import time
import numpy as np
import os
import d2l.torch as d2l


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1_1_conv=False, stride=(1, 1)):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, (3, 3), stride, 1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, (3, 3), padding=1)
        if use_1_1_conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, (1, 1), stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, f_x):
        f_y = functional.relu(self.bn1(self.conv1(f_x)))
        f_y = self.bn2(self.conv2(f_y))
        if self.conv3:
            f_x = self.conv3(f_x)
        f_y += f_x
        return functional.relu(f_y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, True, (2, 2)))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b1 = nn.Sequential(
    nn.Conv2d(1, 64, (7, 7), (2, 2), 3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d((3, 3), (2, 2), 1)
)

b2 = nn.Sequential(*resnet_block(64, 64, 2, True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 10)
)

# # check the net definition
# x = torch.rand((1, 1, 224, 224), dtype=torch.float32)
# for layer in net:
#     x = layer(x)
#     print(layer.__class__.__name__, "output shape: \t", x.shape)

batch_size = 256
workers = 4


def load_data_fashion_mnist(f_batch_size, resize=None):
    """download Fashion-MNIST data and load them into the memory"""
    f_trans = [transforms.ToTensor()]
    if resize:
        f_trans.insert(0, transforms.Resize(resize))
    f_trans = transforms.Compose(f_trans)
    f_mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=f_trans, download=True)
    f_mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=f_trans, download=True)
    return (data.DataLoader(f_mnist_train, f_batch_size, shuffle=True, num_workers=workers),
            data.DataLoader(f_mnist_test, f_batch_size, shuffle=True, num_workers=workers))


train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)


def accuracy(f_y_hat, f_y):
    # calculate the number of correct prediction
    if len(f_y_hat.shape) > 1 and f_y_hat.shape[1] > 1:
        f_y_hat = f_y_hat.argmax(axis=1)
    cmp = f_y_hat.type(f_y.dtype) == f_y
    return float(cmp.type(f_y.dtype).sum())


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def evaluate_accuracy_gpu(f_net, f_data_iter, device=None):
    """use GPU to calculate the accuracy"""
    if isinstance(f_net, torch.nn.Module):
        f_net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    for f_x_set, f_y in f_data_iter:
        if isinstance(f_x_set, list):
            f_x_set = [f_x.to(device) for f_x in f_x_set]
        else:
            f_x_set = f_x_set.to(device)
        f_y = f_y.to(device)
        metric.add(accuracy(net(f_x_set), f_y), f_y.numel())
    return metric[0] / metric[1]


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


class Animator:  # @save
    """For plotting data in animation."""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


data_file = os.path.join('.', 'train_result.csv')


def train(f_net, f_train_iter, f_test_iter, f_num_epochs, f_lr, f_num_gpus):
    """Train a model with a GPU (defined in Chapter 6)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    f_net.apply(init_weights)
    devices = [try_gpu(i) for i in range(f_num_gpus)]
    f_net = nn.DataParallel(f_net, device_ids=devices)
    # use nn.DataParallel to copy data to each gpu divice
    optimizer = torch.optim.SGD(f_net.parameters(), lr=f_lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, f_num_epochs], legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(f_train_iter)
    for epoch in range(f_num_epochs):
        metric = Accumulator(3)
        f_net.train()
        for i, (f_x_set, f_y) in enumerate(f_train_iter):
            # f_x_set.shape = torch.Size[256, 1, 28, 28]
            timer.start()
            optimizer.zero_grad()
            # The self-defined stochastic gradient descent function will set the grd to be zero at the end of the loop
            f_x_set, f_y = f_x_set.to(devices[0]), f_y.to(devices[0])
            f_y_hat = f_net(f_x_set)
            f_y_hat = f_y_hat.to(devices[0])
            l = loss(f_y_hat, f_y)
            l = l.to(devices[0])
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * f_x_set.shape[0], accuracy(f_y_hat, f_y), f_x_set.shape[0])
            timer.stop()
            f_train_loss = metric[0] / metric[2]
            # The loss is calculated in one batch size
            f_train_accuracy = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (f_train_loss, f_train_accuracy, None))
        f_test_accuracy = evaluate_accuracy_gpu(f_net, f_test_iter, devices[0])
        animator.add(epoch + 1, (None, None, f_test_accuracy))
    print(f'loss {f_train_loss:.3f}, train accuracy {f_train_accuracy:.3f}, test accuracy {f_test_accuracy:.3f}')
    print(f'{metric[2] * f_num_epochs / timer.sum():.1f} examples/sec' f' on {str(devices)}')
    with open(data_file, 'w') as f:
        f.write(f'loss {f_train_loss:.3f}, train accuracy {f_train_accuracy:.3f}, test accuracy {f_test_accuracy:.3f}\n')
        f.write(f'{metric[2] * f_num_epochs / timer.sum(): .1f} examples/sec on {str(devices)}\n')


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


lr, num_epochs = 0.05, 10
train(net, train_iter, test_iter, num_epochs, lr, 4)

plt.show()
