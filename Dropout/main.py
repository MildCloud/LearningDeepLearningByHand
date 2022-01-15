import torch
import torchvision
from torch.utils import data
from torch import nn
from d2l import torch as d2l
from torchvision import transforms
from IPython import display


def dropout_layer(f_x, f_dropout):
    assert 0 <= f_dropout <= 1
    if f_dropout == 1:
        return torch.zeros_like(f_x)
    if f_dropout == 0:
        return f_x
    mask = (torch.rand(f_x.shape) > f_dropout).float()
    return mask * f_x / (1.0 - f_dropout)
    # Multiplication is more faster than index ([mask])
    # Consider the  consuming of GPU


num_inputs, num_outputs, num_hidden1, num_hidden2 = 784, 10, 256, 256

dropout1, dropout2 = 0.2, 0.5


class Net(nn.Module):
    def __init__(self, c_num_inputs, c_num_outputs, c_num_hidden1, c_num_hidden2,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = c_num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(c_num_inputs, c_num_hidden1)
        self.lin2 = nn.Linear(c_num_hidden1, c_num_hidden2)
        self.lin3 = nn.Linear(c_num_hidden2, c_num_outputs)
        self.relu = nn.ReLU()

    def forward(self, f_x):
        h1 = self.relu(self.lin1(f_x.reshape(-1, self.num_inputs)))
        if self.training:
            h1 = dropout_layer(h1, dropout1)
        h2 = self.relu(self.lin2(h1))
        if self.training:
            h2 = dropout_layer(h2, dropout2)
        out = self.lin3(h2)
        return out


net = Net(num_inputs, num_outputs, num_hidden1, num_hidden2)

num_epochs, learn_rate, batch_size, workers = 10, 0.5, 256, 4
loss = nn.CrossEntropyLoss()


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
    # To return two iterator, () is used


train_iter, test_iter = load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=learn_rate)


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


def evaluate_accuracy(f_network, f_data_iter):
    # use a data_iter to check evaluate the accuracy of the specific network
    if isinstance(f_network, torch.nn.Module):
        f_network.eval()  # change the model to evaluation
    metric = Accumulator(2)
    for X, y in f_data_iter:
        metric.add(accuracy(f_network(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_softmax(f_network, f_train_iter, f_loss, f_updater):
    if isinstance(f_network, torch.nn.Module):
        f_network.train()
    metric = Accumulator(3)
    for X, y in f_train_iter:
        # X.shape = torch.Size[256, 1, 28, 28]
        y_hat = f_network(X)
        # softmax
        f_l = f_loss(y_hat, y)
        # cross entropy
        # f_l.shape = torch.Size([256])
        if isinstance(f_updater, torch.optim.Optimizer):
            # use torch.optim.SGD
            f_updater.zero_grad()
            f_l.backward()
            f_updater.step()
            metric.add(float(f_l) * len(y), accuracy(y_hat, y), y.numel())
        else:
            f_l.sum().backward()
            f_updater(X.shape[0])
            metric.add(float(f_l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]
    # metric[0]: all the sum of loss function
    # metric[1]:


class Animator:  # @save
    """For plotting data in animation."""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
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


def train_softmax(f_network, f_train_iter, f_test_iter, f_loss, f_num_epochs, f_updater):
    """Train the model define in Chapter 3"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train accuracy', 'test accuracy'])
    for epoch in range(f_num_epochs):
        train_metrics = train_epoch_softmax(f_network, f_train_iter, f_loss, f_updater)
        test_accuracy = evaluate_accuracy(f_network, f_test_iter)
        animator.add(
            epoch + 1, train_metrics + (test_accuracy,)
        )
    train_loss, train_accuracy = train_metrics
    assert train_loss < 0.5, train_loss
    assert 0.7 < train_accuracy <= 1, train_accuracy
    assert 0.7 < train_accuracy <= 1, test_accuracy


train_softmax(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()
