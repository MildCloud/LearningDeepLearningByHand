import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn
from IPython import display
from d2l import torch as d2l

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


train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hidden = 784, 10, 256
# num_inputs and num_outputs are determined by the data
# num_hidden is used to realized a single hidden layer

# w1_matrix = nn.Parameter() : to declare that it is a torch parameter
w1_matrix = nn.Parameter(torch.normal(0, 0.01, size=(num_inputs, num_hidden), requires_grad=True))
# print("w1_matrix.shape = ", w1_matrix.shape)
# # w1_matrix.shape = torch.Size([784, 256]), row = 784, column = 256
b1 = nn.Parameter(torch.zeros(num_hidden, requires_grad=True))
# b1.shape = torch.Size([256])
w2_matrix = nn.Parameter(torch.normal(0, 0.01, size=(num_hidden, num_outputs), requires_grad=True))
# w2_shape = torch.size([256, 10])
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [w1_matrix, b1, w2_matrix, b2]


def relu(x):
    a = torch.zeros_like(x)
    # the shape of a is the same as x
    return torch.max(x, a)


def softmax(f_x_matrix):
    # # f_x_matrix.shape = torch.Size([256, 10])
    # print("f_matrix.shape = ", f_x_matrix.shape)
    f_x_matrix_exp = torch.exp(f_x_matrix)
    # print("f_x_matrix_exp = ", f_x_matrix_exp)
    partition = f_x_matrix_exp.sum(1, keepdim=True)
    # print("partition = ", partition)
    return f_x_matrix_exp / partition  # use broadcasting


def net(x):
    x = x.reshape(-1, num_inputs)
    h = relu(x @ w1_matrix + b1)
    # @ is used as matrix multiplication
    return softmax(h @ w2_matrix + b2)


def cross_entropy(f_y_hat, f_y):
    # print("f_y_hat[range(len(f_y_hat)), f_y] = ", f_y_hat[range(len(f_y_hat)), f_y])
    # print("f_y_hat[range(len(f_y_hat)), f_y].shape = ", f_y_hat[range(len(f_y_hat)), f_y].shape)
    return -torch.log(f_y_hat[range(len(f_y_hat)), f_y])
    # len(f_y_hat) = 256
    # f_y_hat[range(len(f_y_hat)), f_y].shape = torch.Size([256])


num_epochs, learn_rate = 10, 0.1


def accuracy(f_y_hat, f_y):
    # calculate the number of correct prediction
    if len(f_y_hat.shape) > 1 and f_y_hat.shape[1] > 1:
        f_y_hat = f_y_hat.argmax(axis=1)
    cmp = f_y_hat.type(f_y.dtype) == f_y
    return float(cmp.type(f_y.dtype).sum())


# print("accuracy = ", accuracy(test_index_y_hat, test_index_y))


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


def train_epoch_softmax(f_network, f_train_iter, loss, f_updater):
    if isinstance(f_network, torch.nn.Module):
        f_network.train()
    metric = Accumulator(3)
    for X, y in f_train_iter:
        # X.shape = torch.Size[256, 1, 28, 28]
        y_hat = f_network(X)
        # softmax
        f_l = loss(y_hat, y)
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


def train_softmax(f_network, f_train_iter, f_test_iter, loss, f_num_epochs, f_updater):
    """Train the model define in Chapter 3"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train accuracy', 'test accuracy'])
    for epoch in range(f_num_epochs):
        train_metrics = train_epoch_softmax(f_network, f_train_iter, loss, f_updater)
        test_accuracy = evaluate_accuracy(f_network, f_test_iter)
        animator.add(
            epoch + 1, train_metrics + (test_accuracy,)
        )
    train_loss, train_accuracy = train_metrics
    assert train_loss < 0.5, train_loss
    assert 0.7 < train_accuracy <= 1, train_accuracy
    assert 0.7 < train_accuracy <= 1, test_accuracy


def stochastic_gradient_descent(parameters, f_learn_rate, f_batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        # 更新的时候不需要梯度参与运算
        for parameter in parameters:
            # print('parameter = ', parameter)
            # print('parameter.grad = ', parameter.grad)
            parameter -= f_learn_rate * parameter.grad / f_batch_size
            parameter.grad.zero_()


def updater(f_batch_size):
    return stochastic_gradient_descent(params, learn_rate, f_batch_size)

train_softmax(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
d2l.plt.show()
