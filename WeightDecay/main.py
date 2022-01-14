# %matplotlib inline
import random

import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l


def synthetic_data(f_w, f_b, f_num_samples):
    f_x = torch.normal(0, 1, (f_num_samples, len(f_w)))
    f_y = torch.matmul(f_x, f_w) + f_b
    f_y += torch.normal(0, 0.01, f_y.shape)
    return f_x, f_y.reshape((-1, 1))


def load_array(f_data_arrays, f_batch_size, is_train=True):
    f_dataset = data.TensorDataset(*f_data_arrays)
    return data.DataLoader(f_dataset, f_batch_size, shuffle=is_train)


n_train, n_validation, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
train_data = synthetic_data(true_w, true_b, n_train)
train_iter = load_array(train_data, batch_size)
validation_data = synthetic_data(true_w, true_b, n_validation)
validation_iter = load_array(validation_data, batch_size, is_train=False)


def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


def linear_regression(f_x, f_w, f_b):
    """The linear regression model."""
    return torch.matmul(f_x, f_w) + f_b


def squared_loss(y_hat, y):
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def stochastic_gradient_descent(parameters, f_learn_rate, f_batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        # 更新的时候不需要梯度参与运算
        for parameter in parameters:
            parameter -= f_learn_rate * parameter.grad / f_batch_size
            parameter.grad.zero_()


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def evaluate_loss(f_net, f_data_iter, f_loss):
    """Evaluate the loss of a model on the given dataset."""
    metric = Accumulator(2)  # Sum of losses, no. of examples
    for X, y in f_data_iter:
        out = f_net(X)
        y = y.reshape(out.shape)
        loss = f_loss(out, y)
        metric.add(loss.sum(), loss.numel())
    return metric[0] / metric[1]


def train(regularization_constant):
    w, b = init_params()
    net, loss = lambda l_x: linear_regression(l_x, w, b), squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        for x, y in train_iter:
            reg_loss = loss(net(x), y) + regularization_constant * l2_penalty(w)
            reg_loss.sum().backward()
            stochastic_gradient_descent([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, validation_iter, loss)))
    print("L2 norm of w:", torch.norm(w).item())


# train(0)
# # The gap between train loss and validation loss is huge, which shows that the model is over fitting
# train(3)
# # The gap between train loss and validation loss is smaller
d2l.plt.show()
