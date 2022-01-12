import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

""" Use a synthetic data set to illustrate over fitting and under fitting"""
max_degree = 20
n_train, n_validation = 100, 100
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_validation, 1))
# random.normal(loc=0.0, scale=1.0, size=None)
# Parameters
# loc: float or array_like of floats
# Mean (“centre”) of the distribution.
#
# scale: float or array_like of floats
# Standard deviation (spread or “width”) of the distribution. Must be non-negative.
#
# size: int or tuple of ints, optional Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples
# are drawn. If size is None (default), a single value is returned if loc and scale are both scalars. Otherwise,
# np.broadcast(loc, scale).size samples are drawn.
#
# Returns
# nd_array or scalar
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
# print("np.arange(max_degree).reshape(1, -1) = ", np.arange(max_degree).reshape(1, -1))
# # np.arange(max_degree).reshape(1, -1).shape = (1, 20)
# print("power = ", poly_features)
# # features.shape = (200, 1)
# # use broadcasting to let features.shape = (200, 20)
# # copy every element in features 20 times
# # requires the two variables to have 2 dimensions
# print("poly_features.shape = ", poly_features.shape)
# # poly_features.shape = (200, 20)
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)
    # math.gamma(n) = (n - 1)!
labels = np.dot(poly_features, true_w)
# true_w.shape = (20)
# labels.shape = (200)
labels += np.random.normal(scale=0.1, size=labels.shape)

true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32)
                                           for x in [true_w, features, poly_features, labels]]


def evaluate_loss(f_net, f_data_iter, f_loss):
    """To evaluate the validation loss"""
    metric = d2l.Accumulator(2)
    for X, y in f_data_iter:
        output = f_net(X)
        y = y.reshape(output.shape)
        validation_loss = f_loss(output, y)
        metric.add(validation_loss.sum(), validation_loss.numel())
    return metric[0] / metric[1]


def train(train_features, validation_features, train_labels, validation_labels, num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    # print("input_shape = ", input_shape)
    # # input_shape = 4
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    validation_iter = d2l.load_array((validation_features, validation_labels.reshape(-1, 1)), batch_size,
                                     is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, validation_iter, loss)))
    print('weight:', net[0].weight.data.numpy())


# train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
# # the weights after training are very close to the true weights
# # the gap between train loss and validation loss is quite small
# train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])
# # Both train loss and validation loss are large and the gap is random
# # Under fitting
# train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])
# # Gap is large, which shows that over fitting occurs
d2l.plt.show()
