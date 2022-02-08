from audioop import mul
import torch
from torch import nn
from torch.utils import data
from matplotlib import pyplot as plt
from d2l import torch as d2l

# print(torch.arange(4))
# print(torch.arange(1, 4))
# print(torch.zeros(3, 5).shape)
# tensor([0, 1, 2, 3])
# tensor([1, 2, 3])
# torch.Size([3, 5])

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.
    Defined in :numref:`sec_calculus`"""
    plt.rcParams['figure.figsize'] = figsize


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


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


t = 1000
time = torch.arange(1, t + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (t, ))
plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
plt.savefig("sin.png")
# plt.show()


tau = 4
features = torch.zeros(t - tau, tau)
for i in range(tau):
    features[:, i] = x[i: t - tau + i]
labels = x[tau:].reshape(-1, 1)
# print("labels.shape = ", labels.shape)
# # labels.shape =  torch.Size([996, 1])
# # features.shape = torch.Size([996, 4])

# tensor1 = torch.zeros(3, 4)
# tensor2 = torch.arange(12)
# for i in range(tensor1.shape[1]):
#     tensor1[:, i] = tensor2[i : tensor1.shape[0] + i]
#     print("tesnsor1 = ", tensor1)
# print(tensor1[0][1])
# print(tensor1[1][1])    
# print(tensor1[2][1])
# print(tensor1[:, 1])
# print(tensor1[:][1])
# print(tensor1[1][:])
# tensor(1.)
# tensor(2.)
# tensor(3.)
# tensor([1., 2., 3.])
# tensor([1., 2., 3., 4.])
# tensor([1., 2., 3., 4.])


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size, n_train = 16, 600
train_iter = load_array((features[: n_train], labels[: n_train]), batch_size, is_train=True)

# tensor1 = torch.arange(8)
# tensor1 = tensor1.reshape(4, 2)
# tensor2 = torch.arange(2, 10)
# tensor2 = tensor2.reshape(4, 2)
# for load_tensor in load_array((tensor1, tensor2), 2, False):
#     print("load_tensor = ", load_tensor)
# # load_tensor =  [tensor([[0, 1, 2, 3]]), tensor([[2, 3, 4, 5]])]
# # load_tensor =  [tensor([[4, 5, 6, 7]]), tensor([[6, 7, 8, 9]])]
# # load_tensor =  [tensor([[0, 1, 2, 3],
# #         [4, 5, 6, 7]]), tensor([[2, 3, 4, 5],
# #         [6, 7, 8, 9]])]


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


net = nn.Sequential(
    nn.Linear(4, 10), 
    nn.ReLU(), 
    nn.Linear(10, 1)
)


net.apply(init_weights)
loss = nn.MSELoss(reduction='none')


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def evaluate_loss(net, data_iter, f_loss):
    """Evaluate the loss of a model on the given dataset.

    Defined in :numref:`sec_model_selection`"""
    metric = Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = f_loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def train(f_net, f_train_iter, f_loss, f_epochs, f_lr):
    trainer = torch.optim.Adam(f_net.parameters(), f_lr)
    for epoch in range(f_epochs):
        for x_set, y in f_train_iter:
            trainer.zero_grad()
            l = f_loss(net(x_set), y)
            l.sum().backward()
            trainer.step()
        print(
            f'epoch {epoch + 1}, '
            f'loss: {evaluate_loss(f_net, f_train_iter, f_loss):f}'
        )


train(net, train_iter, loss, 5, 0.01)

onestep_preds = net(features)
plot([time, time[tau:]], [x.detach().numpy(), onestep_preds.detach().numpy()], 'time', 
    'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3)
)
plt.savefig('prediction_result1.png')


multistep_preds = torch.zeros(t)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, t):
    multistep_preds[i] = net(
        multistep_preds[i - tau: i].reshape(1, -1)
    )
plot([time, time[tau:], time[n_train + tau:]], 
    [x.detach().numpy(), onestep_preds.detach().numpy(), multistep_preds[n_train + tau:].detach().numpy()], 
    'time', 'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3)
)
plt.savefig('prediction_result2.png')
plt.show()
