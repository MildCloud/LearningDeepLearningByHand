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


def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    # The default value of reduction is mean and it can also be set to sum or none
    num_epochs, lr = 100, 0.003
    trainer = torch.optim.SGD([{"params": net[0].weight, 'weight_decay': wd},
                              {"params": net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        for x, y in train_iter:
            trainer.zero_grad()
            loss_val = loss(net(x), y)
            loss_val.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, validation_iter, loss)))
    print("L2 norm of w:", net[0].weight.norm().item())


train_concise(0)
train_concise(3)
d2l.plt.show()
