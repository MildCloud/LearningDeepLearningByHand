import torch
from torch import nn
from torch.nn import functional

net = torch.nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
x = torch.rand(2, 20)


# Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1).
# print("x = ", x)
# print("net(x) = ", net(x))


# The Sequential method is equivalent to the class method below
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, f_x):
        return self.out(functional.relu(self.hidden(f_x)))


mlp_net = MLP()
print(mlp_net(x))


class MySequential(nn.Module):
    # define a class to realize nn.Sequential
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, f_x):
        for block in self._modules.values():
            f_x = block(f_x)
        return f_x


my_sequential_net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
my_sequential_net(x)


class FixedHiddenMLP(nn.Module):
    # define a class to realize flexible forward function
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, f_x):
        f_x = self.linear(f_x)
        f_x = functional.relu(torch.mm(f_x, self.rand_weight) + 1)
        f_x = self.linear(f_x)
        while f_x.abs().sum() > 1:
            f_x /= 2
        return f_x.sum()


fixed_net = FixedHiddenMLP()
fixed_net(x)


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, f_x):
        return self.linear(self.net(f_x))


chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(x)
