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
# print(mlp_net(x))


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


simple_net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
# x = torch.rand(2, 4)
# print("simple_net[2].state_dict() = ", simple_net[2].state_dict())
# # simple_net[2] = nn.Linear(8, 1)
# # OrderedDict([('weight', tensor([]), ('bias', tensor([]))])
# print("type(simple_net[2].bias) = ", type(simple_net[2].bias))
# # type(simple_net[2].bias) = <class 'torch.nn.parameter.Parameter'>
# print("simple_net[2].bias = ", simple_net[2].bias)
# # simple_net[2].bias = Parameter containing:
# # tensor([-0.1472], requires_grad=True)
# print("simple_net[2].bias.data = ", simple_net[2].bias.data)
# # simple_net[2].bias.data = tensor([-0.1472])
# print("simple_net[2].bias.grad = ", simple_net[2].bias.grad)
# # None

# print(*[(name, param.shape) for name, param in simple_net.named_parameters()])
# # To go through every parameter in the net
#
# print("net.state_dict()['2.bias'].data = ", simple_net.state_dict()['2.bias'].data)
# # simple_net[2].bias


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():
    f_net = nn.Sequential()
    for i in range(4):
        f_net.add_module(f'block {i}', block1())
    return f_net


nested_net = nn.Sequential(block2(), nn.Linear(4, 1))

print("nested_net = ", nested_net)


def init_normal(f_m):
    if type(f_m) == nn.Linear:
        # if the type of f_m is Linear
        nn.init.normal_(f_m.weight, mean=0, std=0.01)
        # nn.init.constant_(f_m.weight, 1)
        nn.init.zeros_(f_m.bias)


nested_net.apply(init_normal)
# apply is a function that can be used to traverse the whole sequence and make changes


def xavier(f_m):
    if type(f_m) == nn.Linear:
        nn.init.xavier_uniform_(f_m.weight)
        # xavier_uniform_(): uniform distribution


def init_42(f_m):
    if type(f_m) == nn.Linear:
        nn.init.constant_(f_m.weight, 42)


def my_init(f_m):
    if type(f_m) == nn.Linear:
        print(
            "Init",
            *[(name, param.shape) for name, param in f_m.name_parameters()][0]
        )
        nn.init.uniform_(f_m.weight, -10, 10)
        f_m.weight.data *= f_m.weight.data.abs() >= 5


net[0].weight.data[:] += 1

# parameter binding
shared = nn.Linear(8, 8)
shared_net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 1))
# shared_net[2].data == shared_net[4].data
