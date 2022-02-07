import torch
from torch import nn
from d2l import torch as d2l


t = 1000
time = torch.arange(1, t + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.02, (t, ))
d2l.plot(time, [x], 'time', x, xlim=[1, 1000], figsize=(6, 3))
