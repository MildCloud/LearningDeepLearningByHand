import torch
from torch import nn

loss = nn.CrossEntropyLoss(reduction='none')
y_hat = torch.arange(6.0).reshape(2, 3)
y = torch.tensor([0, 1])

for i in range(2):
    y_hat[i][0] = 1
    y_hat[i][1] = 2
    y_hat[i][2] = 3

l = loss(y_hat, y)

print('y_hat = ', y_hat)
print('y = ', y)
print('l = ', l)
