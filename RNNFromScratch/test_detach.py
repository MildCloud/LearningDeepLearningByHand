import torch

a = torch.tensor([[1., 2], [3, 4]])
a.requires_grad_(True)
b = torch.tensor([[2., 2], [2, 2]])

# c = torch.matmul(a, b)
# d = c
# d.sum().backward()
# print('a.grad = ', a.grad)
# Success to call the backward function

# c = torch.matmul(a, b)
# d = c.view(2, 2)
# d.sum().backward()
# print('a.grad = ', a.grad)
# Success to call the backward function

# c = torch.matmul(a, b)
# d = c
# d.detach()
# c.sum().backward()
# print('a.grad = ', a.grad)
# Success to call the backward function

# c = torch.matmul(a, b)
# d = c
# d.detach_()
# d.sum().backward()
# print('a.grad = ', a.grad)
# Fail to call the backward function
