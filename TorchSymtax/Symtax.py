import torch
from torch.functional import Tensor

x = torch.arange(4.0)
print("x = ", x)

x.requires_grad_(True)
y = torch.dot(x,x)
print("y = ", y)
y *= 2
print("y = ", y)
y.backward()
print("x.grad = ", x.grad)
#The result is equal to 4 * x

x.grad.zero_()
y = x.sum()
y.backward()
print("x.grad = ", x.grad)

x.grad.zero_()
y = x * x
#operate on each element
print("y = ", y)
y.sum().backward()
print("y = ", y)
print("x.grad = ", x.grad)
#The result is equal to 2 * x

x.grad.zero_()
u = y.detach()
z = u * x
z.sum().backward()
print("x.grad = ", x.grad)
print("Compare to u", x == u)

x.grad.zero_()

def f(a):
    b = a * 2
    while b.norm() < 100:
        b *= 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn (size = (), requires_grad = True)
print(a)
d = f(a)
print(d)
d.backward()
print(a.grad)
b = torch.tensor([-1.0])
c = f(b)
print(c)
