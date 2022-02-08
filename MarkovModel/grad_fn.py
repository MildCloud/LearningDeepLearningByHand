import torch
from torch import nn


def stochastic_gradient_descent(parameters, f_learn_rate, f_batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        # 更新的时候不需要梯度参与运算
        for parameter in parameters:
            # print('parameter = ', parameter)
            # print('parameter.grad = ', parameter.grad)
            parameter -= f_learn_rate * parameter.grad /f_batch_size
            parameter.grad.zero_()


x = torch.tensor([[0.0, 1], [2, 3]])
w1 = torch.tensor([[0.4], [0.5]])
w1.requires_grad_(True)
w2 = torch.tensor([[0.4], [0.5]])
w2.requires_grad_(True)
w3 = torch.tensor([[0.4], [0.5]])
w3.requires_grad_(True)
w4 = torch.tensor([[0.4], [0.5]])
w4.requires_grad_(True)

y_hat1 = torch.matmul(x, w1)
y_hat2 = torch.matmul(x, w2)
y_hat3 = torch.matmul(x, w3)
y_hat4 = torch.matmul(x, w4)
y = torch.tensor([[1.0], [1]])
print("y = ", y)
print("y_hat = ", y_hat1)

l1 = nn.MSELoss(reduction='none')(y_hat1, y)
print("l1 = ", l1)
l1.sum().backward()
print("w1.grad = ", w1.grad)
torch.optim.SGD([w1], 0.1).step()
print("w1 = ", w1)

l2 = nn.MSELoss(reduction='sum')(y_hat2, y)
print("l2 = ", l2)
l2.backward()
print("w2.grad = ", w2.grad)
torch.optim.SGD([w2], 0.1).step()
print("w2 = ", w2)

l3 = nn.MSELoss(reduction='none')(y_hat3, y)
print("l3 = ", l3)
l3.sum().backward()
print("w3.grad = ", w3.grad)
stochastic_gradient_descent([w3], 0.1, 2)
print("w3 = ", w3)

l4 = nn.MSELoss(reduction='sum')(y_hat4, y)
print("l4 = ", l4)
l4.backward()
print("w4.grad = ", w4.grad)
stochastic_gradient_descent([w4], 0.1, 2)
print("w4 = ", w4)

# y =  tensor([[1.],
#         [1.]])
# y_hat =  tensor([[0.5000],
#         [2.3000]], grad_fn=<MmBackward0>)
# l1 =  tensor([[0.2500],
#         [1.6900]], grad_fn=<MseLossBackward0>)
# w1.grad =  tensor([[5.2000],
#         [6.8000]])
# w1 =  tensor([[-0.1200],
#         [-0.1800]], requires_grad=True)
# l2 =  tensor(1.9400, grad_fn=<MseLossBackward0>)
# w2.grad =  tensor([[5.2000],
#         [6.8000]])
# w2 =  tensor([[-0.1200],
#         [-0.1800]], requires_grad=True)
# l3 =  tensor([[0.2500],
#         [1.6900]], grad_fn=<MseLossBackward0>)
# w3.grad =  tensor([[5.2000],
#         [6.8000]])
# w3 =  tensor([[0.1400],
#         [0.1600]], requires_grad=True)
# l4 =  tensor(1.9400, grad_fn=<MseLossBackward0>)
# w4.grad =  tensor([[5.2000],
#         [6.8000]])
# w4 =  tensor([[0.1400],
#         [0.1600]], requires_grad=True)
