import numpy as np
import torch
from torch.utils import data
from torch import nn


def synthetic_data(w, b, num_examples):
    '''
    Generate y = Xw + b + noise
    '''
    X = torch.normal(0, 1, (num_examples, len(w)))
    # The mean value is 0, the satndard deviation is 1, 
    # row is num_examples, column is len(w)
    y = torch.matmul(X,w) + b
    # matrix multiplication
    y += torch.normal(0, 0.01, y.shape)
    # y :  1000 row and 1 column (one dimension)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle = is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))
# change data_iter to a python iterator

network = nn.Sequential(nn.Linear(2, 1))
'''
Linear 线性层或全连接层
The input dimension is 2 and the output dimension is 1
Sequantial is a container that can be viewed as list of layers
'''
network[0].weight.data.normal_(0, 0.01)
'''
network[0] means list[0]
.weight 取权重
.data 取权重向量里面的数据
.normal_ 正太分布均值为0， 方差为0.01
'''
network[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(network.parameters(), lr = 0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(network(X), y)
        # The function MSELoss() has done the sum
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(network(features), labels)
    print(f'epoch {epoch + 1}, loss {l : f}')
