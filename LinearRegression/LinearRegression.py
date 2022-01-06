import random
import torch

# The following function is not used to feed forward, 
# it is used to create an artificial data set
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
    print("X = ", X)
    print("X.shape = ", X.shape)
    print("y = ", y)
    print("y.shape = ", y.shape)
    print("y.reshape((-1, 1)) = ", y.reshape((-1, 1)))
    print("y.reshape.shape = ", y.reshape((-1, 1)).shape)
    # reshape function will not change the shape of the tensor directly
    return X, y.reshape((-1, 1))
    # -1 auto calculate

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('feature:', features[0], '\nlabel:', labels[0])

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    # features.shape = torch.Size([1000, 2]), len = 1000
    indices = list(range(num_examples))
    # indices = [0, 1, 2, ..., 999]
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        # i += batch_size
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        # batch_indices = tensor([indices[i], indices[i + 1], ..., indices[min(i + batch_size, num_examples)]])
        yield features[batch_indices], labels[batch_indices]
        """
        To generate a python iterator use data.DataLoader() or use for and yeild
        """

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)

w = torch.normal(0, 0.01, size = (2, 1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)
#b is a scalar quantity

def linreg(X, w, b):
    '''线性模型'''
    return torch.matmul(X, w) + b
 
def squared_loss(y_hat, y):
    '''均方误差'''
    # y_hat represents the predictic value while y represents the true value
    return (y_hat - y.reshape(y_hat.shape))**2 / 2
    # The above operation is used to modify each single element
    # **2 means square

def stochastic_gradient_descent(parameters, learn_rate, batch_size):
    '''小批量随机梯度下降'''
    with torch.no_grad():
    # 更新的时候不需要梯度参与运算
        for parameter in parameters:
            print('parameter = ', parameter)
            print('parameter.grad = ', parameter.grad)
            parameter -= learn_rate * parameter.grad / batch_size
            parameter.grad.zero_()

learn_rate = 0.03
num_epochs = 3
network = linreg
# feed forward
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        '''
        Each time choose batch_size datas from features and labels
        '''
        # print('w = ', w)
        l = loss(network(X, w, b), y)
        # print('y_predict = ', network(X, w, b))
        # print('y_predict.shape = ', network(X, w, b).shape)
        print('loss = ', l)
        # y.shape = torch.Size([10, 1])
        l.sum().backward()
        '''
        针对batchsize大小的数据集进行网络训练时， 网络中每个参数减去的
        梯度是batchsize中每个样本对应参数梯度求和后的平均值
        '''
        # print('[w, b = ]', [w, b])
        stochastic_gradient_descent([w, b], learn_rate, batch_size)
    with torch.no_grad():
        train_l = loss(network(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
