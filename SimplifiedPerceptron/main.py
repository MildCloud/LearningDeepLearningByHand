import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from torch import nn
from d2l import torch as d2l

batch_size = 256
workers = 4


def load_data_fashion_mnist(f_batch_size, resize=None):
    """download Fashion-MNIST data and load them into the memory"""
    f_trans = [transforms.ToTensor()]
    if resize:
        f_trans.insert(0, transforms.Resize(resize))
    f_trans = transforms.Compose(f_trans)
    f_mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=f_trans, download=True)
    f_mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=f_trans, download=True)
    return (data.DataLoader(f_mnist_train, f_batch_size, shuffle=True, num_workers=workers),
            data.DataLoader(f_mnist_test, f_batch_size, shuffle=True, num_workers=workers))


train_iter, test_iter = load_data_fashion_mnist(batch_size)
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))
# Pytorch will not adjust the shape of inputs implicitly
# Flatten is used to adjust the shape from dim = 0
# e.g. tensor = torch.tensor([[[1, 2], [3, 4]],[[5, 6], [7, 8]]])
# torch.flatten(tensor) = ([1, 2, 3, 4, 5, 6, 7, 8])


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        # mean = m.weight(default value is 0)


net.apply(init_weights)
loss = nn.CrossEntropyLoss()
# nn.CrossEntropyLoss = nn.softmax + nn.log + nn.NLLLoss

trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
