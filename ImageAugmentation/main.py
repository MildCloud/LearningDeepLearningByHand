import torch
from torch import nn
from torch.utils import data
from torch.nn import functional
import torchvision
from IPython import display
from matplotlib import pyplot as plt, scale
import time
import numpy as np
import os
from PIL import Image
from d2l import torch as d2l


figsize=(3.5, 2.5)
plt.rcParams['figure.figsize'] = figsize
image = Image.open('./cat1.png')
plt.imshow(image)
# plt.show()


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_fashion_mnist`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def apply(f_img, f_aug, f_num_rows=2, f_num_columns=4, f_scale=1.5):
    f_y = [f_aug(f_img) for _ in range(f_num_rows * f_num_columns)]
    show_images(f_y, f_num_rows, f_num_columns, scale=f_scale)
    plt.show()

# apply(image, torchvision.transforms.RandomHorizontalFlip())
# apply(image, torchvision.transforms.RandomVerticalFlip())
shape_aug = torchvision.transforms.RandomResizedCrop(size=(200, 200), scale=(0.1, 1), ratio=(0.5, 2))
# apply(image, shape_aug)
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)
# apply(image, color_aug)

augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), 
    color_aug, shape_aug
])
# apply(image, augs)

# all_images = torchvision.datasets.CIFAR10(train=True, root="../data", download=True)
# show_images(
#     [all_images[i][0] for i in range(32)], 4, 8, scale=0.8
# )
# plt.show()

train_augs = torchvision.transforms.Compose(
    [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor()]
)

test_aug = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)
# The ToTensor function is used to generate the images to 4D tensor


def load_cifar10(f_is_train, f_augs, f_batch_size):
    dataset = torchvision.datasets.CIFAR10(
        root="../data", train=f_is_train, transform=f_augs, download=True
    )
    dataloader = data.DataLoader(
        dataset, batch_size=f_batch_size, shuffle=f_is_train, num_workers=4
    )
    return dataloader


def accuracy(f_y_hat, f_y):
    # calculate the number of correct prediction
    if len(f_y_hat.shape) > 1 and f_y_hat.shape[1] > 1:
        f_y_hat = f_y_hat.argmax(axis=1)
    cmp = f_y_hat.type(f_y.dtype) == f_y
    return float(cmp.type(f_y.dtype).sum())


def train_batch_ch13(f_net, f_x_set, f_y, f_loss, f_trainer, f_devices):
    if isinstance(f_x_set, list):
        f_x_set = [f_x.to(f_devices[0]) for f_x in f_x_set]
    else:
        f_x_set = f_x_set.to(f_devices[0])
    f_y = f_y.to(f_devices[0])
    f_net.train()
    f_trainer.zero_grad()
    f_y_hat = f_net(f_x_set)
    l = f_loss(f_y_hat, f_y)
    l.sum().backward()
    f_trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(f_y_hat, f_y)
    return train_loss_sum, train_acc_sum


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Train a model with mutiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')


batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)


def train_with_data_aug(f_train_augs, f_test_augs, f_net, f_lr=0.001):
    train_iter = load_cifar10(True, f_train_augs, batch_size)
    test_iter = load_cifar10(False, f_test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(f_net.parameters(), lr=f_lr)
    train_ch13(f_net, train_iter, test_iter, loss, trainer, 10, devices)
