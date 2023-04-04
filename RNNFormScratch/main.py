import torch
from torch import nn
from torch.nn import functional as F
import random
import collections
import re
from matplotlib import pyplot as plt
from d2l import torch as d2l
import os
from IPython import display
import numpy as np
import time
import math


def read_time_machine():
    """Load the time machine dataset into a list of text line"""
    with open('time_machine', 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokensize(f_lines, f_token = 'word'):
    if f_token == 'word':
        return [line.split() for line in f_lines]
    elif f_token == 'char':
        return [list(line) for line in f_lines]
    else: 
        print('error ' + f_token)


class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [
            token for token, freq in self.token_freqs
            if freq >= min_freq and token not in uniq_tokens
        ]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def count_corpus(f_tokens):
    """Count token frequencies."""
    if len(f_tokens) == 0 or isinstance(f_tokens[0], list):
        f_tokens = [token for line in f_tokens for token in line]
    return collections.Counter(f_tokens)


def load_corpus_time_machine(max_tokens = -1):
    lines = read_time_machine()
    tokens = tokensize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[: max_tokens]
    return corpus, vocab



def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.
    Defined in :numref:`sec_calculus`"""
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.
    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def seq_data_iter_random(corpus, batch_size, num_steps):
    """Generate a minibatch of subsequences using random sampling."""
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subsequences = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subsequences * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos:pos + num_steps]
    
    num_batches = num_subsequences // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        x = [data(j) for j in initial_indices_per_batch]
        y = [data(j+1) for j in initial_indices_per_batch]
        yield torch.tensor(x), torch.tensor(y)



def seq_data_iter_sequential(corpus, batch_size, num_steps): 
    """Generate a minibatch of subsequences using sequential partitioning"""
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    xs = torch.tensor(corpus[offset:offset + num_tokens])
    ys = torch.tensor(corpus[offset + 1:offset + num_tokens + 1])
    xs, ys = xs.reshape(batch_size, -1), ys.reshape(batch_size, -1)
    num_batches = xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        x = xs[:, i:i + num_steps]
        y = ys[:, i:i + num_steps]
        yield x, y


class SeqDataLoader:
    """An iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        # corpus is a one dimension list which contains the index of every word or character in the text
        # vocab is an object that mainly has two variable: a dict with {token: index} and a list [token]
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


batch_size, num_steps = 32, 35
# corpus = [vocab[token] for line in tokens for token in line]
train_iter, vocab = load_data_time_machine(batch_size, num_steps)
# train_iter can be used as a iterator, the x, y element in train_iter has the shape of torch.Size([32, 35])
# 35 continuouse words or characters

# print('one hot = ', F.one_hot(torch.tensor([0, 4]), len(vocab)), '\nlen(vocab) = ', len(vocab))
# # one hot =tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
# #          0, 0, 0, 0],
# #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
# #          0, 0, 0, 0]]) 
# # len(vocab) =  28
# # shape = torch.Size([2, 28])

x = torch.arange(10).reshape(2, 5)
# print('F.one_hot(x.T, 28).shape = ', F.one_hot(x.T, 28).shape)
# # T means transpose
# # torch.Size([5, 2, 28])

"""The first dimension represents batch_size and the second dimension represents time step"""


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device = device) * 0.01

    w_xh = normal((num_inputs, num_hiddens))
    w_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    w_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device = device)
    params = [w_xh, w_hh, b_h, w_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
    # The return value should be a tuple to satisfy further need


def rnn(inputs, state, params):
    w_xh, w_hh, b_h, w_hq, b_q = params
    h, = state
    outputs = []
    for x in inputs:
        # print('in input x.shape = ', x.shape)
        # inputs is a 3D tensor, which the first dimension is the length of time step, 
        # the second dimension is batchsize, the third dimension is the length of vocab
        h = torch.tanh(torch.mm(x, w_xh) + torch.mm(h, w_hh) + b_h)
        y = torch.mm(h, w_hq) + b_q
        # y.shape = (batch_size, 28)
        outputs.append(y)
    return torch.cat(outputs, dim=0), (h, )
    # The return size = torch.Size([batchsize * time_step, ])
    

class RNNModuleScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, x, state):
        # x is batchsize * time step
        # Using tranpose means that the second dimension(the column) of the input is a continue text, 
        # and since not using random iter is false, the corresponding column in next batch will just follow the previous batch 
        x = F.one_hot(x.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(x, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().
    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


num_hiddens = 512
net = RNNModuleScratch(len(vocab), num_hiddens, try_gpu(), get_params, init_rnn_state, rnn)
state = net.begin_state(x.shape[0], try_gpu())
# x.shape = torch.Size([2, 5])
# h.shape = torch.Size([2, 512])
# After one hot operation: x.shape = torch.Size([5, 2, 28])
# In input: x.shape = torch.Size([2, 28])
# w_xh.shape = torch.Size([28, 512])
# w_hh.shape = torch.Size([512, 512])
# w_hq.shape = torch.Size([512, 28])
# b_h.shape = torch.Size([512])
# b_q.shape = torch.Size([28])
# y.shape = torch.Size([2, 28])
# outputs.shape = torch.Size([10, 28])

"""Doing forward function will only change h(i.e. the state), other parameters will not be changed"""

# new_state is a tuple which has only one element
# print(y.shape, len(new_state), new_state[0].shape)
# y.shape = torch.Size([10, 28])
# len(new_state) = 1
# new_state[0].shape = torch.Size([2, 512])

# print(torch.zeros(2))
# print(torch.tensor([[1, 2], [3, 4]]) + torch.ones(2))
# boarding casting
# print('len((1, 2)) = ', len((1, 2)))
# print('len(1,) = ', len((1,)))
# len((1, 2)) =  2
# len(1,) =  1
# outputs = []
# for i in range(5):
#     outputs.append(torch.arange(10).reshape(2, 5))
# print(outputs)
# print(torch.cat(outputs, dim=0))
# print(torch.cat(outputs, dim=0).shape)
# print(torch.cat(outputs, dim=1))
# print(torch.cat(outputs, dim=1).shape)
# [tensor([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]]), tensor([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]]), tensor([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]]), tensor([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]]), tensor([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])]
# tensor([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9],
#         [0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9],
#         [0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9],
#         [0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9],
#         [0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])
# torch.Size([10, 5])
# tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3,
#          4],
#         [5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8,
#          9]])
# torch.Size([2, 25])


def predict_ch8(prefix, num_preds, net, vocab, device):
    # num_preds means the number of character or word need to predic
    state = net.begin_state(batch_size=1, device=device)
    # batch_size = 1 means only predict one character
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(1, 1)
    # outputs[-1] means to get the last element in output
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
        # argmax function will return the max value through a given axis
    return ''.join([vocab.idx_to_token[i] for i in outputs])
    # string.join() fucntion will add string between every element in a iterator


# prediction1 = predict_ch8('time traveller ', 10, net, vocab, try_gpu())
# print('prediction1 = ', prediction1)
"""The prediction will only produce random result since the parameters in the net have not be trained."""


def grad_clipping(net, theta):
    """Clip the gradient"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch."""
    # use_random_iter means the (i+1)th element in the first batch does not follow the ith element
    state, timer = None, Timer()
    metric = Accumulator(2)
    # timer.start()
    # There is start function in __init__
    for x_m, y_m in train_iter:
        if state is None or use_random_iter:
            # if use_random_iter, which means that there is no continuity between batches
            state = net.begin_state(batch_size=x_m.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = y_m.T.reshape(-1)
        # -1 means change y_m to one dimension tensor
        # y.shape = torch.Size([35 * 32])
        x_m, y = x_m.to(device), y.to(device)
        # x_m.shape = torch.Size([32, 35])
        # After one hot x_m.shape = torch.Size([35, 32, 28])
        y_hat, state = net(x_m, state)
        # vocab_size = 28
        # num_inputs = num_outputs = 28
        # num_hidden = 512
        # in input: x.shape = torch.Size([32, 28])
        # h.shape = torch.Size([32, 512])
        # h.shape = batch_size * num_hidden
        # w_xh.shape = torch.Size([28, 512])
        # w_hh.shape = torch.Size([512, 512])
        # b_h.shape = torch.Size([512])
        # w_hq.shape = torch.Size([512, 28])
        # b_q.shape = torch.Size([28])
        # state = (h, )
        # h = x @ w_xh + h @ w_hh + b_h
        # h.shape = torch.Size([32, 512])
        # y = h @ w_hq + b_q
        # y.shape = torch.Size([32, 28])
        # outputs.shape = torch.Size([32 * 35, 28]) 
        # loss.shape = torch.Size([]), which means that loss is a scalar
        l = loss(y_hat, y)
        # In cross entropy function the target(y) is converted into one hot code
        # The log in cross entropy means ln
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        # after update state will not be changed
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


class Animator:
    """For plotting data in animation."""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


data_file = os.path.join('.', 'train_result.csv')


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
    with open(data_file, 'w') as f:
        f.write(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
        f.write('\n')
        f.write(predict('time traveller'))
        f.write('\n')
        f.write(predict('traveller'))

num_epochs, lr = 500, 1
# train_ch8(net, train_iter, vocab, lr, num_epochs, try_gpu())

plt.savefig("train_result.png")
plt.show()
