import math
import torch
from torch import nn
from torch.nn import functional as F
import random
import collections
import re
from matplotlib import pyplot as plt
from d2l import torch as d2l


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
    print('offset = ', offset)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    print('num_tokens = ', num_tokens)
    xs = torch.tensor(corpus[offset:offset + num_tokens])
    print('xs = ', xs)
    ys = torch.tensor(corpus[offset + 1:offset + num_tokens + 1])
    print('ys = ', ys)
    xs, ys = xs.reshape(batch_size, -1), ys.reshape(batch_size, -1)
    print('xs = ', xs)
    print('ys = ', ys)
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
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

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
# # torch.Size([5, 2, 2F.one_hot(x.T, 28).shape

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
        # inputs is a 3D tensor, which the first dimension is the length of time step, 
        # the second dimension is batchsize, the third dimension is the length of vocab
        h = torch.tanh(torch.mm(x, w_xh) + torch.mm(h, w_hh) + b_h)
        y = torch.mm(h, w_hq) + b_q
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
y, new_state = net(x.to(try_gpu()), state)
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



