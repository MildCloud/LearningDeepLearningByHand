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


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (
            normal((num_inputs, num_hiddens)), 
            normal((num_hiddens, num_hiddens)), 
            torch.zeros(num_hiddens, device=device)
        )

    w_xz, w_hz, b_z = three()
    w_xr, w_hr, b_r = three()
    w_xh, w_hh, b_h = three()

    w_hq = normal(num_hiddens, num_outputs)
    b_q = torch.zeros(num_outputs, device=device)

    params = [w_xz, w_hz, b_z, w_xr, w_hr, b_r, w_xh, w_hh, b_h, w_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def gru(inputs, state, params):
    w_xz, w_hz, b_z, w_xr, w_hr, b_r, w_xh, w_hh, b_h, w_hq, b_q = params
    h, = state
    outputs = []
    for x in inputs:
        r = F.sigmoid((x @ w_xr) + (h @ w_hr) + b_r)
        z = F.sigmoid((x @ w_xz) + (h @ w_hz) + b_z)
        # The only difference between torch.sigmoid and torch.nn.functional.sigmoid is that: 
        # torch. will make pyhon function call and torch.nn.function will make c function call
        h_tilda = F.tanh((x @ w_xh) + ((r * h) @ w_hh) + b_h)
        h = z * h + (1 - z) * h_tilda
        h = z * h + (1 - z) * h_tilda
        y = h @ w_hq + b_q
        outputs.append(y)
    return torch.cat(outputs, dim=0), (h, )


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().
    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


class RNNModuleScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, x, state):
        x = F.one_hot(x.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(x, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)



batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size=batch_size, num_steps=num_steps)

vacab_size, num_hiddens, device = len(vocab), 256, try_gpu()
num_epochs, lr = 500, 1
model = RNNModuleScratch(len(vocab), num_hiddens, device, get_params(), init_gru_state, gru)

