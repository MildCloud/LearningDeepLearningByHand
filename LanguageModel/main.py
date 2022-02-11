import random
import torch
import collections
import re
from matplotlib import pyplot as plt


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


tokens = tokensize(read_time_machine())
corpus = [token for line in tokens for token in line]
vocab = Vocab(corpus)
# print("vocab.idx_to_token[:10]", vocab.idx_to_token[:10])
freqs = [freq for _, freq in vocab.token_freqs]
# print('freqs[:10] = ', freqs[:10])


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

    # Return True if `X` (tensor or list) has 1 axis
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


# plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='log', yscale='log')
# plt.savefig('frequency.png')
# plt.show()

bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = Vocab(bigram_tokens)
# print('bigram_vovab.token_freqs[:10] = ', bigram_vovab.token_freqs[:10])

# zip_list = [1, 2, 3, 4, 5, 6]
# print('zip_list[:-1] = ', zip_list[:-1])
# print('zip_list[:-2] = ', zip_list[:-2])
# print('zip_list[1:-1] = ', zip_list[1:-1])
# # zip_list[:-1] =  [1, 2, 3, 4, 5]
# # zip_list[:-2] =  [1, 2, 3, 4]
# # zip_list[1:-1] =  [2, 3, 4, 5]
# pair_list = [pair for pair in zip(zip_list[:-1], zip_list[1:])]
# print("pair_list = ", pair_list)
# # pair_list =  [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = Vocab(trigram_tokens)
# print('trigram_vocab.token_freqs[:5] = ', trigram_vocab.token_freqs[:5])
# trigram_vocab.token_freqs[:10] =  [(('the', 'time', 'traveller'), 59), 
#                                    (('the', 'time', 'machine'), 30), 
#                                    (('the', 'medical', 'man'), 24), 
#                                    (('it', 'seemed', 'to'), 16), 
#                                    (('it', 'was', 'a'), 15), 

bigram_freqs = [freq for _, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for _, freq in trigram_vocab.token_freqs]
plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x', ylabel='frequency: n(x)', 
    xscale='log', yscale='log', legend=['unigram', 'bigram', 'trigram'])
# plt.savefig('3frquency.png')
# plt.show()
