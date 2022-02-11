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
# print('zip_list[1:3] = ', zip_list[1:6])
# print('zip_list[:-1] = ', zip_list[:-1])
# print('zip_list[:-2] = ', zip_list[:-2])
# print('zip_list[1:-1] = ', zip_list[1:-1])
# # zip_list[1:3] =  [2, 3, 4, 5, 6]
# # list[a:b]: the first number means the index and 
# the second number means the place(the 6th element)
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


def seq_data_iter_random(corpus, batch_size, num_steps):
    """Generate a minibatch of subsequences using random sampling."""
    corpus = corpus[random.randint(0, num_steps - 1):]
    # both included
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


# print('list(range(0, 4, 4)) = ', list(range(0, 4, 4)))
# # list(range(0, 4, 4)) =  [0]
# my_seq = list(range(35))
# for x, y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
#     print('x = ', x, 'y = ', y)
# # num_subsequences =  6
# # initial_indices =  [0, 5, 10, 15, 20, 25]
# # initial_indices_per_batch =  [20, 0]
# # x =  [[20, 21, 22, 23, 24], [0, 1, 2, 3, 4]]
# # y =  [[21, 22, 23, 24, 25], [1, 2, 3, 4, 5]]
# # x =  tensor([[20, 21, 22, 23, 24],
# #         [ 0,  1,  2,  3,  4]]) 
# # y =  tensor([[21, 22, 23, 24, 25],
# #         [ 1,  2,  3,  4,  5]])
# # initial_indices_per_batch =  [10, 25]
# # x =  [[10, 11, 12, 13, 14], [25, 26, 27, 28, 29]]
# # y =  [[11, 12, 13, 14, 15], [26, 27, 28, 29, 30]]
# # x =  tensor([[10, 11, 12, 13, 14],
# #         [25, 26, 27, 28, 29]]) 
# # y =  tensor([[11, 12, 13, 14, 15],
# #         [26, 27, 28, 29, 30]])
# # initial_indices_per_batch =  [15, 5]
# # x =  [[15, 16, 17, 18, 19], [5, 6, 7, 8, 9]]
# # y =  [[16, 17, 18, 19, 20], [6, 7, 8, 9, 10]]
# # x =  tensor([[15, 16, 17, 18, 19],
# #         [ 5,  6,  7,  8,  9]]) 
# # y =  tensor([[16, 17, 18, 19, 20],
# #         [ 6,  7,  8,  9, 10]])


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


# my_seq = list(range(35))
# for x, y in seq_data__iter_sequential(my_seq, batch_size=2, num_steps=5):
#     print('x = ', x, '\ny = ', y)
# # offset =  2
# # num_tokens =  32
# # xs =  tensor([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
# #         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33])
# # ys =  tensor([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
# #         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])
# # xs =  tensor([[ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17],
# #         [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]])
# # ys =  tensor([[ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
# #         [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]])
# # x =  tensor([[ 2,  3,  4,  5,  6],
# #         [18, 19, 20, 21, 22]]) 
# # y =  tensor([[ 3,  4,  5,  6,  7],
# #         [19, 20, 21, 22, 23]])
# # x =  tensor([[ 7,  8,  9, 10, 11],
# #         [23, 24, 25, 26, 27]]) 
# # y =  tensor([[ 8,  9, 10, 11, 12],
# #         [24, 25, 26, 27, 28]])
# # x =  tensor([[12, 13, 14, 15, 16],
# #         [28, 29, 30, 31, 32]]) 
# # y =  tensor([[13, 14, 15, 16, 17],
# #         [29, 30, 31, 32, 33]])

class SeqDataLoader:  #@save
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
