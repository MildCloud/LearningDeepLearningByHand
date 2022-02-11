import collections
import re


# print([i for i in range(5)])
# # [0, 1, 2, 3, 4]


def read_time_machine():
    """Load the time machine dataset into a list of text line"""
    with open('time_machine', 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    # [^A-Za-z] means regular expression, sub represents substitution, strip(): remove /n
    # The output text only contains 26 letters in lower case and space


lines = read_time_machine()
# print(f'# text lines: {len(lines)}')
# print(lines[0])
# print(lines[10])
# the time machine by h g wells
# twinkled and his usually pale face was flushed and animated the


def tokensize(f_lines, f_token = 'word'):
    if f_token == 'word':
        return [line.split() for line in f_lines]
        # split every word
    elif f_token == 'char':
        return [list(line) for line in f_lines]
        # split every character
    else: 
        print('error ' + f_token)


# print(list('home'))
# # ['h', 'o', 'm', 'e']
tokens = tokensize(lines, 'word')
# for i in range(1):
#     print(tokens[i])
# # ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
# # ['t', 'h', 'e', ' ', 't', 'i', 'm', 'e', ' ', 'm', 'a', 'c', 'h', 'i', 'n', 'e', ' ', 'b', 'y', ' ', 'h', ' ', 'g', ' ', 'w', 'e', 'l', 'l', 's']

class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The counter is a dict, and item() will return a dict_items([]) of tuple
        # The first element of the tuple is a word, and the second one is its frequency
        # The index for the unknown token is 0
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
            # if tokens is a list or a tuple
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def count_corpus(f_tokens):
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(f_tokens) == 0 or isinstance(f_tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        f_tokens = [token for line in f_tokens for token in line]
    return collections.Counter(f_tokens)


# test_tokens = [[1, 2], [1, 2, 3], [1, 2, 3, 4]]
# # test_tokens = [1, 2, 3, 4] one dimension list will cause error
# test_tokens = [token for line in test_tokens for token in line]
# print("test_tokens = ", test_tokens)
# # test_tokens =  [1, 2, 1, 2, 3, 1, 2, 3, 4]
# print("count_corpus(test_tokens) = ", count_corpus(test_tokens))
# # count_corpus(test_tokens) =  Counter({1: 3, 2: 3, 3: 2, 4: 1})
# print(sorted([3, 2, 1, 3, 4]))
# # [1, 2, 3, 3, 4]

vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
for i in [0, 10]:
    print('words: ', tokens[i])
    print('indices: ', vocab[tokens[i]])


def load_corpus_time_machine(max_tokens = -1):
    lines = read_time_machine()
    tokens = tokensize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[: max_tokens]
    return corpus, vocab


corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))
