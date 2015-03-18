# Cameron Little
# CSCI 404 Winter 2015
#
# N-Gram Models

from __future__ import print_function
import random
import math


class Token(str):
    val = ''

    def __init__(self, val):
        self.val = val
        super(Token, self).__init__()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<{}>".format(self.val)


START = Token('start')
END = Token('end')


class Tokenizer(object):
    def __init__(self, lines):
        if type(lines) == str:
            lines = lines.lower().splitlines()
        words = [[w for w in [START] + s.split() + [END]] for s in lines]
        self.tokens = [y for x in words for y in x]


class NGramModel(object):
    def __init__(self, tokens, max_n=3):
        self.tokens = tokens
        self.counts = {}
        self._vocab_sizes = {}
        self.max_n = max_n

        self.process()

    @property
    def corpus_size(self):
        return len(self.tokens)

    @property
    def unique_tokens(self):
        return list(set(self.tokens))

    def random_token(self):
        w = random.choice(self.unique_tokens)
        while w in [START, END]:
            w = random.choice(self.tokens)
        return w

    def random_token_weighted(self):
        """
        Returns a random word from the model, weighing more frequent ones
        higher. Excludes END and START. Assumes tokens exist outside of START
        and END.
        """
        w = random.choice(self.tokens)
        while w in [START, END]:
            w = random.choice(self.tokens)
        return w

    _vocab_sizes = {}

    def vocab_size(self, n):
        if n in self._vocab_sizes:
            return self._vocab_sizes[n]

        if n == 1:
            self._vocab_sizes[n] = len(set(self.tokens))
        else:
            keys = set()
            for i, token in enumerate(self.tokens):
                key = self.gen_key(*self.tokens[i-n:i])
                keys.add(key)

            self._vocab_sizes[n] = len(keys)

        return self.vocab_size(n)

    def process(self):
        for n in range(1, self.max_n+1):
            for i, token in enumerate(self.tokens):
                key = self.gen_key(*self.tokens[i-n:i])
                if key in self.counts:
                    self.counts[key] += 1
                else:
                    self.counts[key] = 1

    @staticmethod
    def gen_key(*tokens):
        # using ;; as separator because tokens have been stripped of punctuation
        return ";;".join(tokens)

    def count(self, *tokens):
        key = self.gen_key(*tokens)
        if key in self.counts:
            return self.counts[key]
        else:
            return 0

    def probability(self, *tokens):
        """
        return probability of bigram <token1 token2> using laplace smoothing
        """
        n = len(tokens)
        if n > self.max_n:
            raise Exception("too many tokens: max {}, recieved {}".format(self.max_n, n))
        elif n == 1:
            return float(self.count(*tokens) + 1) / \
                float(self.corpus_size + self.vocab_size(n-1))
        else:
            return float(self.count(*tokens) + 1) / \
                float(self.count(*tokens[0:-1]) + self.vocab_size(n-1))

    def sequence_probability(self, sequence, n, add_start_end=True):
        """
        Returns probability of a sequence of tokens in log space
        """
        if type(sequence) == str:
            tokens = sequence.split()
        elif type(sequence) == list or type(sequence) == tuple:
            tokens = list(sequence)
        else:
            raise Exception("sequence has wrong type")
        if add_start_end:
            tokens = [START] + tokens + [END]
        prob = 0
        for i in range(n-1, len(tokens)):
            prob += math.log(self.probability(*tokens[i-(n-1):i]))
        if prob == 0:
            return float('inf')
        return prob

    def next_token(self, *start_tokens):
        if len(start_tokens) > self.max_n - 1:
            raise Exception("too many tokens: max {}, recieved {}".format(self.max_n-1, len(start_tokens)))
        most_likely = (0, "")
        for token in self.tokens:
            new_tokens = list(start_tokens) + [token]
            p = self.probability(*new_tokens)
            if p > most_likely[0]:
                most_likely = (p, token)

        return most_likely

    def missing_word(self, prev_tokens, following_tokens, n):
        if len(prev_tokens) > n-1:
            raise Exception("to many prev tokens")
        if len(following_tokens) > n-1:
            raise Exception("to many next tokens")
        prev_tokens = list(prev_tokens)
        following_tokens = list(following_tokens)

        most_likely = (0, "")
        for token in self.tokens:
            p = 0
            sequence = prev_tokens + [token] + following_tokens
            for i in range(0, len(sequence) - n):
                p += self.probability(*sequence[i:i+n])
            if p > most_likely[0]:
                most_likely = (p, token)

        return most_likely


def classify(sequence, models, label_probs, n):
    """
    :param sequence: a string to test
    :param models: a dictionary of labels to ngram models
    :param n: n of ngram to use
    :return: best label
    """
    sequence = sequence.strip()
    sequence = Tokenizer(' '.join(sequence.splitlines())).tokens

    maxi = float('inf')
    best = None
    for model in models:
        p = models[model].sequence_probability(sequence, n, False)
        p = p * (1 - label_probs[model])

        if p < maxi:
            maxi = p
            best = model
    return best
