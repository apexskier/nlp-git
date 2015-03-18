# -*- coding: utf-8 -*-
# CSCI 404 Project 1
# Cameron Little
#
# A tokenizer that produces statistics about a corpus's words, word types,
# sentences, and bigrams.
#
# The biggest current issue is that there is no determinationbetween posessive
# and contractitive "'s"'s

import copy
import re
import sys


# the following regex doesn't test for apostrophes
PUNCTUATION_RE = re.compile("[.,;:\-\(\)\[\]\"\“\”\*\!?\/]")


try:
    with open('stopwords.txt', 'r') as f:
        STOPWORDS = [w.strip() for w in f.readlines()]
        STOPWORDS += ["'s", "'t"]
except IOError:
    print ("couldn't find stopwords.txt")
    sys.exit(1)


class IDGenerator(object):
    num = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        self.num += 1
        return self.num


class Token(object):
    def __init__(self, token, raw):
        self.raw = raw
        self.token = token

    @property
    def is_sentence(self):
        return self.raw[-1] in [".", "!", "?"]

    @property
    def contains_digits(self):
        return not not re.search("[0-9]", self.token)

    @property
    def raw_contains_punctuation(self):
        return not not re.search(PUNCTUATION_RE, self.raw)

    @property
    def contains_punctuation(self):
        return not not re.search(PUNCTUATION_RE, self.token)

    @property
    def contains_mix(self):
        return self.contains_digits and not not re.search("[A-Za-z]", self.token)

    @property
    def lex_type(self):
        return self.token


class Tokenizer(object):
    def __init__(self, text):
        def search_apostrophe(tok):
            count = len(re.findall("'", tok))
            if count == 0:
                return False
            elif count > 1:
                return True
            else:  # count == 1
                # assuming already processed (it's the tail of a contraction)
                if "'" == tok[0]:
                    return False
                else:
                    return True

        # convert everything to lowercase
        # initially, just split on whitespace, that's a given.
        tokens = [[t, copy.deepcopy(t)] for t in text.lower().split()]
        # initially storing as (token, original_raw_token)

        """
        Originally I used a pop() and append method, and just processed
        whatever token found. I'm doing the following type of loop instead to
        maintain order.
        """
        pointer = 0
        while True:
            try:
                token = str(tokens[pointer][0])
            except IndexError:
                break

            # remove blank tokens
            if not token:
                del tokens[pointer]
            # strip preceding punctuation
            elif PUNCTUATION_RE.search(token[0]):
                tokens[pointer][0] = token[1:]
            # strip following punctuation
            elif PUNCTUATION_RE.search(token[-1]):
                tokens[pointer][0] = token[:-1]
            # seperate -- words
            elif re.search('--', token):
                tsplit = token.split('--')
                tokens[pointer][0] = tsplit.pop(0)
                # TODO: verify that the raw form is correct for the following tokens
                tokens[pointer+1:pointer+1] = [[s, s] for s in tsplit]
            # remove apostrophe quotes around word
            elif token[0] == "'" and token[-1] == "'":
                tokens[pointer][0] = token[1:-1]
            # possessive words
            elif token[-1] == "'":
                tokens[pointer][0] = token[:-1]
            # seperate contractions
            elif search_apostrophe(token):
                tsplit = token.split("'")
                tokens[pointer][0] = tsplit.pop(0)
                tokens[pointer+1:pointer+1] = [["'" + s, "'" + s] for s in tsplit]
            # if no processing required, go move to the next word
            else:
                pointer += 1

        self.tokens = tokens


class Corpus(object):
    tokens = []
    types = set([])  # a type is a string

    def __init__(self, tokens):
        for t in tokens:
            self.tokens.append(Token(*t))
            self.types.add(t[0])

    @property
    def words(self):
        return [t.token for t in self.tokens]

    @property
    def sentences(self):
        sents = []
        sent = []
        for token in self.tokens:
            sent.append(token.token)
            if token.is_sentence:
                sents.append(sent)
                sent = []
        return sents

    @property
    def with_digits(self):
        return [t.token for t in self.tokens if t.contains_digits]

    @property
    def with_punctuation(self):
        return [t.token for t in self.tokens if t.contains_punctuation]

    @property
    def with_mix(self):
        return [t.token for t in self.tokens if t.contains_mix]

    @property
    def _type_frequency(self):
        """
        returns: dictionary in the format: {"word_type": number_occurences}
        """

        types = {}
        for token in self.tokens:
            if token.token in types:
                types[token.token] += 1
            else:
                types[token.token] = 1

        return types

    @property
    def type_frequency_list(self):
        return [(k, v) for k, v, in self._type_frequency.iteritems()]

    @property
    def singleton_percentage(self):
        return len([True for (_, v) in self._type_frequency.iteritems() if v == 1]) / float(len(self.types))

    def type_frequency(self, count=100, punctuation=True, digits=True, letters=True, stopwords=True):
        """
        parameters:
          count: number of results to return
          args*: (the rest) whether something is allowed
        """
        types = {}
        for typ, count in sorted([(t, c) for t, c in self.type_frequency_list if
                                      (stopwords or t not in STOPWORDS) and
                                      (punctuation or not re.search(PUNCTUATION_RE, t)) and
                                      (punctuation or "'" not in t) and
                                      (digits or not re.search("[0-9]", t)) and
                                      (letters or not re.search("[A-Za-z]", t))],
                                 reverse=True,
                                 key=lambda token: token[1])[0:count]:
            types[typ] = count

        return types

    @property
    def _consecutive_types(self):
        bigrams = {}
        for i, token in enumerate(self.tokens[:-1]):
            key = "{} {}".format(token.token, self.tokens[i+1].token)

            if key in bigrams:
                bigrams[key] += 1
            else:
                bigrams[key] = 1

        return bigrams

    @property
    def bigram_frequency_list(self):
        return [(k, v) for k, v, in self._consecutive_types.iteritems()]

    def bigram_frequency(self, count=100, stopwords=False):
        bigrams = {}
        for typ, count in sorted([(t, c) for t, c in self.bigram_frequency_list if
                (stopwords or len([w for w in t.split() if w not in STOPWORDS]) == 2)
                ], reverse=True, key=lambda token: token[1])[0:count]:
            bigrams[typ] = count

        return bigrams
