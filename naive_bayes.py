# Cameron Little
# CSCI 404 Winter 2015
#
# Text Categorization using Naive Bayes

from __future__ import print_function
import os
import math


class Vocab(object):
    wordcounts = {}
    dictSize = 2500

    def __init__(self, tokens):
        self.wordcounts = {}

        for word in tokens:
            if word in self.wordcounts:
                self.wordcounts[word] += 1
            else:
                self.wordcounts[word] = 1

    def top_words(self, n):
        count = 0
        counts = {}
        for word in sorted(self.wordcounts, key=lambda w: self.wordcounts[w], reverse=True):
            if count > n - 1:
                return counts

            counts[word] = self.wordcounts[word]
            count += 1

        return counts

    @property
    def dictionary(self):
        ws = self.top_words(self.dictSize)
        return sorted(ws, key=lambda w: ws[w], reverse=True)


class FeatureVector(object):
    filename = ""
    dictionary = None

    dictionaryCounts = {}
    tokens = set()

    def __init__(self, raw_tokens, dictionary):
        self.raw_tokens = raw_tokens
        self.dictionary = dictionary
        self.dictionaryCounts = {}
        self.tokens = set()

        for word in self.raw_tokens:
            if word in self.dictionary:
                self.tokens.add(word)
                if word in self.dictionaryCounts:
                    self.dictionaryCounts[word] += 1
                else:
                    self.dictionaryCounts[word] = 1

    def word_appears(self, word):
        if word not in self.dictionary:
            return 0
            #raise Exception("word not in dictionary")
        return word in self.dictionaryCounts

    def word_count(self, word):
        if word not in self.dictionary:
            return 0
            #raise Exception("word not in dictionary")
        if word in self.dictionaryCounts:
            return self.dictionaryCounts[word]
        else:
            return 0


class FeatureSet(object):
    def __init__(self, document_tokens, vocabulary):
        self.document_tokens = document_tokens
        self.vocabulary = vocabulary

        self.documents = []
        self.wordCounts = {}

        for doc in document_tokens:
            self.documents.append(FeatureVector(doc, self.vocabulary.dictionary))

    def word_count(self, word):
        if word in self.wordCounts:
            return self.wordCounts[word]

        t = 0
        for document in self.documents:
            t += document.word_count(word)

        self.wordCounts[word] = t
        return t


class NaiveBayes(object):
    def __init__(self, labels, categories, vocabulary):
        if len(categories) != len(labels):
            raise Exception("length of categories and labels don't match")

        self.categories = categories
        self.labels = labels
        self.vocabulary = vocabulary
        self.wordprobs = {}
        self.labelprobs = {}

        self.featureSets = {}
        print("generating naivebayes")
        increment = (len(self.labels) / 72) + 1
        for i, label in enumerate(labels):
            if i % increment == 0:
                print('.', end='')
            self.featureSets[label] = FeatureSet(categories[i], vocabulary)
            self.wordprobs[label] = {}
        print()

    def p_word_label(self, word, label):
        """
        Log space, Laplace smoothing, memoization
        """
        if word in self.wordprobs[label]:
            return self.wordprobs[label][word]

        fs = self.featureSets[label]
        num = fs.word_count(word) + 1
        den = 0
        for word in self.vocabulary.dictionary:
            den += fs.word_count(word)
        den += self.vocabulary.dictSize

        p = math.log(num / float(den))
        self.wordprobs[label][word] = p
        return p

    @property
    def num_documents(self):
        t = 0
        for label in self.labels:
            t += len(self.featureSets[label].documents)
        return t

    def p_label(self, label):
        if label in self.labelprobs:
            return self.labelprobs[label]

        p = math.log(len(self.featureSets[label].documents) / float(self.num_documents))
        self.labelprobs[label] = p
        return p

    def test(self, document):
        fv = FeatureVector(document, self.vocabulary.dictionary)
        maxL = (float('-inf'), None)

        for label in self.labels:
            pr = self.p_label(label)
            for token in list(fv.tokens):
                pr += self.p_word_label(token, label)

            if pr > maxL[0]:
                maxL = (pr, label)

        return maxL[1]


if __name__ == "__main__":
    v = Vocab("data/q2")

    """
    # Top 250 words in vocabulary
    counts = v.top_words(2500)
    for word in sorted(counts, key=lambda word: counts[word], reverse=True):
        print("{} {}".format(word, counts[word]))
    """

    """
    hfs = FeatureSet("data/q2/nonspam-train", v)
    sfs = FeatureSet("data/q2/spam-train", v)

    # feature file
    for document in hfs.documents + sfs.documents:
        for word in v.dictionary:
            print("{} {} {}".format(os.path.basename(document.filename), word, document.word_count(word)))
    """

    trainedSet = NaiveBayes(["spam", "ham"], ["data/q2/spam-train", "data/q2/nonspam-train"], v)
    print("trained")
    print()

    correctSpam = 0
    falseSpam = 0
    correctHam = 0
    falseHam = 0

    # testing on spam files
    for root, dirs, files in os.walk("data/q2/spam-test"):
        for file_ in files:
            fn = os.path.join(root, file_)
            result = trainedSet.test(fn)
            if result == "spam":
                correctSpam += 1
            else:
                falseSpam += 1
            print("{} is {}".format(fn, result))

    # testing on nonspam files
    for root, dirs, files in os.walk("data/q2/nonspam-test"):
        for file_ in files:
            fn = os.path.join(root, file_)
            result = trainedSet.test(fn)
            if result == "ham":
                correctHam += 1
            else:
                falseHam += 1
            print("{} is {}".format(fn, result))

    print()

    print("                labeled spam   labeled ham")
    print("              +--------------+-------------+")
    print("actually spam | {:^12} | {:^11} |".format(correctSpam, falseHam))
    print("              +--------------+-------------+")
    print("actually ham  | {:^12} | {:^11} |".format(falseSpam, correctHam))
    print("              +--------------+-------------+")

    print()
    precision = correctSpam / float(correctSpam + falseSpam)
    recall = correctSpam / float(correctSpam + falseHam)
    f_measure = 2 * ((precision * recall) / (precision + recall))
    print("spam detection precision: {:.02f}%".format(precision * 100))
    print("spam detection recall: {:.02f}%".format(recall * 100))
    print("spam detection balanced f-measure: {:.02f}%".format(f_measure * 100))
