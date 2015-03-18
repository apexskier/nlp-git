#!/bin/python
#
# Cameron Little
# CSCI 404 Winter 2015
#
# Final Project

from __future__ import print_function
import os
import pickle
import random

from git_utils import Repo

import naive_bayes
from tokenizer import Tokenizer

import ngrams

from nltk import FreqDist
from nltk.classify import NaiveBayesClassifier


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="a path to a git repository to scan")
    parser.add_argument("--naive-bayes", action="store_true", dest="naive_bayes")
    parser.add_argument("--ngrams", action="store_true", dest="ngrams")
    parser.add_argument("--nltk-naive-bayes", action="store_true", dest="nltk_nb")
    parser.add_argument("--top-committer", action="store_true", dest="top_committer")
    parser.add_argument("--all", action="store_true", dest="allmethods")
    args = parser.parse_args()

    if args.allmethods:
        args.naive_bayes = True
        args.ngrams = True
        args.nltk_nb = True
        args.top_committer = True

    repo = Repo(args.repo)

    pickle_file = repo.topdir.replace('/', '-') + '.pkl'

    if os.path.isfile(pickle_file):
        with open(pickle_file, 'rb') as input_:
            commits = pickle.load(input_)
    else:
        commits = repo.log()
        random.shuffle(commits)

        with open(pickle_file, 'wb') as output:
            pickle.dump(commits, output, pickle.HIGHEST_PROTOCOL)

    nb = None
    ngram_models = None
    ngram_probs = None
    top_committer = None

    # train on 3/4 of the data
    num_train = int((len(commits) / float(4)) * 3)
    train_commits = commits[:num_train]
    bulk_text = " ".join([commit.message for commit in train_commits])
    if args.naive_bayes:
        print("Naive Bayes")
        filename = 'nb_' + pickle_file
        if os.path.isfile(filename):
            print(" loading data")
            with open(filename, 'rb') as input_:
                t, v, nb = pickle.load(input_)
        else:
            print(" tokenizing")
            t = Tokenizer(bulk_text)

            print(" generating vocabulary")
            v = naive_bayes.Vocab([a[0] for a in t.tokens])

            print(" sorting by author")
            user_messages = {}
            unknown = []
            for commit in train_commits:
                if commit.author:
                    if commit.author['email'] not in user_messages:
                        user_messages[commit.author['email']] = []
                    user_messages[commit.author['email']].append([i[0] for i in Tokenizer(commit.message).tokens])
                else:
                    unknown.append(commit)

            print(" training")
            nb = naive_bayes.NaiveBayes(user_messages.keys(), [user_messages[k] for k in user_messages], v)

            with open(filename, 'wb') as output:
                pickle.dump((t, v, nb), output, pickle.HIGHEST_PROTOCOL)

    if args.ngrams:
        print("Trigrams")
        filename = 'tg_' + pickle_file
        if os.path.isfile(filename):
            print(" loading data")
            with open(filename, 'rb') as input_:
                t, ngram_models, ngram_probs = pickle.load(input_)
        else:
            print(" tokenizing")
            t = ngrams.Tokenizer(bulk_text)

            print(" sorting by author")
            user_messages = {}
            unknown = []
            total = 0
            for commit in train_commits:
                if commit.author:
                    if commit.author['email'] not in user_messages:
                        user_messages[commit.author['email']] = []
                    user_messages[commit.author['email']].append(commit.message)
                    total += 1
                else:
                    unknown.append(commit)

            ngram_probs = {}
            ngram_models = {}
            print(" generating models for each user")
            for user in user_messages:
                ngram_probs[user] = len(user_messages[user]) / float(total)
                ngram_models[user] = ngrams.NGramModel(user_messages[user], max_n=3)

            with open(filename, 'wb') as output:
                pickle.dump((t, ngram_models, ngram_probs), output, pickle.HIGHEST_PROTOCOL)

    if args.nltk_nb:
        print("NLTK Naive Bayes")
        filename = 'nltk_nb_' + pickle_file
        #if os.path.isfile(filename):
        #    print(" loading data")
        #    with open(filename, 'rb') as input_:
        #        t, nltk_trained = pickle.load(input_)
        #else:
        print(" tokenizing")
        t = bulk_text.split()

        print(" analyizing data")
        train = []
        for commit in train_commits:
            if commit.author:
                tokens = commit.message.split()
                train.append((dict(FreqDist(tokens).items()), commit.author['email']))
            else:
                unknown.append(commit)

        print(" training")
        nltk_trained = NaiveBayesClassifier.train(train)

        #with open(filename, 'wb') as output:
            #pickle.dump((t, nltk_trained), output, pickle.HIGHEST_PROTOCOL)

    if args.top_committer:
        print("Top Committer")
        authors = [commit.author['email'] for commit in train_commits]
        top_committer = max(set(authors), key=authors.count)

    test_commits = commits[num_train:]

    if args.naive_bayes:
        print("Testing Naive Bayes")
        num_correct = 0
        num_incorrect = 0
        count = 0
        for commit in test_commits:
            if count % 10 == 0:
                print('.', end='')
            label = nb.test([i[0] for i in Tokenizer(commit.message).tokens])
            if label == commit.author['email']:
                num_correct += 1
            else:
                num_incorrect += 1
            count += 1
        print()

        score = (num_correct / float(len(test_commits))) * 100
        print("Naive Bayes: {:.02f}% correct".format(score))

    if args.ngrams:
        print("Testing Ngrams")
        num_correct = 0
        num_incorrect = 0
        count = 0
        for commit in test_commits:
            if count % 10 == 0:
                print('.', end='')
            label = ngrams.classify(commit.message, ngram_models, ngram_probs, 3)
            if label == commit.author['email']:
                num_correct += 1
            else:
                num_incorrect += 1
            count += 1
        print()

        score = (num_correct / float(len(test_commits))) * 100
        print("Trigrams: {:.02f}% correct".format(score))

    if args.nltk_nb:
        print("Testing NLTK Naive Bayes")
        num_correct = 0
        num_incorrect = 0
        count = 0
        for commit in test_commits:
            if count % 10 == 0:
                print('.', end='')
            tokens = commit.message.split()
            data = dict(FreqDist(tokens).items())
            label = nltk_trained.classify(data)
            if label == commit.author['email']:
                num_correct += 1
            else:
                num_incorrect += 1
            count += 1
        print()

        score = (num_correct / float(len(test_commits))) * 100
        print("NLTK Naive Bayes: {:.02f}% correct".format(score))

    if args.top_committer:
        print("Testing Top Committer")
        num_correct = 0
        num_incorrect = 0
        count = 0
        for commit in test_commits:
            if count % 10 == 0:
                print('.', end='')
            if commit.author['email'] == top_committer:
                num_correct += 1
            else:
                num_incorrect += 1
            count += 1
        print()

        score = (num_correct / float(len(test_commits))) * 100
        print("Top Committer: {:.02f}% correct".format(score))
