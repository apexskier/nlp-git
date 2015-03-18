# NLP Git Fix

A set of tools to identify authors of git commits based on their commit
messages using natural language processing.

Developed as classwork, unfinished, and abandoned.

Requires NLTK.

Usage:

```
python main.py -h
usage: main.py [-h] [--naive-bayes] [--ngrams] [--nltk-naive-bayes]
               [--top-committer] [--all] [--match]
               repo

positional arguments:
  repo                a path to a git repository to scan

optional arguments:
  -h, --help          show this help message and exit
  --naive-bayes
  --ngrams
  --nltk-naive-bayes
  --top-committer
  --all
  --match
```
