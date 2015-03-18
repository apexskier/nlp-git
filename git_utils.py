#!/bin/python
#
# Cameron Little
# CSCI 404 Winter 2015
#
# Final Project

from __future__ import print_function
import os
import re
from subprocess import Popen, PIPE


AUTHOR_RE = re.compile('^Author: (?P<name>.*)\s<(?P<email>.*)>$')
DATE_RE = re.compile('^Date:\s+(?P<date>.*)$')


class Commit(object):
    def __init__(self, rawtext=None):
        self.author = None
        self.date = None
        self.message = ""
        self.raw = rawtext

        if rawtext is None:
            return

        meta = []
        lines = rawtext.splitlines()
        line = lines.pop(0).strip()
        while line:
            meta.append(line)
            line = lines.pop(0).strip()

        self.sha = meta.pop(0).split()[1]
        for line in meta:
            adetails = re.match(AUTHOR_RE, line)
            ddetails = re.match(DATE_RE, line)
            if adetails:
                self.author = {
                    'name': adetails.group('name'),
                    'email': adetails.group('email')
                }
            elif ddetails:
                self.date = ddetails.group('date')  #datetime.strptime(ddetails.group('date'), '%a %b %d %H:%M:%S %Y %z')

        self.message = "\n".join([line.strip() for line in lines])

        self.fix_email()

    def fix_email(self):
        if self.author['email'] == "mattebailey@Mattes-MacBook-Pro.local":
            self.author['email'] = "MatteABailey@gmail.com"
        elif self.author['email'] == "cameron@cloudcitylabs.co":
            self.author['email'] = "cameron@camlittle.com"
        elif self.author['email'] == "erdillon@users.noreply.github.com":
            self.author['email'] = "eric@cloudcitylabs.co"


class Repo(object):
    def __init__(self, repopath):
        self.origdir = repopath
        p = Popen("git rev-parse --show-cdup".split(), stdout=PIPE, cwd=repopath)
        self.topdir = os.path.join(os.getcwd(), repopath, p.stdout.read().strip())

    def log(self):
        process = Popen("git log".split(), stdout=PIPE, cwd=self.topdir)
        currcommit = ""

        commits = []

        output = process.stdout.readlines()

        for line in output:
            if line.startswith("commit"):
                if currcommit:
                    commits.append(Commit(currcommit))
                currcommit = ""

            currcommit += line

        return commits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="a path to a git repository to scan")
    args = parser.parse_args()

    repo = Repo(args.repo)

    for commit in repo.log():
        print(commit.author)
