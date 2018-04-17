#!/usr/bin/env python
# Author: Thamme Gowda tg@isi.edu
# Crated on: October 30, 2017
"""
Morfessor utility for morphological segmentation of bitext data.


Note:
 1. Training can be done using `morfessor-train` command line tool.
 Example::
    morfessor-train -s morf-model.src.pkl 1A-v1-train.src
 2. Use only the training data split for the morfessor training
"""
from __future__ import print_function
import traceback
import sys
import cPickle
import os
from os import path
import codecs

src_dir = path.join(path.dirname(path.realpath(__file__)), 'src')
sys.path.insert(0, src_dir)

from model import MC
from ILP import ILP as ILP
import logging as log
log.basicConfig(level=log.DEBUG)

def load_model(model_path, data_path, lang):
    log.debug("Loading model from %s" % model_path)
    wvec_path = path.join(data_path, 'wv.%s' %  lang)
    wlist_path = path.join(data_path, 'wordlist.%s' %  lang)
    model = cPickle.load(open(model_path, 'r'))
    if isinstance(model, ILP):
        model = model.base
    log.info('reading word vectors from %s' % wvec_path)
    model.read_word_vectors(wvec_path) # word vectors are not saved with the model
    log.info('reading word list from %s' % wlist_path)
    model.read_wordlist(wlist_path) # word vectors are not saved with the model
    return model

def main(model, data_dir, lang, fin, fout, **args):
    if (sys.stdout.encoding is None):
        print("please `export PYTHONIOENCODING=UTF-8`", file=sys.stderr)
        exit(1)

    model = load_model(model, data_dir, lang)
    words = (line.strip() for line in fin)

    for token, segs in model.segment_all(words):
        segs = segs.split()
        # first pass to identify intervals of consecutive hyphens
        intervals = list()
        s = None
        t = None
        for i, seg in enumerate(segs):
            if seg == '-':
                if s is None:
                    s = i
                t = i + 1
            else:
                if s is not None:
                    intervals.append((s, t))
                s = None
        if s is not None:
            intervals.append((s, t))

        # combine consecutive hyphens
        i = len(intervals) - 1
        while i >= 0:
            s, t = intervals[i]
            if t - s > 1 or s == 0:
                segs = segs[:s] + ['-' * (t - s)] + segs[t:]
            else:
                segs = segs[:s] + segs[t:]
            i -= 1
        segs = ' '.join(segs)
        out_line = '%s\t%s\n' % (token, segs)
        fout.write(out_line)


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser("""Morfessor.""")
    parser.add_argument('lang', help='language')
    parser.add_argument('model', help='Segmenter model, one per text column (for TSV)')
    parser.add_argument('-in', '--input', dest='fin', help='Input file. Default=STDIN', default=sys.stdin)
    parser.add_argument('-out', '--output', dest='fout', help='Output file. DEFAULT=STDOUT', default=sys.stdout)
    parser.add_argument('--data_dir', '-dd', help='data directory', default='data/')
    args = vars(parser.parse_args())

    for f, mode in [('fin', 'r'), ('fout', 'w')]:
        if type(args[f]) is str:
            args[f] = codecs.open(args[f], mode, 'UTF-8')
    main(**args)
