#!/usr/bin/env python
# -*- coding: utf-8 -*-


import gensim, logging, sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Sentences(object):

    def __init__(self, f_name, lang):
        self.f_name = f_name
        self.lang = lang

    def __iter__(self):
        with open(self.f_name, 'r') as fin:
            for line in fin:
                yield line.split()

    def _standardize(self, line):
        if lang == 'ger':
            output = ''
            for c in line:
                if c == u'ü' or c == u'Ü': output += u'ue'
                elif c == u'ö' or c == u'Ö': output += u'oe'
                elif c == u'ä' or c == u'Ä': output += u'ae'
                elif c == u'ß': output += u'ss'
                else: output += c
            return output.lower()   # lowercased
        elif lang == 'fin':
            return line.lower()
        else:
            raise NotImplementedError, lang

f_name = sys.argv[1]
lang = sys.argv[2]
sentences = Sentences(f_name, lang)
model = gensim.models.Word2Vec(sentences, min_count=10, size=200, workers=20)
model.save_word2vec_format(f_name + '.wv')
