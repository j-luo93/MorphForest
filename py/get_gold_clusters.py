#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cPickle
import codecs
from collections import defaultdict, Counter
from utils import  UnionFind
import re
import argparse

def standardize(word):
    output = ''
    for c in word:
        if c == u'ü' or c == u'Ü': output += u'ue'
        elif c == u'ö' or c == u'Ö': output += u'oe'
        elif c == u'ä' or c == u'Ä': output += u'ae'
        elif c == u'ß': output += u'ss'
        else: output += c.lower()
    return output


parser = argparse.ArgumentParser()
parser.add_argument('lang', help='language')
args = parser.parse_args()

# i2w = dict()
# get roots
roots = defaultdict(set)
roots_surface = defaultdict(set)
cnt = Counter()
if args.lang == 'eng':
    with codecs.open('../data/goldstd.tmp', 'r', 'utf8') as fin:
        for line in fin:
            segs = line.split()
            assert segs[0].isdigit() and segs[2].isdigit()
            word = segs[1].lower()
            word_id = int(segs[0])
            # i2w[word_id] = word
            parts = ' '.join(segs[3:]).split(",")
            for part in parts:
                groups = re.findall(r'[^:\|\s~]+:[^:\|_]+\|[\S]*', part)
                stem = ''
                for g in groups:
                    code = g.split('|')[-1]
                    if code == 'p' or code == 's': continue
                    assert len(g.split(':')) == 2
                    assert len(g.split(':')[1].split('|')) == 2
                    stem += g.split(':')[1].split('|')[0]
                    #stem += g.split(':')[0]
                if stem:
                    roots[word].add(stem.lower())
elif args.lang == 'ger':
    with codecs.open('../data/gml.cd', 'r', 'utf8') as gml, codecs.open('../data/gmw.cd', 'r', 'utf8') as gmw:
        i2w = dict()
        for line in gml:
            parts = line.split('\\')
            segs = [parts[13]]
            word = standardize(parts[1])
            idx = int(parts[0])
            i2w[idx] = word
            if len(parts) == 35: segs.append(parts[28])
            if len(parts) == 50: segs.append(parts[43])
            for seg in segs:
                stems = re.findall(r'\([^,\(\)]+\)\[[^\|,\[\]]+\]', seg)
                root = ''
                for stem in stems:
                    s = re.findall(r'(?<=\()[^\(\)]+(?=\))', stem)
                    assert len(s) == 1
                    s = s.pop()
                    root += s
                if not root:
                    root = word
                root = root.lower()
                roots[word].add(root)
        for line in gmw:
            parts = line.split('\\')
            root_idx = int(parts[3])
            word = standardize(parts[1])
            roots[word].add(i2w[root_idx])
else: raise NotImplementedError
# merge clusters
directory = UnionFind()
for k, s in roots.iteritems():
    cl = directory[k]
    directory.union(cl, *s)
vocab = set(directory.parents.keys())
print 'vocab size:', len(vocab)

clusters = defaultdict(set)
for w in directory:
    root = directory[w]
    clusters[root].add(w)

all_sets = clusters.values()
# sanity check
vocab_set = set()
for s in all_sets: vocab_set.update(s)
assert vocab_set == vocab

with codecs.open('../data/vocab_celex.%s' %args.lang, 'w', 'utf8') as fout:
    for w in vocab:
        fout.write('%s\n' %w)
# directory = dict()
# for k, s in roots.iteritems():
#     seen = set()
#     not_seen = set()
#     for stem in s:
#         if stem in directory: seen.add(stem)
#         else: not_seen.add(stem)
# #     if  'act' in s or 'acting' in s:import pdb; pdb.set_trace()
#     if seen:
#         s_chosen = directory[seen.pop()]
#         s_chosen.add(k)
#         for stem in seen:
#             s_chosen.update(directory[stem])
#             directory[stem] = s_chosen
#     else:
#         s_chosen = set([k])
#     for stem in not_seen:
#         s_chosen.add(stem)
#         directory[stem] = s_chosen
with codecs.open('../data/gold_clusters_dict.%s' %args.lang, 'w', 'utf8') as fout:
    for k, s in roots.iteritems():
        r = None
        for x in s:
            if not r or len(x) < len(r):
                r = x
        fout.write('%s\t%s\n' %(k, r))

cPickle.dump(all_sets, open('../data/gold_clusters.%s' %args.lang, 'w'))
