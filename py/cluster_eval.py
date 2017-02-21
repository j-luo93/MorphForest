from __future__ import division

import codecs
import cPickle
import sys
import argparse
from utils import UnionFind
from collections import defaultdict
from itertools import combinations
import random
import numpy as np

################################################################################
def gen_all_pairs(S):
    res = set()
    for S_i in S:
        assert type(S_i) == set
        for x, y in combinations(S_i, r=2):
            res.add((x, y))
    return res
################################################################################

random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--stem-file', '-S', metavar='', dest='stem_file', help='stem file')
parser.add_argument('--gold-file', '-G', metavar='', dest='gold_file', help='gold file')
#parser.add_argument('--output-file', '-O', metavar='', dest='output_file', help='output file')
args = parser.parse_args()


clusters_pred = dict()
with codecs.open(args.stem_file, 'r', 'utf8') as fin:
    for line in fin:
        parts = line.strip().split('\t')
        clusters_pred[parts[0]] = parts[1]
directory = UnionFind()
for k, v in clusters_pred.iteritems():
    root = directory[k]
    directory.union(root, v)
clusters = defaultdict(set)
for w in directory:
    root = directory[w]
    clusters[root].add(w)
clusters_pred = clusters.values()

#r = directory['good']
#print clusters[r]
#1/0
# clustered = dict()
# for k, v in clusters_pred.iteritems():
#     if v in clustered:
#         clustered[v].add(k)
#     else:
#         clustered[v] = set([k, v])
# clusters_pred = clustered.values()
clusters_gold = cPickle.load(open(args.gold_file, 'r'))
vocab = set()
for cl in clusters_gold:
    vocab.update(cl)
for cl in clusters_pred:
    cl &= vocab
vocab_pred = set()
for cl in clusters_pred:
    vocab_pred.update(cl)
# import pdb; pdb.set_trace()

assert vocab == vocab_pred

purity = 0.0
cnt = 0
inter = np.zeros((len(clusters_pred), len(clusters_gold)), dtype='int')
#inter = [[0] * len(clusters_gold) for _ in xrange(len(clusters_pred))]
# set up directory
print 'setting up directory...'
dir_g = dict()
for i, cl in enumerate(clusters_gold):
    for w in cl:
        dir_g[w] = i
# compute confusion table
print 'computing confusion table...'
for i, cl_p in enumerate(clusters_pred):
    for w in cl_p:
        j = dir_g[w]
        inter[i,j] += 1

print 'done'
purity = np.mean(np.max(inter, axis=1) / np.asarray(map(len, clusters_pred)))

        #best = max(best, inter[i,j])
   # purity += best / len(clusters_pred[i])
#for i, cl in enumerate(clusters_pred):
#    if not cl: continue
#    cnt += 1
#    print '\r%d' %i,
#    sys.stdout.flush()
#    best = 0
#    for j, cl_g in enumerate(clusters_gold):
#        t = len(cl & cl_g)
#        inter[(i, j)] = t
#        best = max(best, t)
#    purity += best / len(cl)
#purity /= cnt
print 'purity:', purity

x_len = np.asarray(map(len, clusters_pred))
y_len = np.asarray(map(len, clusters_gold))
C = (inter ** 2 / y_len[np.newaxis, :]).sum()
T = ((x_len[:, np.newaxis] - inter) * inter / y_len[np.newaxis, :]).sum()
D = ((y_len[np.newaxis, :] - inter) * inter / y_len[np.newaxis, :]).sum()
print C, T, D
p = C / (C + T)
r = C / (C + D)
f = 2 * p * r / (p + r)
print p, r, f
#    print('Initialization %d' %(i + 1))
#    indices = range(len(clusters_pred))    
#    taken = set()
#    indices_g = range(len(clusters_gold))    
#    purity_one2one = 0.0
#    random.shuffle(indices)
#    random.shuffle(indices_g)
#    cnt = 0
#    for j in indices:
#        cl = clusters_pred[j]
#        if not cl: continue
#        cnt += 1
#        print '\r%d' %cnt,
#        sys.stdout.flush()
#        best = 0
#        best_g = None
#        if len(taken) == len(clusters_gold): break
#        for k in indices_g:
#            if k not in taken:
#                cl_g = clusters_gold[k]
#                t = inter[(j, k)]
#                #t = len(cl & cl_g)
#                if t > best:
#                    best_g = k
#                    best = t
#                    
#                #best = max(best, inter[(j, k)])
#        if best > 0: taken.add(best_g)
#        purity_one2one += best / len(cl)
#    purity_one2one /= cnt
#    print 'purity 1 to 1:', purity_one2one
#    print('################################################################################')
P = gen_all_pairs(clusters_pred)
Q = gen_all_pairs(clusters_gold)
a = len(P & Q)
b = len(P - Q)
c = len(Q - P)
p = a / (a + b)
r = a / (a + c)
f = 2 * p * r / (p + r)
print p, r, f
