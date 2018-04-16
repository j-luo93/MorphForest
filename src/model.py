#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import codecs
import math
import random
import traceback
from operator import itemgetter
from collections import Counter, defaultdict
from time import localtime, strftime
from itertools import combinations

import numpy as np
from scipy.linalg import norm
import scipy.sparse as sp
from scipy.spatial.distance import cosine as cos_dist
import theano
import theano.sparse
import theano.tensor as T
from copy import deepcopy

from evaluate import evaluate
from SGD import Adam
from path import Path
from utils import Logger
from pair import ChildParentPair as Pair

class MC(object):

    ############################# Initialization ###############################

    def __init__(self, **kwargs):
        random.seed(kwargs['seed'])
        self.data_path = kwargs['data_path']
        self.out_path = self._get_out_path(kwargs['msg'])

        self.lang = kwargs['lang']
        self.top_affixes = kwargs['top_affixes']
        self.top_words = kwargs['top_words']
        self.word_vector_file = self.data_path + 'wv.%s' %self.lang
        self.gold_segs_file = self.data_path + 'gold.%s' %self.lang
        self.wordlist_file = self.data_path + 'wordlist.%s' %self.lang
        self.predicted_file = {'train': self.out_path + 'pred.train.%s' %self.lang, 'test': self.out_path + 'pred.test.%s' %self.lang}
        if 'gold_affixes' in kwargs:
            self.gold_affix_file = {'pre': self.data_path + 'gold_pre.%s' %self.lang, 'suf': self.data_path + 'gold_suf.%s' %self.lang}

        sys.stderr = Logger(self.out_path + 'log')

        self.compounding = kwargs['compounding']
        self.sibling = kwargs['sibling']
        self.supervised = kwargs['supervised']
        self.transform = (self.lang == 'eng')# or self.lang == 'ger')
        self.pruner = None

        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.reg_l2 = 1.0
        self.wv_dim = 200

        self.clear_caches()
        self._save_model_params()

        self.DEBUG = kwargs['DEBUG']
        self.INDUCTIVE = False

        if self.DEBUG:
            self.INDUCTIVE = False
            self.word_vector_file += '.toy'
            self.wordlist_file += '.toy'
            self.gold_segs_file += '.toy'

    def _get_out_path(self, msg):
        tmp_time = localtime()
        date = strftime('%m-%d', tmp_time)
        timestamp = strftime('%H:%M:%S', tmp_time)
        if msg is None:
            out_path = 'out/' + date + '/' + timestamp + "/"
        else:
            out_path = 'out/' + date + '/' + msg + '-' + timestamp + "/"
        if not os.path.isdir('out'): os.mkdir('out')
        if not os.path.isdir('out/' + date): os.mkdir('out/' + date)
        if not os.path.isdir(out_path): os.mkdir(out_path)
        return out_path

    def _save_model_params(self):
        print('-----------------------------------', file=sys.stderr)
        print('language\t', self.lang, '\ncompounding\t', self.compounding, '\nsibling\t\t', self.sibling, file=sys.stderr)
        print('wv\t\t', self.word_vector_file, '\t', self.wv_dim, '\ngold segs\t', self.gold_segs_file, '\nwordlist\t', self.wordlist_file, file=sys.stderr)
        print('reg_l2\t\t', self.reg_l2, '\nsupervised\t', self.supervised, '\ntop_words\t', self.top_words, '\ngold_affixes\t', hasattr(self, 'gold_affix_file'), '\t', self.top_affixes, file=sys.stderr)
        print('out path\t', self.out_path, file=sys.stderr)
        print('-----------------------------------', file=sys.stderr)

    ############################# Get stuff from data ##########################

    def read_all_data(self):
        self.read_wordlist()
        self.read_word_vectors()
        self.read_gold_segs()
        self._add_top_words_from_wordlist()
        self.read_affixes()
        if self.INDUCTIVE: self._add_words_from_gold()
        if self.supervised:
            self.get_gold_parents()
            self.train_set = set(self.gold_parents.keys())
        # self.update_train_set()

    # assume utf8 encoding for word vector files
    def read_word_vectors(self, wv_path=None):
        assert not hasattr(self, 'wv') or self.wv is None
        self.wv = dict()
        if not wv_path:
            wv_path = self.word_vector_file
        with codecs.open(wv_path, encoding='utf8', errors='strict') as fin:
            for line in fin:
                segs = line.strip().split(' ')
                self.wv[self._standardize(segs[0])] = np.asarray(map(float, segs[1:]))
        print('Read %d word vectors.' %(len(self.wv)), file=sys.stderr)

    # assume standard texts for gold segmentation files / iso-8859-1 for Finnish
    def read_gold_segs(self):
        assert not hasattr(self, 'gold_segs')
        self.gold_segs = dict()
        with codecs.open(self.gold_segs_file, 'r', 'utf8', errors='strict') as fin:
            for line in fin:
                segs = line.strip().split('\t')
                assert len(segs) % 2 == 0, segs
                segs = '\t'.join(segs[: len(segs) // 2]), '\t'.join(segs[len(segs) // 2:])
                # if len(segs) != 2:
                #     continue
                self.gold_segs[segs[0]] = segs[1].split()
        print('Read %d gold segmentations.' %(len(self.gold_segs)), file=sys.stderr)

    # assume standard texts for wordlist files / iso-8859-1 for Finnish and German
    def read_wordlist(self, wordlist_file=None):

        if not wordlist_file:
            wordlist_file = self.wordlist_file
        assert not hasattr(self, 'word_cnt') or self.word_cnt is None
        self.word_cnt = dict()
        if self.lang == 'fin' or self.lang == 'ger' or self.lang == 'eng':
            f = codecs.open(wordlist_file, 'r', 'iso-8859-1', errors='strict')
        elif self.lang in ['sw', 'tl']:
            f = codecs.open(wordlist_file, 'r', 'utf8')
        else:
            f = open(wordlist_file, 'r')
        for line in f:
            segs = line.split()
            if len(segs) != 2: continue
            if len(segs[0]) >= 3:   # Don't include very short words (length < 3)
                self.word_cnt[segs[0]] = int(segs[1])
        f.close()
        print('Read %d words from wordlist' %(len(self.word_cnt)), file=sys.stderr)

    def _add_top_words_from_wordlist(self):
        assert not hasattr(self, 'train_set')
        self.train_set = set()
        cnt = Counter()
        ptr = defaultdict(list)
        for k, v in self.word_cnt.iteritems():
            if len(k) < 3: continue
            # if '-' not in k or len(k.split("'")) != 2: # ignore hyphenated words, or apostrophed words
            cnt[v] += 1
            ptr[v].append(k)
        cnt = sorted(cnt.items(), key=itemgetter(0), reverse=True)
        if self.top_words > 0:
            i = 0
            while len(self.train_set) < self.top_words and i < len(cnt):
                self.train_set.update(ptr[cnt[i][0]])
                self.freq_thresh = cnt[i][0]
                i += 1
        # self.train_set = self._decompose(self.train_set)
        print("Add %d words from the wordlist to the training set." %(len(self.train_set)), file=sys.stderr)

    def _add_words_from_gold(self):
        self.train_set.update(filter(lambda w: len(w) >= 3, self.gold_segs.keys()))
        print('Now %d words in training set, inductive mode.' %(len(self.train_set)), file=sys.stderr)

    def decompose(self, s):
        if self.lang == 'eng':
            new_set = set()
            for w in s:
                parts = w.split("'")
                if len(parts) == 2:
                    w = parts[0]
                if '-' in w:
                    new_set.update(w.split('-'))
                else:
                    new_set.add(w)
            return new_set
        return s

    def read_affixes(self):
        assert not hasattr(self, 'prefixes') and not hasattr(self, 'suffixes')
        self.prefixes, self.suffixes = set(), set()
        if hasattr(self, 'gold_affix_file'):
            with open(self.gold_affix_file['pre'], 'r') as fpre, open(self.gold_affix_file['suf'], 'r') as fsuf:
                for line in fpre:
                    self.prefixes.add(self._standardize(line.strip()[:-1]))
                for line in fsuf:
                    self.suffixes.add(self._standardize(line.strip()[1:]))
        else:
            suf_cnt, pre_cnt = Counter(), Counter()
            for word in self.train_set:
                for pos in xrange(1, len(word)):
                    left, right = word[:pos], word[pos:]
                    if left in self.word_cnt: suf_cnt[right] += 1
                    if right in self.word_cnt: pre_cnt[left] += 1
            suf_cnt = sorted(suf_cnt.items(), key=itemgetter(1), reverse=True)
            pre_cnt = sorted(pre_cnt.items(), key=itemgetter(1), reverse=True)
            self.suffixes = set([suf for suf, cnt in suf_cnt[:self.top_affixes]])
            self.prefixes = set([pre for pre, cnt in pre_cnt[:self.top_affixes]])


    # Generally, lowercasing shouldn't happen here.
    def _standardize(self, word):
        if self.lang == 'eng' or self.lang == 'ara':
            return word
        elif self.lang == 'tur':
            lower = word.lower()
            tmp = ''
            for i, c in enumerate(word):
                if c == u'I': tmp += u'I'
                else: tmp += lower[i]
            output = ''
            for c in tmp:
                if c == u'ç' or c == u'Ç': output += u'C'
                elif c == u'ğ' or c == u'Ğ': output += u'G'
                elif c == u'ö' or c == u'Ö': output += u'O'
                elif c == u'ş' or c == u'Ş': output += u'S'
                elif c == u'ü' or c == u'Ü': output += u'U'
                elif c == u'ı': output += u'I'
                elif c == u'İ': output += u'i'
                else: output += c
            return output
        elif self.lang == 'ger':
            output = ''
            for c in word:
                if c == u'ü' or c == u'Ü': output += u'ue'
                elif c == u'ö' or c == u'Ö': output += u'oe'
                elif c == u'ä' or c == u'Ä': output += u'ae'
                elif c == u'ß': output += u'ss'
                else: output += c
            return output
        else:
            return word
            #raise NotImplementedError

    def get_gold_parents(self):
        assert not hasattr(self, 'gold_parents') and self.prefixes and self.suffixes
        self.gold_parents = dict()
        for child in self.gold_segs.iterkeys():
            if '-' in child: continue # ignore all hyphenated words.
            segmentation = self.gold_segs[child][0]  # only take the first segmentation
            while True:
                parts = segmentation.split("-")
                ch = "".join(parts)
                # print ch
                if len(parts) == 1:
                    self.gold_parents[ch] = (ch, 'STOP')
                    break
                # simple heuristic to determine the parent.
                scores = dict()
                prefix, suffix = parts[0], parts[-1]
                right_parent = ''.join(parts[1:])
                left_parent = ''.join(parts[:-1])
                prefix_score = self.get_similarity(ch, right_parent)
                suffix_score = self.get_similarity(ch, left_parent)
                p_score, s_score = 0.0, 0.0
                if prefix in self.prefixes: p_score += 1.0
                if suffix in self.suffixes: s_score += 1.0
                prefix_score += p_score
                suffix_score += s_score
                scores[(right_parent, 'PREFIX')] = prefix_score
                scores[(left_parent, 'SUFFIX')] = suffix_score + 0.1

                if right_parent in self.word_cnt and prefix in self.word_cnt:
                    scores[(prefix, right_parent), 'COM_RIGHT'] = 0.75 * self.get_similarity(right_parent, ch) + 0.25 * self.get_similarity(prefix, ch) + 1
                if left_parent in self.word_cnt and suffix in self.word_cnt:
                    scores[(left_parent, suffix), 'COM_LEFT'] = 0.25 * self.get_similarity(suffix, ch) + 0.75 * self.get_similarity(left_parent, ch) + 1

                if self.transform:
                    if (len(left_parent) > 1 and left_parent[-1] == left_parent[-2]):
                        repeat_parent = left_parent[:-1]
                        score = self.get_similarity(ch, repeat_parent) + s_score - 0.25
                        scores[(repeat_parent, 'REPEAT')] = score
                    if left_parent[-1] == 'i':
                        modify_parent = left_parent[:-1] + 'y' # only consider y -> i modification
                        score = self.get_similarity(ch, modify_parent) + s_score - 0.25
                        scores[(modify_parent, 'MODIFY')] = score
                    if left_parent[-1] != 'e':
                        delete_parent = left_parent + "e"    # only consider e deletion.
                        score = self.get_similarity(ch, delete_parent) + s_score - 0.25
                        scores[(delete_parent, 'DELETE')] = score

                best = max(scores.items(), key=itemgetter(1))[0]
                type_ = best[1]
                # print best, scores
                if type_ == 'PREFIX':
                    segmentation = segmentation[len(prefix) + 1:]
                elif type_ == 'SUFFIX':
                    segmentation = segmentation[:len(segmentation) - len(suffix) - 1]
                elif type_ == 'MODIFY':
                    segmentation = segmentation[:len(segmentation) - len(suffix) - 2] + 'y'
                elif type_ == 'REPEAT':
                    segmentation = segmentation[:len(segmentation) - len(suffix) - 2]
                elif type_ == 'DELETE':
                    segmentation = segmentation[:len(segmentation) - len(suffix) - 1] + 'e'
                elif type_ == 'COM_LEFT':
                    segmentation = segmentation[:-len(suffix) - 1]
                elif type_ == 'COM_RIGHT':
                    segmentation = segmentation[len(prefix) + 1:]
                self.gold_parents[ch] = best
        print('Read %d gold parents.' %(len(self.gold_parents)), file=sys.stderr)

    ############################# features #####################################

    def get_raw_features(self, child, candidate): # pair is a child-parent pair
        # if (child, candidate) in self.features_cache: self.features_cache[(child, candidate)]
        if (child, candidate) in self.features_cache: self.features_cache[(child, candidate)]

        parent, type_ = candidate
        pair = Pair(child, *candidate)

        features = dict()
        features['BIAS'] = 1.0
        if type_ == 'STOP':
            assert child == parent
            bi_start = parent[:2]
            bi_end = parent[-2:]
            features['BI_S_' + bi_start] = 1.0
            features['BI_E_' + bi_end] = 1.0
            features['LEN_%d' %(len(parent))] = 1.0
            max_cos = self._get_max_cos(parent)
            if max_cos > -2:    # -2 is exit error code
                features['MAXCOS_%d' %(int(max_cos * 10))] = 1.0
        else:
            if type_ != 'COM_LEFT' and type_ != 'COM_RIGHT':
                cos = self.get_similarity(child, parent)
                features['COS'] = cos
                if parent in self.word_cnt:
                    features['CNT'] = math.log(self.word_cnt[parent])
                    features['IV'] = 1.0
                else:
                    features['OOV'] = 1.0
            affix, trans = pair.get_affix_and_transformation()
            if self.sibling:
                if pair.type_coarse == 'suf' and affix in self.suffixes or pair.type_coarse == 'pre' and affix in self.prefixes:
                    self._get_sibling_feature(pair, features)
            if type_ == 'PREFIX':
                if affix in self.prefixes: features['PRE_' + affix] = 1.0
            elif type_ == 'SUFFIX' or type_ == 'APOSTR':
                if affix in self.suffixes: features['SUF_' + affix] = 1.0
            elif type_ == 'MODIFY':
                if affix in self.suffixes: features['SUF_' + affix] = 1.0
                if not self.pruner or trans not in self.pruner['MODIFY']:
                    features[trans] = 1.0
            elif type_ == 'DELETE':
                if affix in self.suffixes:
                    features['SUF_' + affix] = 1.0
                if not self.pruner or trans not in self.pruner['DELETE']:
                    features[trans] = 1.0
            elif type_ == 'REPEAT':
                if affix in self.suffixes:
                    features['SUF_' + affix] = 1.0
                if not self.pruner or trans not in self.pruner['REPEAT']:
                    features[trans] = 1.0
            elif type_ == 'COM_LEFT':
                parent, aux = parent
                features['HEAD_CNT'] = math.log(self.word_cnt[parent])
                features['HEAD_COS'] = self.get_similarity(child, parent)
                features['AUX_CNT'] = math.log(self.word_cnt[aux])
                features['AUX_COS'] = self.get_similarity(child, aux)
            elif type_ == 'COM_RIGHT' or type_ == 'HYPHEN':
                aux, parent = parent
                features['HEAD_CNT'] = math.log(self.word_cnt[parent])
                features['HEAD_COS'] = self.get_similarity(child, parent)
                features['AUX_CNT'] = math.log(self.word_cnt[aux])
                features['AUX_COS'] = self.get_similarity(child, aux)
            else:
                raise NotImplementedError, 'no such type %s' %(type_)
        self.features_cache[(child, candidate)] = features
        return features

    def _get_max_cos(self, child):
        if child in self.max_cos: return self.max_cos[child]
        max_cos = max([-2] + [self.get_similarity(child, parent) for parent, type_ in self.get_candidates(child) if type_ != 'STOP']) # -2 as exit code
        self.max_cos[child] = max_cos
        return max_cos

    def _get_sibling_feature(self, pair, features, top_K=1000000):
        neighbors = self.prefixes if pair.type_coarse == 'pre' else self.suffixes
        name = 'COR_' + ('P_' if pair.type_coarse == 'pre' else 'S_') + pair.get_affix()
        cnt = 0
        for neighbor in neighbors:
            if pair.affix != neighbor:
                if self._get_surface_form(pair, neighbor) in self.word_cnt:
                    if name in features:
                        features[name] += 1.0
                    else:
                        features[name] = 1.0
                    cnt += 1
                    if cnt == top_K: break
        if name in features: features[name] = math.log(1.0 + features[name])

    def _get_surface_form(self, pair, neighbor):
        parent, type_ = pair.parent, pair.type_
        if type_ == 'PREFIX': return neighbor + parent
        if type_ == 'SUFFIX': return parent + neighbor
        if type_ == 'MODIFY': return parent[:-1] + pair.trans[-1] + neighbor
        if type_ == 'DELETE': return parent[:-1] + neighbor
        assert type_ == 'REPEAT'
        return parent + parent[-1] + neighbor

    ############################# Candidates and neighbors #####################

    # gold means only take gold segmentation if available
    def get_candidates(self, word, gold=False, top=False):
        candidates = set()
        if self.supervised:
            if word in self.gold_parents:
                candidates.add(self.gold_parents[word])
            if gold:
                return candidates

        if word in self.candidates_cache: return self.candidates_cache[word]

        candidates.add((word, 'STOP'))
        if len(word) < 3: return candidates
        for pos in xrange(1, len(word)):
            parent = word[:pos]
            if self.compounding and parent in self.word_cnt and word[pos:] in self.word_cnt:
                if self.word_cnt[parent] >= self.freq_thresh and self.word_cnt[word[pos:]] >= self.freq_thresh:
                    candidates.add(((parent, word[pos:]), 'COM_LEFT'))
                    candidates.add(((parent, word[pos:]), 'COM_RIGHT'))
            if 2 * len(parent) >= len(word):
                pair = Pair(word, parent, 'SUFFIX')
                suf, _ = pair.get_affix_and_transformation()
                if not self.pruner or suf not in self.pruner['suf']:
                    candidates.add((parent, 'SUFFIX'))
                if self.transform:
                    if pos < len(word) - 1 and word[pos - 1] == word[pos]:
                        pair = Pair(word, parent, 'REPEAT')
                        suf, trans = pair.get_affix_and_transformation()
                        if not self.pruner or suf not in self.pruner['suf']:
                            if not self.pruner or trans not in self.pruner['REPEAT']:
                                candidates.add((parent, 'REPEAT'))
                    if parent[-1] in self.alphabet:
                        for char in self.alphabet:
                            if char == parent[-1]: continue
                            new_parent = parent[:-1] + char
                            if self.get_similarity(new_parent, word) > 0.2:
                                pair = Pair(word, new_parent, 'MODIFY')
                                suf, trans = pair.get_affix_and_transformation()
                                if not self.pruner or suf not in self.pruner['suf']:
                                    if not self.pruner or trans not in self.pruner['MODIFY']:
                                        candidates.add((new_parent, 'MODIFY'))
                    if pos < len(word) - 1 and word[pos:] in self.suffixes:
                        for char in self.alphabet:
                            new_parent = parent + char
                            if word == new_parent: continue
                            if new_parent in self.word_cnt:
                                pair = Pair(word, new_parent, 'DELETE')
                                suf, trans = pair.get_affix_and_transformation()
                                if not self.pruner or suf not in self.pruner['suf']:
                                    if not self.pruner or trans not in self.pruner['DELETE']:
                                        candidates.add((new_parent, 'DELETE'))
            parent = word[pos:]
            if len(parent) * 2 >= len(word):
                pair = Pair(word, parent, 'PREFIX')
                pre, _ = pair.get_affix_and_transformation()
                if not self.pruner or pre not in self.pruner['pre']:
                    candidates.add((parent, 'PREFIX'))
        self.candidates_cache[word] = candidates
        return candidates

    # expand means we need to get the neighbors in addition to the word itself. This means no supervision.
    def get_neighbors(self, word, expand=True):
        if not expand and self.supervised:  # only activated if it's in suprvised mode
            # if word in self.gold_segs:
            return set([word])
        if word in self.neighbors_cache: return self.neighbors_cache[word]
        neighbors = set()
        neighbors.add(word)  # word is in its own neighbors set
        positions = set()
        n = len(word)
        num_neighbors = 5
        for i in xrange(num_neighbors):
            if n > i + 1:
                positions.add(i)
            if n - i - 2 >= 0:
                positions.add(n - i - 2)
        for pos in positions:
            # permute and add
            new_word = word[:pos] + word[pos + 1] + word[pos]
            if pos + 2 < n:
                new_word += word[pos + 2:]
            neighbors.add(new_word)

        if n >= 4:
            neighbors.add(word[1] + word[0] + word[2: n - 2] + word[n - 1] + word[n - 2])
            if n >= 5:
                neighbors.add(word[0] + word[2] + word[1] + word[3: n - 2] + word[n - 1] + word[n - 2])
            if n >= 6:
                neighbors.add(word[0] + word[2] + word[1] + word[3: n - 3] + word[n - 2] + word[n - 3] + word[n - 1])
            if n >= 5:
                neighbors.add(word[1] + word[0] + word[2: n - 3] + word[n - 2] + word[n - 3] + word[n - 1])

        self.neighbors_cache[word] = neighbors
        return neighbors

    ############################# Computation graph ############################

    def _compute_loss(self):
        # first pass to know the how many paddings we need
        max_r_num, max_r_den = 0, 0
        print('First pass to gather info...')
        for word_i, word in enumerate(self.train_set):
            print('\r%d/%d' %(word_i + 1, len(self.train_set)), end='')
            sys.stdout.flush()
            # if self.supervised and word not in self.gold_parents: continue
            acc = 0
            for neighbor in self.get_neighbors(word, expand=False):
                acc += len(self.get_candidates(neighbor, gold=False))#, top=top))
            max_r_num = max(max_r_num, len(self.get_candidates(word, gold=True)))
            max_r_den = max(max_r_den, acc)

        self.feature2index = dict()
        self.index2feature = list()
        self.weights = None

        data_num, row_num, col_num = list(), list(), list()
        data_den, row_den, col_den = list(), list(), list()
        cnt_num, cnt_den = list(), list()
        print('\nBuilding sparse matrix for MC...')
        i = 0
        for word_i, word in enumerate(self.train_set):
            print('\r%d/%d' %(word_i + 1, len(self.train_set)), end='')
            sys.stdout.flush()
            # if self.supervised and word not in self.gold_parents: continue
            r_num, r_den = max_r_num * i, max_r_den * i
            for neighbor in self.get_neighbors(word, expand=False):
                candidates = self.get_candidates(neighbor, gold=False)
                for candidate in candidates:
                    features = self.get_raw_features(neighbor, candidate)
                    self._populate_coordinates(features, data_den, row_den, col_den, r_den)
                    cnt_den.append(1.0)
                    r_den += 1
            for r in xrange(r_den, max_r_den * (i + 1)):
                cnt_den.append(0.0)
            candidates = self.get_candidates(word, gold=True)
            for candidate in candidates:
                features = self.get_raw_features(word, candidate)
                self._populate_coordinates(features, data_num, row_num, col_num, r_num)
                cnt_num.append(1.0)
                r_num += 1
            for r in xrange(r_num, max_r_num * (i + 1)):
                cnt_num.append(0.0)
            i += 1
        print('\nNum of features for MC is', len(self.feature2index), file=sys.stderr)
        # build matrices
        weights = [0.0] * len(self.feature2index)
        weights = theano.shared(np.asarray(weights), name='w')
        cnt_num = theano.shared(np.asarray(cnt_num), name='cnt_num')
        cnt_den = theano.shared(np.asarray(cnt_den), name='cnt_den')

        N = len(self.train_set) if not self.supervised else len(self.gold_parents)
        numerator = sp.csr_matrix((data_num, (row_num, col_num)), shape=(N * max_r_num, len(self.feature2index)))
        denominator = sp.csr_matrix((data_den, (row_den, col_den)), shape=(N * max_r_den, len(self.feature2index)))
        num = (T.exp(theano.sparse.dot(numerator, weights)) * cnt_num).reshape((N, max_r_num))
        den = (T.exp(theano.sparse.dot(denominator, weights)) * cnt_den).reshape((N, max_r_den))
        loss = -T.sum(T.log(T.sum(num, axis=1) / T.sum(den, axis=1)))
        return weights, loss

    # populate coordinates with features, expanding feature set along the way
    def _populate_coordinates(self, features, data, row, col, r):
        for k, v in features.items():
            ind = self._get_index(k)
            data.append(v)
            row.append(r)
            col.append(self.feature2index[k])

    def _get_index(self, name):
        if name in self.feature2index: return self.feature2index[name]
        else:
            self.feature2index[name] = len(self.feature2index)
            self.index2feature.append(name)
            return self.feature2index[name]

    #############################  helper functions ############################

    def get_similarity(self, w1, w2):
        if w1 not in self.wv or w2 not in self.wv: return -0.5
        sim = 1.0 - cos_dist(self.wv[w1], self.wv[w2])
        return sim

    def get_weight(self, name):
        assert name in self.feature2index
        return self.weights[self.feature2index[name]]

    def get_prob(self, child, candidate):
        if (child, candidate) in self.prob_cache: return self.prob_cache[(child, candidate)]
        candidates = set(self.get_candidates(child))
        # # this is a bit messy
        # if child in self.gold_parents:
        #     candidates.add(self.gold_parents[child])
        # candidates.add(candidate)
        scores = {cand: math.exp(self.score_candidate(child, cand)) for cand in candidates}
        z = sum(scores.values())
        for cand in candidates:
            self.prob_cache[(child, cand)] = scores[cand] / z
        return self.prob_cache[(child, candidate)]

    ############################# Inference ####################################

    def score_candidate(self, child, candidate):
        s = 0.0
        for k, v in self.get_raw_features(child, candidate).items():
            if k in self.feature2index:
                ind = self.feature2index[k]
                s += v * self.weights[ind]
        return s

    def predict(self, word):
        scores = [(self.score_candidate(word, candidate), candidate) for candidate in self.get_candidates(word)]
        best = max(scores, key=itemgetter(0))
        return best[1]

    def segment(self, word, mode='surface'):
        path = self.get_seg_path(word)
        return path.get_segmentation(mode=mode)

    def get_seg_path(self, word):
        path = Path(word)
        while not path.is_ended():
            child = path.get_fringe_word()
            parts = child.split("'")
            if len(parts) == 2 and len(parts[0]) > 0 and self.lang == 'eng':
                path.expand(child, parts[0], 'APOSTR')
            else:
                parts = child.split('-')
                if len(parts) > 1:
                    p1, p2 = parts[0], child[len(parts[0]) + 1:]
                    path.expand(child, (p1, p2), 'HYPHEN')
                else:
                    parent, type_ = self.predict(child)
                    path.expand(child, parent, type_)
        return path

    def run(self, reread=True):
        if reread: self.read_all_data()
        if not self.supervised:
            self.clear_caches()
            w, loss = self._compute_loss()
            loss += self.reg_l2 * T.sum(w ** 2)
            optimizer = Adam()
            optimizer.run(w, loss)
            self.weights = w.get_value()
            # write weights to log
            with codecs.open(self.out_path + 'MC.weights', 'w', 'utf8', errors='ignore') as fout:
                for i, v in sorted(list(enumerate(self.weights)), key=itemgetter(1), reverse=True):
                    tmp = '%s\t%f\n' %(self.index2feature[i], v)
                    try:
                        fout.write(tmp)
                    except:
                        import ipdb; ipdb.set_trace()
            self.clear_caches()
            self.write_segments_to_file()
            p, r, f = self.evaluate()
            print(p, r, f)
        else:
            p, r, f = 0.0, 0.0, 0.0
            self.gold_segs_copy = dict(self.gold_segs)
            fold = random.random()
            for iter_ in range(5):
                self.clear_caches()
                print('###########################################')
                print('Iteration %d' %iter_)
                del self.gold_parents
                test_set = set(random.sample(self.gold_segs_copy.keys(), len(self.gold_segs_copy) // 5))
                self.gold_segs = {k: v for k, v in self.gold_segs_copy.iteritems() if k not in test_set}
                self.get_gold_parents()
                self.train_set = set(self.gold_parents.keys())
                w, loss = self._compute_loss()
                loss += self.reg_l2 * T.sum(w ** 2)
                optimizer = Adam()
                optimizer.run(w, loss)
                self.weights = w.get_value()
                # write weights to log
                with codecs.open(self.out_path + 'MC.weights', 'w', 'utf8', errors='strict') as fout:
                    for i, v in sorted(list(enumerate(self.weights)), key=itemgetter(1), reverse=True):
                        tmp = '%s\t%f\n' %(self.index2feature[i], v)
                        fout.write(tmp)
                print('###########################################')
                self.clear_caches()
                self.write_segments_to_file(wordset=test_set)
                p1, r1, f1 = self.evaluate()
                p += p1; r += r1; f += f1
            self.gold_segs = self.gold_segs_copy
            print(p / 5, r / 5, f /5)

    def segment_all(self, wordset):
        for i, word in enumerate(wordset):
            seg = None
            try:
                if type(word) is str:
                    word = word.decode('utf8')
                seg = self.segment(word)
                if type(seg) is str:
                    seg = seg.decode('utf8')
                # note, if string is already in unicode, do not decode again with utf8
                # ERROR --> 'abcd'.decode('utf8').decode('utf8')
                yield word, seg
            except Exception as e:
                print(type(word), type(seg), file=sys.stderr)
                print(word, file=sys.stderr)
                print(seg, file=sys.stderr)
                traceback.print_exc()
                raise e

    def write_segments_to_file(self, wordset=None, out_file=None):
        if not wordset:
            wordset = set(self.gold_segs.keys())
        if not out_file:
            out_file = self.predicted_file['train']
        if isinstance(out_file, str):
            fout = codecs.open(out_file, 'w', 'utf8', errors='strict')
        else:
            fout = out_file
        for word, seg in self.segment_all(wordset):
            line = u'%s\t%s\n' % (word, seg)
            fout.write(line)
        fout.close()

    def evaluate(self):
        p, r, f = evaluate(self.gold_segs_file, self.predicted_file['train'], quiet=True)
        print('MC: precision =', p, 'recall =', r, 'f =', f, file=sys.stderr)
        return (p, r, f)

    def update_pruner(self, pruner):
        self.pruner = pruner
        for p in pruner['pre']:
            if p in self.prefixes: self.prefixes.remove(p)
        for s in pruner['suf']:
            if s in self.suffixes: self.suffixes.remove(s)

    def clear_caches(self):
        if not hasattr(self, 'neighbors_cache'): self.neighbors_cache = dict()
        self.candidates_cache, self.features_cache, self.prob_cache = [dict() for _ in range(3)]
        self.max_cos = dict()
