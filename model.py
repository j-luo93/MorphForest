import logging
import os
import pathlib
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
from operator import itemgetter
from time import localtime, strftime

import numpy as np
import scipy.sparse as sp
from path import Path
from scipy.linalg import norm
from scipy.spatial.distance import cosine as cos_dist

from arglib import use_arguments_as_properties
from dev_misc import cache, clear_cache, log_this
from pair import ChildParentPair as Pair
from SGD import Adam
from utils import Logger

from .feature_extractor import FeatureExtractor
from .trainer import Trainer
from .word_vectors import WordVectors

EN_ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


@use_arguments_as_properties('data_path', 'lang', 'top_affixes', 'top_words', 'compounding', 'sibling', 'supervised', 'debug', 'log_dir', 'gold_affixes')
class MC:

    ############################# Initialization ###############################

    def __init__(self, **kwargs):
        if self.supervised:
            raise NotImplementedError('Supervised mode not supported yet.')

        self._prepare_paths()

        self.pruner = None

        clear_cache()

        # TODO save now???
        self._save_model_params()

        self.inductive = True
        if self.debug:  # TODO this should be in another cfg.
            self.inductive = False
            self.word_vector_file += '.toy'
            self.wordlist_file += '.toy'
            self.gold_segs_file += '.toy'

        self.feature_extractor = FeatureExtractor()  # FIXME
        self.trainer = Trainer()  # FIXME

    def _prepare_paths(self):
        self.word_vector_file = self.data_path + 'wv.%s' % self.lang
        self.gold_segs_file = self.data_path + 'gold.%s' % self.lang
        self.wordlist_file = self.data_path + 'wordlist.%s' % self.lang
        self.predicted_file = {'train': self.log_dir + 'pred.train.%s' % self.lang,
                               'test': self.log_dir + 'pred.test.%s' % self.lang}
        if self.gold_affixes:
            self.gold_affix_file = {'pre': self.data_path + 'gold_pre.%s' % self.lang,
                                    'suf': self.data_path + 'gold_suf.%s' % self.lang}

    # TODO lookes weird.
    def _save_model_params(self):
        print('-----------------------------------', file=sys.stderr)
        print('language\t', self.lang, '\ncompounding\t', self.compounding,
              '\nsibling\t\t', self.sibling, file=sys.stderr)
        print('wv\t\t', self.word_vector_file, '\t', self.wv_dim, '\ngold segs\t',
              self.gold_segs_file, '\nwordlist\t', self.wordlist_file, file=sys.stderr)
        print('reg_l2\t\t', self.reg_l2, '\nsupervised\t', self.supervised, '\ntop_words\t', self.top_words,
              '\ngold_affixes\t', hasattr(self, 'gold_affix_file'), '\t', self.top_affixes, file=sys.stderr)
        print('out path\t', self.out_path, file=sys.stderr)
        print('-----------------------------------', file=sys.stderr)

    ############################# Get stuff from data ##########################

    def read_all_data(self):
        self.read_wordlist()
        self.read_word_vectors()  # TODO add option to not read them.
        self.read_gold_segs()
        self._add_top_words_from_wordlist()
        self.read_affixes()
        if self.inductive:
            self._add_words_from_gold()
        if self.supervised:
            self.get_gold_parents()
            self.train_set = set(self.gold_parents.keys())
        # self.update_train_set()

    def _check_first_time_read(self, attr):
        if hasattr(self, attr):
            raise AttributeError(f'Attribute "{attr}" has already been read.')

    @log_this('INFO')
    def read_word_vectors(self):
        self._check_first_time_read('wv')
        self.wv = WordVectors(self.word_vector_file)  # FIXME do this

    @log_this('INFO')
    def read_gold_segs(self):
        self._check_first_time_read('gold_segs')
        self.gold_segs = dict()
        with pathlib.Path(self.gold_segs_file).open('r', encoding='utf8') as fin:
            for i, line in enumerate(fin, 1):
                segs = line.strip().split(':')
                if len(segs) != 2:
                    raise RuntimeError(f'Something is wrong with the segmentation file at line {i}.')
                self.gold_segs[segs[0]] = segs[1].split()

    @log_this('INFO')
    def read_wordlist(self):
        self._check_first_time_read('word_cnt')
        self.word_cnt = dict()
        with pathlib.Path(self.wordlist_file).open('r', encoding='utf8') as fin:
            for i, line in enumerate(fin, 1):
                segs = line.split()
                if len(segs) != 2:
                    raise RuntimeError(f'Something is wrong with the word count file at line {i}.')
                # Don't include very short words (length < 3)
                if len(segs[0]) >= 3:
                    self.word_cnt[segs[0]] = int(segs[1])

    @log_this('INFO')
    def _add_top_words_from_wordlist(self):
        self._check_first_time_read('train_set')
        self.train_set = set()
        cnt = Counter()
        ptr = defaultdict(list)
        for k, v in self.word_cnt.items():
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
        logging.info('Added %d words to training set.' % (len(self.train_set)))

    @log_this('INFO')
    def _add_words_from_gold(self):
        self.train_set.update(filter(lambda w: len(w) >= 3, self.gold_segs.keys()))
        logging.info('Now %d words in training set, inductive mode.' % (len(self.train_set)))

    @log_this('INFO')
    def read_affixes(self):
        self._check_first_time_read('prefixes')
        self._check_first_time_read('suffixes')
        self.prefixes, self.suffixes = set(), set()
        if hasattr(self, 'gold_affix_file'):
            raise NotImplementedError('Support for gold affixes not implemented.')
        else:
            # TODO isn't this wrong?
            suf_cnt, pre_cnt = Counter(), Counter()
            for word in self.train_set:
                for pos in range(1, len(word)):
                    left, right = word[:pos], word[pos:]
                    if left in self.word_cnt: suf_cnt[right] += 1
                    if right in self.word_cnt: pre_cnt[left] += 1
            suf_cnt = sorted(suf_cnt.items(), key=itemgetter(1), reverse=True)
            pre_cnt = sorted(pre_cnt.items(), key=itemgetter(1), reverse=True)
            self.suffixes = set([suf for suf, cnt in suf_cnt[:self.top_affixes]])
            self.prefixes = set([pre for pre, cnt in pre_cnt[:self.top_affixes]])

    def get_gold_parents(self):
        self._check_first_time_read('gold_parents')
        if not self.prefixes or not self.suffixes:
            raise RuntimeError('Should have read prefixes and suffixes before this.')
        self.gold_parents = dict()
        for child in self.gold_segs:
            # ignore all hyphenated words.
            if '-' in child:
                continue

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
                prefix_score = self.get_similarity(ch, right_parent)  # FIXME fix api
                suffix_score = self.get_similarity(ch, left_parent)
                p_score, s_score = 0.0, 0.0
                if prefix in self.prefixes: p_score += 1.0
                if suffix in self.suffixes: s_score += 1.0
                prefix_score += p_score
                suffix_score += s_score
                scores[(right_parent, 'PREFIX')] = prefix_score
                scores[(left_parent, 'SUFFIX')] = suffix_score + 0.1

                if right_parent in self.word_cnt and prefix in self.word_cnt:
                    scores[(prefix, right_parent), 'COM_RIGHT'] = 0.75 * \
                        self.get_similarity(right_parent, ch) + 0.25 * self.get_similarity(prefix, ch) + 1
                if left_parent in self.word_cnt and suffix in self.word_cnt:
                    scores[(left_parent, suffix), 'COM_LEFT'] = 0.25 * self.get_similarity(suffix, ch) + \
                        0.75 * self.get_similarity(left_parent, ch) + 1

                if self.transform:
                    if (len(left_parent) > 1 and left_parent[-1] == left_parent[-2]):
                        repeat_parent = left_parent[:-1]
                        score = self.get_similarity(ch, repeat_parent) + s_score - 0.25
                        scores[(repeat_parent, 'REPEAT')] = score
                    if left_parent[-1] == 'i':
                        modify_parent = left_parent[:-1] + 'y'  # only consider y -> i modification
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
        print('Read %d gold parents.' % (len(self.gold_parents)), file=sys.stderr)

    ############################# Candidates and neighbors #####################

    @cache(persist=False, full=True)  # NOTE `persist` is set to False because some affixes might be pruned.
    def get_candidates(self, word, gold=False):
        global EN_ALPHABET

        candidates = set()
        if self.supervised:
            if word in self.gold_parents:
                candidates.add(self.gold_parents[word])
            if gold:
                return candidates

        candidates.add((word, 'STOP'))
        if len(word) < 3:
            return candidates
        for pos in range(1, len(word)):
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
                # NOTE Use rules of transformation for English.
                if self.lang == 'en':
                    if pos < len(word) - 1 and word[pos - 1] == word[pos]:
                        pair = Pair(word, parent, 'REPEAT')
                        suf, trans = pair.get_affix_and_transformation()
                        if not self.pruner or suf not in self.pruner['suf']:
                            if not self.pruner or trans not in self.pruner['REPEAT']:
                                candidates.add((parent, 'REPEAT'))
                    if parent[-1] in EN_ALPHABET:
                        for char in EN_ALPHABET:
                            if char == parent[-1]: continue
                            new_parent = parent[:-1] + char
                            if self.get_similarity(new_parent, word) > 0.2:
                                pair = Pair(word, new_parent, 'MODIFY')
                                suf, trans = pair.get_affix_and_transformation()
                                if not self.pruner or suf not in self.pruner['suf']:
                                    if not self.pruner or trans not in self.pruner['MODIFY']:
                                        candidates.add((new_parent, 'MODIFY'))
                    if pos < len(word) - 1 and word[pos:] in self.suffixes:
                        for char in EN_ALPHABET:
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
        return candidates

    @cache(persist=True, full=True)
    def get_neighbors(self, word, expand=True):
        """Get neighbors for a word.

        Args:
            word (str): just a word
            expand (bool, optional): `expand` means we need to get the neighbors in addition to the word itself. This means no supervision. Defaults to True.

        Returns:
            set: the set of neighbors
        """
        if not expand and self.supervised:  # only activated if it's in suprvised mode
            return set([word])

        neighbors = set()
        neighbors.add(word)  # word is in its own neighbors set
        positions = set()
        n = len(word)
        num_neighbors = 5
        for i in range(num_neighbors):
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

        return neighbors

    ############################# Main ####################################

    def run(self, reread=True):
        if reread:
            self.read_all_data()
        if not self.supervised:
            self.trainer.train()  # FIXME
            self.trainer.update_pruner()  # FIXME
        else:
            raise NotImplementedError('Not implemented. Should not have come here.')
            p, r, f = 0.0, 0.0, 0.0
            self.gold_segs_copy = dict(self.gold_segs)
            fold = random.random()
            for iter_ in range(5):
                clear_cache()
                print('###########################################')
                print('Iteration %d' % iter_)
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
                with codecs.open(self.log_dir + 'MC.weights', 'w', 'utf8', errors='strict') as fout:
                    for i, v in sorted(list(enumerate(self.weights)), key=itemgetter(1), reverse=True):
                        tmp = '%s\t%f\n' % (self.index2feature[i], v)
                        fout.write(tmp)
                print('###########################################')
                clear_cache()
                self.write_segments_to_file(wordset=test_set)
                p1, r1, f1 = self.evaluate()
                p += p1; r += r1; f += f1
            self.gold_segs = self.gold_segs_copy
            print(p / 5, r / 5, f / 5)
