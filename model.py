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
from .word_vectors import WordVectors

EN_ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


@use_arguments_as_properties('lang', 'top_affixes', 'top_words', 'compounding', 'sibling', 'debug')
class MC:

    ############################# Initialization ###############################

    def __init__(self):


        self.pruner = None

        clear_cache()

        # TODO save now???
        self._save_model_params()

        if self.debug:  # TODO this should be in another cfg.
            self.inductive = False
            self.word_vector_file += '.toy'
            self.wordlist_file += '.toy'
            self.gold_segs_file += '.toy'

        self.feature_extractor = FeatureExtractor()  # FIXME


    # TODO looks weird. Put this in trainer.
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
