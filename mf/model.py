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

from mf.feature_extractor import FeatureExtractor
from mf.word_vectors import WordVectors


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

        self.feature_extractor = FeatureExtractor()

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
