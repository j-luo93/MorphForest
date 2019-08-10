import logging
import math

import torch
import torch.nn as nn
import torch.sparse as sp
from enlighten import Counter

from arglib import use_arguments_as_properties
from dev_misc import Map, Metric, Metrics, cache

from .path import Path


@use_arguments_as_properties('lang')
class LogLinearModel(nn.Module):

    def __init__(self, feature_ext):
        super().__init__()
        self.feature_ext = feature_ext

    def first_pass(self, batch):
        """First pass to know the how many paddings we need."""
        max_r_num, max_r_den = 0, 0
        pbar = Counter()
        logging.info('First pass to gather info...')
        for word in pbar(batch.wordlist):
            # if self.supervised and word not in self.gold_parents: continue
            acc = 0
            for neighbor in self.feature_ext.get_neighbors(word, expand=False):
                acc += len(self.feature_ext.get_candidates(neighbor, gold=False))  # , top=top))
                max_r_num = max(max_r_num, len(self.feature_ext.get_candidates(word, gold=True)))
                max_r_den = max(max_r_den, acc)

        self.feature2index = dict()
        self.index2feature = list()

        data_num, row_num, col_num = list(), list(), list()
        data_den, row_den, col_den = list(), list(), list()
        cnt_num, cnt_den = list(), list()
        logging.info('Building sparse matrix for MC...')
        i = 0
        pbar = Counter()
        for word_i, word in enumerate(pbar(batch.wordlist)):
            # if self.supervised and word not in self.gold_parents: continue
            r_num, r_den = max_r_num * i, max_r_den * i
            for neighbor in self.feature_ext.get_neighbors(word, expand=False):
                candidates = self.feature_ext.get_candidates(neighbor, gold=False)
                for candidate in candidates:
                    features = self.feature_ext.get_raw_features(neighbor, candidate)
                    self._populate_coordinates(features, data_den, row_den, col_den, r_den)
                    cnt_den.append(1.0)
                    r_den += 1
            for _ in range(r_den, max_r_den * (i + 1)):
                cnt_den.append(0.0)
            candidates = self.feature_ext.get_candidates(word, gold=True)
            for candidate in candidates:
                features = self.feature_ext.get_raw_features(word, candidate)
                self._populate_coordinates(features, data_num, row_num, col_num, r_num)
                cnt_num.append(1.0)
                r_num += 1
            for r in range(r_num, max_r_num * (i + 1)):
                cnt_num.append(0.0)
            i += 1
        logging.info(f'Num of features for MC is {len(self.feature2index)}')

        # NOTE Old theano code.
        # weights = [0.0] * len(feature2index)
        # weights = theano.shared(np.asarray(weights), name='w')
        # cnt_num = theano.shared(np.asarray(cnt_num), name='cnt_num')
        # cnt_den = theano.shared(np.asarray(cnt_den), name='cnt_den')
        # N = len(self.train_set) if not self.supervised else len(self.gold_parents)
        # numerator = sp.csr_matrix((data_num, (row_num, col_num)), shape=(N * max_r_num, len(feature2index)))
        # denominator = sp.csr_matrix((data_den, (row_den, col_den)), shape=(N * max_r_den, len(feature2index)))
        # num = (T.exp(theano.sparse.dot(numerator, weights)) * cnt_num).reshape((N, max_r_num))
        # den = (T.exp(theano.sparse.dot(denominator, weights)) * cnt_den).reshape((N, max_r_den))
        # loss = -T.sum(T.log(T.sum(num, axis=1) / T.sum(den, axis=1)))
        # return weights, loss

        # Build matrix.
        N = len(batch.wordlist)
        M = len(self.feature2index)
        self.cnt_num = torch.FloatTensor(cnt_num)
        self.cnt_den = torch.FloatTensor(cnt_den)
        row_num = torch.LongTensor(row_num)
        col_num = torch.LongTensor(col_num)
        row_den = torch.LongTensor(row_den)
        col_den = torch.LongTensor(col_den)
        # Get numerator.
        indices_num = torch.stack([row_num, col_num], dim=0)
        values_num = torch.FloatTensor(data_num).float()
        self.numerator = sp.FloatTensor(indices_num, values_num, [N * max_r_num, M])
        # Get denominator.
        indices_den = torch.stack([row_den, col_den], dim=0)
        values_den = torch.FloatTensor(data_den).float()
        self.denominator = sp.FloatTensor(indices_den, values_den, [N * max_r_den, M])
        self.weights = nn.Parameter(torch.zeros(M))

    def forward(self, batch):
        N = len(batch.wordlist)
        den = ((self.denominator @ self.weights.view(-1, 1)).exp() * self.cnt_den).view(N, -1)
        num = ((self.numerator @ self.weights.view(-1, 1)).exp() * self.cnt_num).view(N, -1)
        nll = -(num.sum(dim=1) / den.sum(dim=1)).log().sum()
        nll = Metric('nll', nll, batch.num_samples)
        reg_l2 = (self.weights ** 2).sum()
        reg_l2 = Metric('reg_l2', reg_l2, batch.num_samples)
        return Metrics(nll, reg_l2)

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

    def get_weight(self, name):
        assert name in self.feature2index
        return self.weights[self.feature2index[name]]

    @cache(persist=False, full=True)
    def get_probs_for_child(self, child):
        candidates = set(self.feature_ext.get_candidates(child))
        # # this is a bit messy
        # if child in self.gold_parents:
        #     candidates.add(self.gold_parents[child])
        # candidates.add(candidate)
        scores = {cand: math.exp(self.score_candidate(child, cand)) for cand in candidates}
        z = sum(scores.values())
        ret = dict()
        for cand in candidates:
            ret[(child, cand)] = scores[cand] / z
        return ret

    def get_prob(self, child, candidate):
        all_probs = self.get_probs_for_child(child)
        return all_probs[(child, candidate)]

    def score_candidate(self, child, candidate):
        s = 0.0
        for k, v in self.feature_ext.get_raw_features(child, candidate).items():
            if k in self.feature2index:
                ind = self.feature2index[k]
                s += v * self.weights[ind]
        return s

    def predict(self, word):
        scores = [(self.score_candidate(word, candidate), candidate)
                  for candidate in self.feature_ext.get_candidates(word)]
        best = max(scores, key=lambda x: x[0])
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
