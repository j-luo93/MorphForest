import csv
from abc import ABC, abstractmethod

import pandas as pd
from scipy.spatial.distance import cosine as cos_dist

from arglib import use_arguments_as_properties


class BaseWordVectors(ABC):

    @abstractmethod
    def get_similarity(self, w1, w2):
        pass


class DummyWordVectors(BaseWordVectors):

    def get_similarity(self, w1, w2):
        return 0.0


@use_arguments_as_properties('strict_wv', 'default_oov', 'wv_dim')
class WordVectors:

    def __init__(self, wv_path):
        df = pd.read_csv(wv_path, sep=' ', encoding='utf8',
                         keep_default_na=False, quoting=csv.QUOTE_NONE, header=None)
        self._words = df[0].values
        self._word2id = {word: idx for idx, word in enumerate(self._words)}
        self._vectors = df.iloc[:, 1: self.wv_dim + 1].values
        assert len(self._words) == len(self._vectors) == len(self._word2id)
        assert -1 <= self.default_oov <= 1

    def __getitem__(self, word):
        if not isinstance(word, str):
            raise TypeError('Expect a str object here.')
        return self._vectors[self._word2id[word]]

    def __contains__(self, word):
        return word in self._word2id

    def get_similarity(self, w1, w2):
        if w1 not in self or w2 not in self:
            if self.strict_wv:
                raise RuntimeError(f'OOV not allowed in strict wv mode.')
            else:
                return self.default_oov

        sim = 1.0 - cos_dist(self[w1], self[w2])
        return sim
