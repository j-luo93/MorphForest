import csv
from abc import ABC, abstractmethod

import pandas as pd
from arglib import use_arguments_as_properties
from scipy.spatial.distance import cosine as cos_dist


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
        df = pd.read_csv(path, skiprows=1, sep=' ', encoding='utf8',
                         keep_default_na=False, quoting=csv.QUOTE_NONE, header=None)
        self._words = df[0].values
        self._word2id = {word: idx for idx, word in enumerate(self._words)}
        self._vectors = df.iloc[1: self.wv_dim + 1].values
        assert -1 <= self.default_oov <= 1

    def __getitem__(self, word):
        if not isinstance(word, str):
            raise TypeError('Expect a str object here.')
        return self._vectors[self._word2id[word]]

    def get_similarity(self, w1, w2):
        if w1 not in self.wv or w2 not in self.wv:
            if self.strict_wv:
                raise RuntimeError(f'OOV not allowed in strict wv mode.')
            else:
                return self.default_oov

        sim = 1.0 - cos_dist(self.wv[w1], self.wv[w2])
        return sim
