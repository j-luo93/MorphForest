from abc import ABC, abstractmethod

from arglib import use_arguments_as_properties
from scipy.spatial.distance import cosine as cos_dist

class BaseWordVectors(ABC):

    @abstractmethod
    def get_similarity(self, w1, w2):
        pass


class DummyWordVectors(BaseWordVectors):

    def get_similarity(self, w1, w2):
        return 0.0


@use_arguments_as_properties('strict_wv', 'default_oov')
class WordVectors:

    def __init__(self, wv_path):
        # FIXME read first
        assert -1 <= self.default_oov <= 1
        

    def get_similarity(self, w1, w2):
        if w1 not in self.wv or w2 not in self.wv:
            if self.strict_wv:
                raise RuntimeError(f'OOV not allowed in strict wv mode.')
            else:
                return self.default_oov

        sim = 1.0 - cos_dist(self.wv[w1], self.wv[w2])
        return sim
