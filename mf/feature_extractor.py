from dev_misc import cache
import math

from .pair import ChildParentPair as Pair

from arglib import use_arguments_as_properties

EN_ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


# TODO have to deal with pruner -- saving pruner here?
@use_arguments_as_properties('sibling', 'compounding', 'lang', 'use_word_vectors')
class FeatureExtractor:

    def __init__(self, dataset):
        self.dataset = dataset
        self.pruner = None

    @cache(persist=False, full=True)
    def get_raw_features(self, child, candidate):
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
            features['LEN_%d' % (len(parent))] = 1.0
            max_cos = self._get_max_cos(parent)
            if max_cos > -2:    # -2 is exit error code
                features['MAXCOS_%d' % (int(max_cos * 10))] = 1.0
        else:
            if type_ != 'COM_LEFT' and type_ != 'COM_RIGHT':
                cos = self.dataset.wv.get_similarity(child, parent)
                features['COS'] = cos
                if parent in self.dataset.word_cnt:
                    features['CNT'] = math.log(self.dataset.word_cnt[parent])
                    features['IV'] = 1.0
                else:
                    features['OOV'] = 1.0
            affix, trans = pair.get_affix_and_transformation()
            if self.sibling:
                if pair.type_coarse == 'suf' and affix in self.dataset.suffixes or pair.type_coarse == 'pre' and affix in self.dataset.prefixes:
                    self._get_sibling_feature(pair, features)
            if type_ == 'PREFIX':
                if affix in self.dataset.prefixes: features['PRE_' + affix] = 1.0
            elif type_ == 'SUFFIX' or type_ == 'APOSTR':
                if affix in self.dataset.suffixes: features['SUF_' + affix] = 1.0
            elif type_ == 'MODIFY':
                if affix in self.dataset.suffixes: features['SUF_' + affix] = 1.0
                if not self.pruner or trans not in self.pruner['MODIFY']:
                    features[trans] = 1.0
            elif type_ == 'DELETE':
                if affix in self.dataset.suffixes:
                    features['SUF_' + affix] = 1.0
                if not self.pruner or trans not in self.pruner['DELETE']:
                    features[trans] = 1.0
            elif type_ == 'REPEAT':
                if affix in self.dataset.suffixes:
                    features['SUF_' + affix] = 1.0
                if not self.pruner or trans not in self.pruner['REPEAT']:
                    features[trans] = 1.0
            elif type_ == 'COM_LEFT':
                parent, aux = parent
                features['HEAD_CNT'] = math.log(self.dataset.word_cnt[parent])
                features['HEAD_COS'] = self.dataset.wv.get_similarity(child, parent)
                features['AUX_CNT'] = math.log(self.dataset.word_cnt[aux])
                features['AUX_COS'] = self.dataset.wv.get_similarity(child, aux)
            elif type_ == 'COM_RIGHT' or type_ == 'HYPHEN':
                aux, parent = parent
                features['HEAD_CNT'] = math.log(self.dataset.word_cnt[parent])
                features['HEAD_COS'] = self.dataset.wv.get_similarity(child, parent)
                features['AUX_CNT'] = math.log(self.dataset.word_cnt[aux])
                features['AUX_COS'] = self.dataset.wv.get_similarity(child, aux)
            else:
                raise NotImplementedError('no such type %s' % (type_))
        return features

    @cache(persist=False, full=True)
    def _get_max_cos(self, child):
        if self.use_word_vectors:
            max_cos = max([-2] + [self.dataset.wv.get_similarity(child, parent)
                                  for parent, type_ in self.get_candidates(child) if type_ != 'STOP'])  # -2 as exit code
            return max_cos
        else:
            return -2

    def _get_sibling_feature(self, pair, features, top_K=1000000):
        neighbors = self.dataset.prefixes if pair.type_coarse == 'pre' else self.dataset.suffixes
        name = 'COR_' + ('P_' if pair.type_coarse == 'pre' else 'S_') + pair.get_affix()
        cnt = 0
        for neighbor in neighbors:
            if pair.affix != neighbor:
                if self._get_surface_form(pair, neighbor) in self.dataset.word_cnt:
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

    @cache(persist=False, full=True)  # NOTE `persist` is set to False because some affixes might be pruned.
    def get_candidates(self, word, gold=False):
        candidates = set()
        # NOTE Commented out supervised mode.
        # if self.supervised:
        #     if word in self.gold_parents:
        #         candidates.add(self.gold_parents[word])
        #     if gold:
        #         return candidates

        candidates.add((word, 'STOP'))
        if len(word) < 3:
            return candidates
        for pos in range(1, len(word)):
            parent = word[:pos]
            if self.compounding and parent in self.dataset.word_cnt and word[pos:] in self.dataset.word_cnt:
                if self.dataset.word_cnt[parent] >= self.dataset.freq_thresh and self.dataset.word_cnt[word[pos:]] >= self.dataset.freq_thresh:
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
                            if self.dataset.wv.get_similarity(new_parent, word) > 0.2:
                                pair = Pair(word, new_parent, 'MODIFY')
                                suf, trans = pair.get_affix_and_transformation()
                                if not self.pruner or suf not in self.pruner['suf']:
                                    if not self.pruner or trans not in self.pruner['MODIFY']:
                                        candidates.add((new_parent, 'MODIFY'))
                    if pos < len(word) - 1 and word[pos:] in self.dataset.suffixes:
                        for char in EN_ALPHABET:
                            new_parent = parent + char
                            if word == new_parent: continue
                            if new_parent in self.dataset.word_cnt:
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
        # NOTE Commented out supervised mode.
        # if not expand and self.supervised:  # only activated if it's in suprvised mode
        #     return set([word])

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
