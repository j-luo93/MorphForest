
class FeatureExtractor:

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