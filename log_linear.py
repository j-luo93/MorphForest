import torch.nn as nn


class LogLinearModel(nn.Module):

    def _compute_loss(self):
        # first pass to know the how many paddings we need
        max_r_num, max_r_den = 0, 0
        print('First pass to gather info...')
        for word_i, word in enumerate(self.train_set):
            print('\r%d/%d' % (word_i + 1, len(self.train_set)), end='')
            sys.stdout.flush()
            # if self.supervised and word not in self.gold_parents: continue
            acc = 0
            for neighbor in self.get_neighbors(word, expand=False):
                acc += len(self.get_candidates(neighbor, gold=False))  # , top=top))
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
            print('\r%d/%d' % (word_i + 1, len(self.train_set)), end='')
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
