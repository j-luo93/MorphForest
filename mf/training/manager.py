from arglib import use_arguments_as_properties
from mf.dataset.data_preparer import DataPreparer
from mf.models.feature_extractor import FeatureExtractor
from mf.models.log_linear import LogLinearModel

from .trainer import Trainer


@use_arguments_as_properties('supervised', 'lang', 'log_dir')
class Manager:

    def __init__(self):
        self.data_preparer = DataPreparer()
        self.feature_ext = FeatureExtractor(self.data_preparer)
        self.ll_model = LogLinearModel(self.feature_ext)
        self.trainer = Trainer(self.ll_model)

    def train(self):  # , reread=True):
        # if reread:
        #     self.data_preparer.read_all_data()
        # if not self.supervised:
        self.trainer.train(self.data_preparer)
        # NOTE Commented out supervised mode.
        # else:
        #     raise NotImplementedError('Not implemented. Should not have come here.')
        #     p, r, f = 0.0, 0.0, 0.0
        #     self.gold_segs_copy = dict(self.gold_segs)
        #     fold = random.random()
        #     for iter_ in range(5):
        #         clear_cache()
        #         print('###########################################')
        #         print('Iteration %d' % iter_)
        #         del self.gold_parents
        #         test_set = set(random.sample(self.gold_segs_copy.keys(), len(self.gold_segs_copy) // 5))
        #         self.gold_segs = {k: v for k, v in self.gold_segs_copy.iteritems() if k not in test_set}
        #         self.get_gold_parents()
        #         self.train_set = set(self.gold_parents.keys())
        #         w, loss = self._compute_loss()
        #         loss += self.reg_l2 * T.sum(w ** 2)
        #         optimizer = Adam()
        #         optimizer.run(w, loss)
        #         self.weights = w.get_value()
        #         # write weights to log
        #         with codecs.open(self.log_dir + 'MC.weights', 'w', 'utf8', errors='strict') as fout:
        #             for i, v in sorted(list(enumerate(self.weights)), key=itemgetter(1), reverse=True):
        #                 tmp = '%s\t%f\n' % (self.index2feature[i], v)
        #                 fout.write(tmp)
        #         print('###########################################')
        #         clear_cache()
        #         self.write_segments_to_file(wordset=test_set)
        #         p1, r1, f1 = self.evaluate()
        #         p += p1; r += r1; f += f1
        #     self.gold_segs = self.gold_segs_copy
        #     print(p / 5, r / 5, f / 5)
