from pathlib import Path

from evaluate import evaluate
from dev_misc import clear_cache


class Trainer:

    def __init__(self, mc_model):
        self.mc_model = mc_model

    def train(self):
        # FIXME
        clear_cache()
        w, loss = self._compute_loss()
        loss += self.reg_l2 * T.sum(w ** 2)
        optimizer = Adam()
        optimizer.run(w, loss)
        self.weights = w.get_value()
        # write weights to log
        self.write_weights()
        clear_cache()
        self.write_segments_to_file()
        p, r, f = self.evaluate()
        print(p, r, f)

    def write_weights(self):
        with codecs.open(self.mc_model.log_dir + 'MC.weights', 'w', 'utf8', errors='strict') as fout:
            for i, v in sorted(list(enumerate(self.weights)), key=itemgetter(1), reverse=True):
                tmp = '%s\t%f\n' % (self.index2feature[i], v)
                fout.write(tmp)

    def write_segments_to_file(self, wordset=None, out_file=None):
        if not wordset:
            wordset = set(self.mc_model.gold_segs.keys())
        if not out_file:
            out_file = self.mc_model.predicted_file['train']
        with Path(out_file).open('w', encoding='utf8') as fout:
            for word in enumerate(wordset):
                fout.write(word + ':' + self.mc_model.segment(word) + '\n')  # FIXME segment method shouldn't be in ll

    def evaluate(self):
        p, r, f = evaluate(self.mc_model.gold_segs_file, self.mc_model.predicted_file['train'], quiet=True)
        logging.info(f'MC: p/r/f = {p}/{r}/{f}')
        return (p, r, f)

    def update_pruner(self, pruner):
        # FIXME
        self.pruner = pruner
        for p in pruner['pre']:
            if p in self.mc_model.prefixes:
                self.mc_model.prefixes.remove(p)
        for s in pruner['suf']:
            if s in self.mc_model.suffixes:
                self.mc_model.suffixes.remove(s)
