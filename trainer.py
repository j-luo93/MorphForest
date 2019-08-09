from pathlib import Path

import torch
from enlighten import Counter
from torch.optim import Adam

from arglib import use_arguments_as_properties
from dev_misc import Metric, Metrics, clear_cache, log_pp
from evaluate import evaluate


@use_arguments_as_properties('max_epoch', 'learning_rate', 'iteration', 'ILP', 'reg_hyper', 'check_interval')
class Trainer:

    def __init__(self, mc_model, ll_model):
        self.mc_model = mc_model
        self.ll_model = ll_model

    @property
    def epoch(self):
        return self._pbar.count

    def _start_iteration(self):
        self._pbar = Counter(desc='epoch', total=self.max_epoch)

    def _ended_iteration(self):
        return self._pbar.count >= self.max_epoch

    def train(self, dataset):
        if self.ILP:
            for i in range(self.iteration):
                self._train_one_iteration(dataset)
        else:
            self._train_one_iteration(dataset)

    def _train_one_iteration(self, dataset):
        # Clear cache first.
        clear_cache()
        # Get a new optimizer for each iteration.
        optimizer = Adam(self.ll_model.params(), lr=self.learning_rate)
        # Start the iteration.
        self._start_iteration
        # Main body.
        metrics = Metrics()
        while not self._ended_iteration():
            epoch_metrics = self._train_one_epoch(dataset, optimizer)
            metrics += epoch_metrics
            self._pbar.update()
            if self.epoch % self.check_interval == 0:
                self._do_check(metrics)
        self.save()

    def _train_one_epoch(self, dataset, optimizer):
        """Each epoch is actually just one step since no minibatching is used."""
        # Prepare.
        self.ll_model.zero_grad()
        self.ll_model.train()
        optimizer.zero_grad()

        # Forward pass. I chose to not use `weight_decay` provided by `torch.optim` to get a more explicit control of l2 regularization.
        batch = dataset.get_batch()
        metrics = self.ll_model(batch)
        loss = metrics.nll + self.reg_hyper * metrics.reg_l2

        loss.mean.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.params(), 5.0)
        grad_norm = Metric('grad_norm', grad_norm * batch.num_samples, batch.num_samples)
        optimizer.step()
        metrics += Metrics(grad_norm, loss)
        return metrics

    def _do_check(self, metrics):
        log_pp(metrics.get_table())

    # FIXME This is weird
    def save(self):
        self.weights = w.get_value()
        # write weights to log
        self.write_weights()
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
