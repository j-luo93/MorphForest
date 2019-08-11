import logging
import os
import time
from pathlib import Path

import enlighten
import torch
from torch.optim import Adam

from arglib import use_arguments_as_properties
from dev_misc import Metric, Metrics, clear_cache, get_tensor, log_pp
from mf.utils.evaluate import evaluate

_manager = enlighten.Manager()


@use_arguments_as_properties('max_epoch', 'learning_rate', 'iteration', 'ILP', 'reg_hyper', 'check_interval', 'log_dir')
class Trainer:

    def __init__(self, ll_model):
        self.ll_model = ll_model
        self._pbar = _manager.counter(desc='epoch', total=self.max_epoch)

    @property
    def epoch(self):
        return self._pbar.count

    def _start_iteration(self):
        self._pbar.count = 0
        self._pbar.start = time.time()

    def _ended_iteration(self):
        return self._pbar.count >= self.max_epoch

    def train(self, dataset):
        if self.ILP:
            for i in range(self.iteration):
                self._train_one_iteration(dataset)
        else:
            self._train_one_iteration(dataset)

    def _train_one_iteration(self, dataset):
        # NOTE Get a batch. This batch would be the same across all epochs.
        batch = dataset.get_batch()
        # Clear cache first.
        clear_cache()
        # First pass to gather dataset-specific information and prepare weights.
        self.ll_model.first_pass(batch)
        # Move everything to cuda if specified.
        if os.environ.get('CUDA_VISIBLE_DEVICES', False):
            self.ll_model.cuda()
            batch.apply(get_tensor, ignore_all=True)
        # Get a new optimizer for each iteration.
        optimizer = Adam(self.ll_model.parameters(), lr=self.learning_rate)
        # Start the iteration.
        self._start_iteration()
        # Main body.
        metrics = Metrics()
        while not self._ended_iteration():
            epoch_metrics, should_stop = self._train_one_epoch(batch, optimizer)
            metrics += epoch_metrics
            self._pbar.update()
            if self.epoch % self.check_interval == 0:
                self._do_check(metrics)

            if should_stop:
                break
        self.save(dataset)

    def _train_one_epoch(self, batch, optimizer):
        """Each epoch is actually just one step since no minibatching is used."""
        # Prepare.
        self.ll_model.zero_grad()
        self.ll_model.train()
        optimizer.zero_grad()

        # Forward pass. I chose to not use `weight_decay` provided by `torch.optim` to get a more explicit control of l2 regularization.
        metrics = self.ll_model(batch)
        loss = metrics.nll.total + self.reg_hyper * metrics.reg_l2.total
        loss = Metric('loss', loss, 1.0)

        loss.total.backward()
        grad_norm = self.ll_model.weights.grad.norm(2)
        grad_norm = Metric('grad_norm', grad_norm, 1.0)
        optimizer.step()
        metrics += Metrics(grad_norm, loss)
        should_stop = grad_norm.total < 1.0
        return metrics, should_stop

    def _do_check(self, metrics):
        log_pp(metrics.get_table(title=f'Epoch = {self.epoch}'))
        metrics.clear()

    def save(self, dataset):
        torch.save(self.ll_model.state_dict(), f'{self.log_dir}/saved.latest')
        # self.weights = w.get_value()
        # write weights to log
        self.write_weights()
        self.write_segments_to_file(dataset)
        p, r, f = self.evaluate(dataset)
        print(p, r, f)

    def write_weights(self):
        with Path(f'{self.log_dir}/MC.weights').open('w', encoding='utf8') as fout:
            weights = self.ll_model.weights.cpu()
            for i, v in sorted(enumerate(weights), key=lambda x: x[1], reverse=True):
                tmp = '%s\t%f\n' % (self.ll_model.index2feature[i], v)
                fout.write(tmp)

    def write_segments_to_file(self, dataset, wordset=None, out_file=None):
        if not wordset:
            wordset = set(dataset.gold_segs.keys())
        if not out_file:
            out_file = dataset.predicted_file['train']
        with Path(out_file).open('w', encoding='utf8') as fout:
            for word in sorted(wordset):
                fout.write(word + ':' + self.ll_model.segment(word) + '\n')

    def evaluate(self, dataset):
        p, r, f = evaluate(dataset.gold_segs_file, dataset.predicted_file['train'], quiet=True)
        logging.info(f'MC: p/r/f = {p}/{r}/{f}')
        return (p, r, f)

    def update_pruner(self, pruner):
        # TODO for ILP
        self.pruner = pruner
        for p in pruner['pre']:
            if p in self.mc_model.prefixes:
                self.mc_model.prefixes.remove(p)
        for s in pruner['suf']:
            if s in self.mc_model.suffixes:
                self.mc_model.suffixes.remove(s)
