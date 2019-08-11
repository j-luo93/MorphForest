import logging
import math
import sys
from collections import defaultdict
from itertools import permutations

from gurobipy import *
from tqdm import tqdm

from arglib import use_arguments_as_properties
from dev_misc import log_this
from mf.utils.evaluate import evaluate
from mf.utils.pair import ChildParentPair as Pair
from mf.utils.path import Path


@use_arguments_as_properties('alpha', 'beta', 'lang', 'log_dir', 'top_affixes', 'top_words')
class ILP(object):

    def __init__(self, ll_model, feature_ext, dataset):
        self.ll_model = ll_model
        self.feature_ext = feature_ext
        self.dataset = dataset

        self.pruner = {type_: set() for type_ in ['suf', 'pre', 'MODIFY', 'DELETE', 'REPEAT']}

        self.out_file = self.log_dir + \
            '/ILP.%s.%s.%s' % (self.lang, self.top_affixes, self.top_words)
        self.seeds = self.decompose(self.dataset.train_set)

    def decompose(self, s):
        if self.lang == 'en':
            new_set = set()
            for w in s:
                parts = w.split("'")
                if len(parts) == 2:
                    w = parts[0]
                if '-' in w:
                    new_set.update(w.split('-'))
                else:
                    new_set.add(w)
            return new_set
        return s

    @log_this('INFO')
    def build(self):
        self.model = Model('ILP')
        obj_coeffs = list()
        obj_vars = list()
        # X variables for choosing which candidate
        logging.info('Setting up X variables...')
        for w in tqdm(self.fringe):
            for cand in self.feature_ext.get_candidates(w):
                self.model.addVar(vtype=GRB.BINARY, name=self._get_name('x', w, cand=cand))
        self.model.update()

        # z variables for affixes and transformations
        logging.info('Setting up Z variables...')
        for w in tqdm(self.fringe):
            for cand in self.feature_ext.get_candidates(w):
                if cand[1] == 'STOP': continue
                parent, type_ = cand
                pair = Pair(w, parent, type_)
                affix, trans = pair.get_affix_and_transformation()
                if affix:
                    name = self._get_name('z', pair.type_coarse, affix)
                    v = self.model.getVarByName(name)
                    if not v:
                        z = self.model.addVar(vtype=GRB.BINARY, name=name)
                        assert z
                        obj_coeffs.append(self.alpha)
                        obj_vars.append(z)
                if trans:
                    name = 'z_' + trans
                    name = self._get_name('z', trans)
                    v = self.model.getVarByName(name)
                    if not v:
                        z = self.model.addVar(vtype=GRB.BINARY, name=name)
                        assert z
                        obj_coeffs.append(self.alpha)
                        obj_vars.append(z)
        self.model.update()

        # constraints
        logging.info('Setting up constraints...')
        for w in tqdm(self.fringe):
            vars_ = list()
            for cand in self.feature_ext.get_candidates(w):
                name_var = self._get_name('x', w, cand=cand)
                var = self.model.getVarByName(name_var)
                assert var, name_var
                vars_.append(var)

                # average log-likelihood
                obj_vars.append(var)
                obj_coeffs.append(-1.0 / self.N * math.log(self.ll_model.get_prob(w, cand)))
                # average tree size
                if cand[1] == 'STOP':
                    obj_vars.append(var)
                    obj_coeffs.append(self.beta / self.N)
                    continue

                parent, type_ = cand
                pair = Pair(w, parent, type_)
                affix, trans = pair.get_affix_and_transformation()
                if affix:
                    name = self._get_name('z', pair.type_coarse, affix)
                    v = self.model.getVarByName(name)
                    assert var and v, (name_var, name)
                    self.model.addConstr(var <= v, name=self._get_name('c', name_var, name))
                if trans:
                    name = self._get_name('z', trans)
                    v = self.model.getVarByName(name)
                    assert var and v, (name_var, name)
                    self.model.addConstr(var <= v, name=self._get_name('c', name_var, name))
            self.model.addConstr(LinExpr([1.0] * len(vars_), vars_) == 1, name=self._get_name('c', w))
        self.model.update()
        self.model.setObjective(LinExpr(obj_coeffs, obj_vars), GRB.MINIMIZE)
        self.model.update()

    def _get_name(self, prefix, *args, **kwargs):
        name = prefix + '_' + '_'.join(args)
        if 'cand' not in kwargs:
            if len(name) > 200:
                name = str(hash(name))
            return name

        parent, type_ = kwargs['cand']
        name += '_' + type_
        if type(parent) == tuple:
            assert len(parent) == 2
            name += '_' + parent[0] + '_' + parent[1]
        else:
            name += '_' + parent
        if len(name) > 200:
            name = str(hash(name))
        return name

    @log_this('INFO')
    def run(self):
        self.N = 0  # normalization factor
        self.model = None
        self.kept = defaultdict(set)
        n_iter = 0
        self.parents = dict()
        self.fringe = self.get_fringe()
        self.N = len(self.fringe)
        self.build()
        # HACK Ignore messages coming from the solver -- they are propagated back to the root logger anyway.
        logging.getLogger("gurobipy.gurobipy").setLevel(logging.CRITICAL + 1)
        self.model.params.presolve = 2  # aggressive
        # self.model.params.presolve = 0  # no presolving
        self.model.optimize()
        self.kept = self.get_used_affixes_and_transformations()
        kept_affixes = dict([(key, len(self.kept[key])) for key in self.kept])
        logging.info(f'Affixes kept: {kept_affixes}')
        self.parents = self._get_parents_for_fringe()
        n_iter += 1
        self.update_pruner(self.kept)

    def _get_parents_for_fringe(self):
        parents = dict()
        for w in self.fringe:
            parents[w] = self.get_parent(w)
        return parents

    def get_fringe(self):
        if not self.model: return set(self.seeds)
        assert self.model.status == GRB.OPTIMAL, self.model.status
        queue = set(self.seeds)
        fringe = set(self.seeds)
        while queue:
            word = queue.pop()
            if word in self.parents:
                parent, type_ = self.parents[word]
            else: continue
            if type_ == 'STOP': continue
            if type_ in ['COM_LEFT', 'COM_RIGHT']:
                fringe.update(parent)
                queue.update(parent)
            else:
                fringe.add(parent)
                queue.add(parent)
        return fringe

    def get_used_affixes_and_transformations(self):
        kept = defaultdict(set)
        for p in self.dataset.prefixes:
            name = 'z_pre_' + p
            v = self.model.getVarByName(name)
            if v and v.x == 1.0: kept['pre'].add(p)
        for s in self.dataset.suffixes:
            name = 'z_suf_' + s
            v = self.model.getVarByName(name)
            if v and v.x == 1.0: kept['suf'].add(s)
        for v in self.model.getVars():
            name = v.varName
            if name[0] != 'z': continue
            if name[2] == 's' or name[2] == 'p': continue
            code = name[2:5]
            trans = name[2:]
            if v.x == 1.0:
                if code == 'MOD': kept['MODIFY'].add(trans)
                if code == 'REP': kept['REPEAT'].add(trans)
                if code == 'DEL': kept['DELETE'].add(trans)
        return kept

    def update_pruner(self, kept):
        all_ = defaultdict(set)
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        all_['REPEAT'] = set(['REP_%s' % s for s in alphabet])
        all_['DELETE'] = set(['DEL_%s' % s for s in alphabet])
        for c1, c2 in permutations(alphabet, r=2):
            all_['MODIFY'].add('MOD_%s_%s' % (c1, c2))
        all_['pre'] = set(self.dataset.prefixes)
        all_['suf'] = set(self.dataset.suffixes)
        for key in all_:
            self.pruner[key].update(all_[key] - kept[key])
        pruned_affixes = dict([(key, len(self.pruner[key])) for key in self.pruner])
        logging.info(f'Affixes pruned: {pruned_affixes}')

    # fall back to the base model whenever it is not available
    def predict(self, child, fall_back=True):
        if child not in self.parents:
            if fall_back:
                return self.ll_model.predict(child)
            else:
                return None
        return self.parents[child]

    def get_parent(self, child):
        for cand in self.feature_ext.get_candidates(child):
            name = self._get_name('x', child, cand=cand)
            v = self.model.getVarByName(name)
            assert v, name
            if v.x == 1.0:
                return cand
        raise RuntimeError(f'Cannot find the parent for {child}.')

    def get_raw_features(self, child, candidate):
        return self.feature_ext.get_raw_features(child, candidate)

    def get_seg_path(self, w):
        path = Path(w)
        while not path.is_ended():
            child = path.get_fringe_word()
            parts = child.split("'")
            if len(parts) == 2 and len(parts[0]) > 0 and self.lang == 'en':
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

    def parse(self, wordset=None, out_file=None):
        if not wordset: wordset = set(self.dataset.gold_segs.keys())
        if not out_file: out_file = self.out_file
        with open(out_file, 'w', encoding='utf8') as fout:
            for w in tqdm(sorted(wordset)):
                path = self.get_seg_path(w)
                fout.write(w + ':' + path.get_segmentation() + '\n')

    def evaluate(self):
        p, r, f = evaluate(self.dataset.gold_segs_file, self.out_file, quiet=True)
        logging.info(f'p/r/f = {p}/{r}/{f}')
