from gurobipy import *
from path import Path
from evaluate import evaluate
from pair import ChildParentPair as Pair
from collections import defaultdict
from itertools import permutations
import math
import sys
import codecs


class ILP(object):

    def __init__(self, base, alpha, beta):
        self.base = base
        self.alpha = alpha
        self.beta = beta

        self.pruner = {type_: set() for type_ in ['suf', 'pre', 'MODIFY', 'DELETE', 'REPEAT']}

        self.out_file = self.base.out_path + \
            'ILP.%s.%s.%s' % (self.base.lang, self.base.top_affixes, self.base.top_words)
        self.seeds = self.base.decompose(self.base.train_set)

        print('-----------------------------------', file=sys.stderr)
        print('alpha\t\t', self.alpha, file=sys.stderr)
        print('beta\t\t', self.beta, file=sys.stderr)
        print('out file\t', self.out_file, file=sys.stderr)
        print('-----------------------------------', file=sys.stderr)

    def build(self):
        self.model = Model('ILP')
        obj_coeffs = list()
        obj_vars = list()
        # X variables for choosing which candidate
        print('Setting up X variables...')
        for i, w in enumerate(self.fringe):
            print('\r%d/%d' % (i + 1, len(self.fringe)), end='')
            sys.stdout.flush()
            for cand in self.base.get_candidates(w):
                self.model.addVar(vtype=GRB.BINARY, name=self._get_name('x', w, cand=cand))
        print()
        self.model.update()

        # z variables for affixes and transformations
        print('Setting up Z variables...')
        for i, w in enumerate(self.fringe):
            print('\r%d/%d' % (i + 1, len(self.fringe)), end='')
            sys.stdout.flush()
            for cand in self.base.get_candidates(w):
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
        print()
        self.model.update()

        # constraints
        print('Setting up constraints...')
        for i, w in enumerate(self.fringe):
            print('\r%d/%d' % (i + 1, len(self.fringe)), end='')
            sys.stdout.flush()
            vars_ = list()
            for cand in self.base.get_candidates(w):
                name_var = self._get_name('x', w, cand=cand)
                var = self.model.getVarByName(name_var)
                assert var, name_var
                vars_.append(var)

                # average log-likelihood
                obj_vars.append(var)
                obj_coeffs.append(-1.0 / self.N * math.log(self.base.get_prob(w, cand)))
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
        print()
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

    def run(self):
        self.N = 0  # normalization factor
        self.model = None
        self.kept = defaultdict(set)
        n_iter = 0
        self.parents = dict()
        self.fringe = self.get_fringe()
        self.N = len(self.fringe)
        print('-------------------------------------')
        self.build()
        self.model.params.presolve = 2  # aggressive
        # self.model.params.presolve = 0  # no presolving
        self.model.optimize()
        self.kept = self.get_used_affixes_and_transformations()
        print('Affixes kept:')
        print(dict([(key, len(self.kept[key])) for key in self.kept]))
        self.parents = self._get_parents_for_fringe()
        n_iter += 1
        print('-------------------------------------')
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
        for p in self.base.prefixes:
            name = 'z_pre_' + p
            v = self.model.getVarByName(name)
            if v and v.x == 1.0: kept['pre'].add(p)
        for s in self.base.suffixes:
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
        all_['pre'] = set(self.base.prefixes)
        all_['suf'] = set(self.base.suffixes)
        for key in all_:
            self.pruner[key].update(all_[key] - kept[key])
        print('Affixes pruned:')
        print(dict([(key, len(self.pruner[key])) for key in self.pruner]))

    # fall back to the base model whenever it is not available
    def predict(self, child, fall_back=True):
        if child not in self.parents:
            if fall_back:
                return self.base.predict(child)
            else:
                return None
        return self.parents[child]

    def get_parent(self, child):
        for cand in self.base.get_candidates(child):
            name = self._get_name('x', child, cand=cand)
            v = self.model.getVarByName(name)
            assert v, name
            if v.x == 1.0:
                return cand
        raise

    def get_raw_features(self, child, candidate):
        return self.base.get_raw_features(child, candidate)

    def get_seg_path(self, w):
        path = Path(w)
        while not path.is_ended():
            child = path.get_fringe_word()
            parts = child.split("'")
            if len(parts) == 2 and len(parts[0]) > 0 and self.base.lang == 'en':
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
        if not wordset: wordset = set(self.base.gold_segs.keys())
        if not out_file: out_file = self.out_file
        with codecs.open(out_file, 'w', 'utf8', errors='strict') as fout:
            for w in wordset:
                path = self.get_seg_path(w)
                fout.write(w + ':' + path.get_segmentation() + '\n')

    def evaluate(self):
        p, r, f = evaluate(self.base.gold_segs_file, self.out_file, quiet=True)
        print('ILP: precision =', p, 'recall =', r, 'f =', f, file=sys.stderr)
