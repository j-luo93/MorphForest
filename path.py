from __future__ import division, print_function

from pair import ChildParentPair as Pair
import sys

class Path(object):

    def __init__(self, root):
        self.root = root
        self.parents = dict()
        self.types = dict()
        self.fringe = set([root])

    def get_fringe_word(self):
        assert len(self.fringe) > 0
        return next(iter(self.fringe))

    def expand(self, child, parent, type_):
        assert child in self.fringe
        self.fringe.remove(child)
        self.types[child] = type_
        if type_ in ['COM_LEFT', 'COM_RIGHT', 'HYPHEN']:
            assert type(parent) == tuple and len(parent) == 2, parent
            self.parents[child] = (parent[0], parent[1])
            self.fringe.add(parent[0])
            self.fringe.add(parent[1])
        else:
            self.parents[child] = parent
            if type_ != 'STOP':
                self.fringe.add(parent)

    def is_ended(self):
        return len(self.fringe) == 0

    def get_segmentation(self, mode='surface'):
        assert self.is_ended()
        assert mode in ['surface', 'canonical', 'label']
        if not hasattr(self, 'seg'):
            self.seg = self._segment_iter(self.root)
        return getattr(self.seg, mode)

    def _segment_iter(self, node):
        type_ = self.types[node]
        parent = self.parents[node]
        pair = Pair(node, parent, type_)
        if type_ == 'STOP': return Segment(node, node, 'stem')
        if type_ in ['COM_LEFT', 'COM_RIGHT', 'HYPHEN']:
            p1, p2 = self.parents[node]
            return self._segment_iter(p1).splice(self._segment_iter(p2))
        else:
            return self._segment_iter(parent).extend(pair)
            
    # The root is the concatenation of stems.
    def get_root(self, mode='surface'):
        assert mode in ['surface', 'canonical']
        assert self.is_ended()
        if not hasattr(self, 'seg'):
            self.seg = self._segment_iter(self.root)
        segs = getattr(self.seg, mode)
        return ''.join([x for x, y in filter(lambda item: item[0] if item[1] == 'stem' else '', zip(segs.split('-'), self.seg.label.split('-')))])

# Could have multipile morphemes
class Segment(object):

    # label can be a hyphenated list of 'stem', 'suf', or 'pre'
    def __init__(self, surface, canonical, label):
        self.surface = surface
        self.canonical = canonical
        self.label = label

    # update this segment by joining another. return updated self
    def splice(self, other):
        if self.is_empty(): return other
        self.surface += '-' + other.surface
        self.canonical += '-' + other.canonical
        self.label += '-' + other.label
        return self

    # extend current segment by pair
    def extend(self, pair):
        assert not self.is_empty()
        type_ = pair.type_
        affix, trans = pair.get_affix_and_transformation()
        if type_ == 'PREFIX':
            self.surface = affix + '-' + self.surface
            self.canonical = affix + '-' + self.canonical
            self.label = 'pre' + '-' + self.label
        else:
            self.canonical += '-' + affix
            self.label += '-' + 'suf'
            if type_ == 'SUFFIX' or type_ == 'APOSTR':
                self.surface += '-' + affix
            elif type_ == 'REPEAT':
                self.surface += pair.parent[-1] + '-' + affix
            elif type_ == 'MODIFY':
                self.surface = self.surface[:-1] + pair.child[len(pair.parent) - 1] + '-' + affix
            else:
                assert type_ == 'DELETE'
                if len(self.surface) > 1:
                    if self.surface[-2] == '-':
                        self.surface = self.surface[:-1] + affix
                    else:
                        self.surface = self.surface[:-1] + '-' + affix
        return self

    def is_empty(self):
        return len(self.surface) == 0
