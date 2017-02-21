from __future__ import division
import sys
import heapq
import math
from collections import Counter

class Logger(object):

    def __init__(self, log_file):
        self.terminal = sys.stderr
        self.log = open(log_file, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class SmartCounter(Counter):

    def __init__(self, *args, **kwargs):
        super(SmartCounter, self).__init__(*args, **kwargs)
        self.total = 0
        self.entropy = 0.0

    def __setitem__(self, key, value):
        if key is None or type(key) == tuple and key[0] is None: return  # can't be keyed by None
        c_j = self[key]
        s = self.total
        diff = value - c_j
        # print(diff)
        # import pdb; pdb.set_trace()
        # t1 = safe_div(s, (s + diff)) * (-self.entropy - safe_div(c_j, s) * safe_log(safe_div(c_j, s)))
        # t2 = (s - c_j) * safe_div(safe_log(safe_div(s, (s + diff))), (s + diff))
        # t3 = safe_div((c_j + diff), (s + diff)) * safe_log(safe_div((c_j + diff), (s + diff)))
        self.total += diff
        # self.entropy = -(t1 + t2 + t3)
        super(SmartCounter, self).__setitem__(key, value)
        if self[key] == 0: del self[key]

    def __delitem__(self, key):
        if key not in self: return
        if self[key] != 0: raise Exception
        super(SmartCounter, self).__delitem__(key)

def safe_log(value):
    if value == 0.0: return 0.0
    return math.log(value)

def safe_div(a, b):
    if b == 0.0: return 0.0
    return a / b

def update_memory(entry, memory, beam_size):
    assert len(memory) <= beam_size
    heapq.heappush(memory, entry)
    if len(memory) > beam_size:
        new_memory = list()
        while len(new_memory) < beam_size:
            heapq.heappush(new_memory, heapq.heappop(memory))
        memory = new_memory
    return memory

affix_cache = dict()
def get_affix(child, candidate):
    global affix_cache
    key = (child, candidate)
    if key in affix_cache: return affix_cache[key]
    parent, type_ = candidate
    if type_ == 'PREFIX':
        affix = child[:len(child) - len(parent)]
    elif type_ == 'SUFFIX':
        affix = child[len(parent):]
    elif type_ == 'MODIFY':
        affix = child[len(parent):]
    elif type_ == 'DELETE':
        affix = child[len(parent) - 1:]
    elif type_ == 'REPEAT':
        assert child[len(parent)] == child[len(parent) - 1], child + '\t' + parent
        affix = child[len(parent) + 1:]
    # elif type_ == 'COMPOUND':
    #     return (None, 'COMPOUND')
    # elif type(type_) == tuple:
    #     assert type_[1] == 'COMPOUND'
    #     return (None, 'COMPOUND')
    elif type_ == 'COM_LEFT' or type_ == 'COM_RIGHT':
        return (None, type_)
    else:
        assert False, child + '\t' + str(candidate)

    if type_ == 'PREFIX':
        affix_cache[key] = (affix, 'p')
        return (affix, 'p')
    else:
        affix_cache[key] = (affix, 's')
        return (affix, 's')

def get_transform(child, candidate):
    parent, type_ = candidate
    if type_ == 'SUFFIX' or type_ == 'PREFIX': return None
    if type_ == 'MODIFY':
        return (parent[-1] + '->' + child[len(parent) - 1], type_)
    elif type_ == 'DELETE':
        return (parent[-1], type_)
    elif type_ == 'REPEAT':
        return (parent[-1], type_)
    elif type_ == 'COM_LEFT' or type_ == 'COM_RIGHT':
        return (None, type_)
    # elif type_ == 'COMPOUND':
    #     return (None, 'COMPOUND')
    # elif type(type_) == tuple:
    #     assert type_[1] == 'COMPOUND'
    #     return (None, 'COMPOUND')
    assert False
    
"""UnionFind.py

Union-find data structure. Based on Josiah Carlson's code,
http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
with significant additional changes by D. Eppstein.
"""

class UnionFind:
    """Union-find data structure.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.
    """

    def __init__(self):
        """Create a new empty union-find structure."""
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""

        # check for previously unknown object
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root
        
    def __iter__(self):
        """Iterate through all items ever found or unioned by this structure."""
        return iter(self.parents)

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r],r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest
