from __future__ import division, print_function

from model import MC
from ILP import ILP
import cPickle
import random
import sys
from pair import ChildParentPair as Pair
from itertools import izip
from collections  import defaultdict, Counter

import argparse

from scipy.sparse import csc_matrix
import numpy as np
import theano
from theano import tensor as T
from lasagne import layers
from lasagne.regularization import l2, regularize_network_params
import lasagne

################################################################################
def attach_tag_to_param(tag, param_name, layer):
    assert hasattr(layer, param_name)
    param = getattr(layer, param_name)
    layer.params[param].add(tag)
    
def iterate_minibatches(x, y, bs):
    assert x.shape[0] == y.shape[0]
    indices = range(x.shape[0])
    random.shuffle(indices)
    r = len(indices) % bs
    if r > 0: indices = indices[:-r]
    for i in xrange(len(indices) // bs):
        s, t = i * bs, (i + 1) * bs
        yield i, x[s:t, :].todense(), y[s:t, :].todense()
        
def get_dataset(model, stats, order=1):
    vocab = stats['vocab']
    attributes = stats['attributes']
    vocab_split = stats['vocab_split']
    cnt_split = stats['cnt_split']
    if type(model) == ILP:
        if args.keep_train_set:
            model.seeds.update(vocab.keys())
        else:
            model.seeds = set(vocab.keys())
        model.run()
    # assert order == 1
    # model.pruner = None	# -imp -jl
    f2i, a2i = dict(), dict()
    fd_list = list()
    al_list = list()
    w2i = dict(zip(vocab.keys(), range(len(vocab))))
    print('Preparing data...')
    cnt = Counter()
    for i, word in enumerate(vocab.keys()):
    	print('\r%d/%d' %(i + 1, len(vocab)), end='')
    	sys.stdout.flush()
        path = model.get_seg_path(word)
        queue = set([(word, 0)])
        all_fd = defaultdict(lambda: 0.0)
        while queue:
            form, dist = queue.pop()
            parent, type_ = path.parents[form], path.types[form]
            fd = model.get_raw_features(form, (parent, type_))
            pair = Pair(form, parent, type_)
            affix = pair.get_affix()
            if pair.type_coarse == 'pre' and 'PRE_%s' %affix not in fd: fd['PRE_%s' %affix] = 1.0
            if pair.type_coarse == 'suf' and 'SUF_%s' %affix not in fd: 
                import pdb; pdb.set_trace()
                fd['SUF_%s' %affix] = 1.0
            for k, v in fd.iteritems():
                if k[:3] in ['SUF', 'PRE']: 
                    all_fd[str(dist) + '_' + k] += v
                    cnt[k] += 1
            all_fd[str(dist) + '_' + type_] += 1.0
            if dist < order - 1:
                dist += 1
                if type(parent) == tuple:
                    queue.add((parent[0], dist))
                    queue.add((parent[1], dist))
                else: queue.add((parent, dist))
            fd_list.append(all_fd)
            al = list()
            for attr in vocab[word]:
                for value in vocab[word][attr]:
                    name = '%s=%s' %(attr, value)
                    al.append(name) 
                    if name not in a2i: a2i[name] = len(a2i)
            al_list.append(al)
    print("Removing nonce affixes")
    to_remove = set([x for x, y in filter(lambda item: item[1] > 1, cnt.items())])
    v_x, r_x, c_x = list(), list(), list()
    v_y, r_y, c_y = list(), list(), list()
    for fd, al in izip(fd_list, al_list):
        for k, v in fd.iteritems():
            name = '_'.join(k.split('_')[1:])
            if name in to_remove: continue
            if k not in f2i: f2i[k] = len(f2i)
            v_x.append(v)
            r_x.append(i)
            c_x.append(f2i[k])
        for a in al:
	        v_y.append(1)
	        r_y.append(i)
	        c_y.append(a2i[a])
    print('\nInserting data...')
    data_x = csc_matrix((v_x, (r_x, c_x)), shape=(len(vocab), len(f2i)), dtype='float32')
    data_y = csc_matrix((v_y, (r_y, c_y)), shape=(len(vocab), len(a2i)), dtype='int8')
    
    if args.remove_nonce:
	trim = list()
	for w in vocab_split['train']:
	    if cnt_split['train'][w] > 1:
		trim.append(w)
    else:
	trim = vocab.keys()
    indices_train = map(lambda x: w2i[x], trim)
    indices_dev = map(lambda x: w2i[x], vocab_split['dev'])
    indices_test = map(lambda x: w2i[x], vocab_split['test'])
    return data_x[indices_train, :], data_x[indices_dev, :], data_x[indices_test, :], data_y[indices_train, :], data_y[indices_dev, :], data_y[indices_test, :], f2i, a2i
        
        
################################################################################

random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--stats-file', '-S', dest='stats_file', metavar='', required=True, help='stats file to real')
parser.add_argument('--load', '-l', metavar='', required=True, help='pickled model file to load')
parser.add_argument('-order', '-o', metavar='', default=1, type=int, help='order for path')
parser.add_argument('-keep-train-set', dest='keep_train_set', action='store_true', help='whether to keep the train set in the ILP when segmenting new word set')
parser.add_argument('-remove-nonce', dest='remove_nonce', action='store_true', help='whether to remove nonces')
args = parser.parse_args()

print('Loading stats...', end='')
sys.stdout.flush()
stats = cPickle.load(open(args.stats_file, 'r'))
print('Done\nLoading model...', end='')
sys.stdout.flush()
model = cPickle.load(open(args.load, 'r'))
print('Done')

x_train, x_dev, x_test, y_train, y_dev, y_test, f2i, a2i = get_dataset(model, stats, order=args.order)

bs = 100
n_hid = 50
l2_reg = 1e-3
lr = 1e-3
n_epoch = 200

x = T.fmatrix('x')
y = T.imatrix('y')
input_ = layers.InputLayer((None, len(f2i)), input_var=x, name='input')
#hidden = layers.DenseLayer(input_, n_hid, name='hidden')
output = layers.DenseLayer(input_, len(a2i), name='output', nonlinearity=lasagne.nonlinearities.sigmoid)
#attach_tag_to_param('l2', 'W', hidden)
attach_tag_to_param('l2', 'W', output)

prob = layers.get_output(output)
loss = T.sum(-y * T.log(prob) - (1 - y) * T.log(1 - prob)) + regularize_network_params(output, l2, tags={'l2': True}) * l2_reg
pred = T.gt(prob, 0.5)
accuracy = T.mean(T.eq(pred, y))
true_pos = T.sum(pred * y)
pos = T.sum(pred)
true = T.sum(y)
precision = true_pos / pos
recall = true_pos / true
f1 = 2 * precision * recall / (precision + recall)

params = layers.get_all_params(output)
cnt = layers.count_params(output)
print('Total number of parameters:\t%d' %cnt)
updates = lasagne.updates.adam(loss, params, learning_rate=lr)

train_fn = theano.function([x, y], [loss, accuracy], updates=updates)
eval_fn = theano.function([x, y], [loss, accuracy, precision, recall, f1])
#import pdb;pdb.set_trace()
f_dev_best, loss_test, acc_test, p_test, r_test, f_test = None, None, None, None, None, None
for epoch in range(n_epoch):
    print("epoch = %d" %(epoch + 1))
    n_batch = x_train.shape[0] // bs
    loss_train, acc_train = 0.0, 0.0
    for i, xb, yb in iterate_minibatches(x_train, y_train, bs):
        print("\r%d/%d" %(i + 1, n_batch), end='')
        sys.stdout.flush()
        res = train_fn(xb, yb)
        loss_train += res[0]; acc_train += res[1]
    loss_train /= n_batch; acc_train /= n_batch
    print("Training loss:\t%f\nTraining accuracy:\t%f" %(loss_train, acc_train))
    
    loss_dev, acc_dev = 0.0, 0.0
    # for i, xb, yb in iterate_minibatches(x_dev, y_dev, 1):
    loss_dev, acc_dev, p_dev, r_dev, f_dev = eval_fn(x_dev.todense(), y_dev.todense())
    # res = eval_fn(x_dev.todense(), y_dev.todense())
    # loss_dev += res[0]; acc_dev += res[1]
    # loss_dev /= x_dev.shape[0]; acc_dev /= x_dev.shape[0]
    print("Dev loss:\t%f\nDev accuracy:\t%f" %(loss_dev, acc_dev))
    print(p_dev, r_dev, f_dev)
    
    if not f_dev_best or f_dev_best < f_dev:
        loss_, acc_ = 0.0, 0.0
        # for i, xb, yb in iterate_minibatches(x_test, y_test, 1):
        loss_, acc_, p, r, f = eval_fn(x_test.todense(), y_test.todense())
        # res = eval_fn(x_test.todense(), y_test.todense())
        # loss_ += res[0]; acc_ += res[1]
        # loss_ /= x_test.shape[0]; acc_ /= x_test.shape[0]
        f_dev_best = f_dev
    	loss_test = loss_
    	acc_test = acc_	
        p_test = p; r_test = r; f_test = f
	print('UPDATED')
	print(p, r, f)

print("################################################################################")
print("Test loss:\t%f\nTest accuracy:\t%f" %(loss_test, acc_test))
print(p_test, r_test, f_test)
