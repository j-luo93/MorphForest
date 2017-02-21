from __future__ import division

import argparse
import cPickle
import codecs
from model import MC
import sys
from ILP import ILP

################################################################################
# a wrapper function in case it's not defined.
def get_stem(path, mode='surface'):
    assert mode in ['surface', 'canonical']
    if hasattr(path, 'get_stem') and callable(getattr(path, 'get_stem')): 
        return path.get_stem(mode=mode)
    else:
        assert path.is_ended()
        if not hasattr(path, 'seg'):
            path.seg = path._segment_iter(path.root)
        segs = getattr(path.seg, mode)
        return ''.join([x for x, y in filter(lambda item: item[0] if item[1] == 'stem' else '', zip(segs.split('-'), path.seg.label.split('-')))])
################################################################################    


parser = argparse.ArgumentParser()
parser.add_argument('--load', '-l', metavar='', help='model file to load')
parser.add_argument('--gold-file', '-G', dest='gold_file', metavar='', help='gold stem file')
parser.add_argument('--mode', '-m', metavar='', default='surface', help='prediction mode')
parser.add_argument('--output-file', '-O', metavar='', dest='output_file', help='output file')
parser.add_argument('-clusters', action='store_true', help='flag to test on clusters')
parser.add_argument('-keep', action='store_true', dest='keep_train_set', help='flag to keep train set if the model is ILP')

args = parser.parse_args()
print('Loading model...')
model = cPickle.load(open(args.load, 'r'))
print('Done.')
gold_stems = dict()

with codecs.open(args.gold_file, 'r', 'utf8') as fin:
    for line in fin:
        if args.clusters:
            form = line.strip()
            stem = form 
        else:
            form, stem = line.strip().split('\t')
        gold_stems[form] = stem
    
if type(model) == ILP:
    if args.keep_train_set:
        model.seeds.update(gold_stems.keys())
    else:
        model.seeds = set(gold_stems.keys())
    #del model.base.train_set
    #model.base.top_words = 50000
    #model.base._add_top_words_from_wordlist()
    #model.seeds = set(model.base.train_set)
    #model.seeds.update(gold_stems.keys())
    #model.order = 5
    model.run()
#import pdb; pdb.set_trace()

acc = 0
print("producing stems...")
with codecs.open(args.output_file, 'w', 'utf8') as fout:
    for i, w in enumerate(gold_stems):
        print '\r%d' %(i + 1),
        sys.stdout.flush()
        path = model.get_seg_path(w)
        pred_stem = get_stem(path, args.mode)
        if pred_stem == gold_stems[w]: acc += 1
        fout.write('%s\t%s\n' %(w, pred_stem))
print('Accuracy = %f' %(acc / len(gold_stems)))

