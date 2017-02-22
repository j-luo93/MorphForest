import codecs
import argparse
from model import MC
from ILP import ILP as ILP
import cPickle

parser = argparse.ArgumentParser()
parser.add_argument('lang', metavar='language', help='language')
parser.add_argument('--top-affixes', '-A', dest='top_affixes', default=100, type=int, help='top K frequent affixes to use', metavar='')
parser.add_argument('--top-words', '-W', dest='top_words', default=5000, type=int, help='top K words to use', metavar='')
parser.add_argument('-compounding', action='store_true', help='use compounding, default False')
parser.add_argument('-sibling', action='store_true', help='use sibling feature, default False')
parser.add_argument('-supervised', action='store_true', help='flag to use supervised')
parser.add_argument('-ILP', action='store_true', help='ILP mode, default False')
parser.add_argument('--seed', default=1234, type=int, help='random seed, default 1234', metavar='')
parser.add_argument('--alpha', '-a', default=0.001, type=float, help='alpha value for ILP', metavar='')
parser.add_argument('--beta', '-b', default=1.0, type=float, help='beta value for ILP', metavar='')
parser.add_argument('-DEBUG', action='store_true', help='debug mode, default false')
parser.add_argument('--load', '-l', help='file to load the model from', metavar='')
parser.add_argument('--save', '-s', help='file to save the model to', metavar='')
parser.add_argument('--iter', default=5, type=int, help='number of ILP iterations', metavar='')
parser.add_argument('--data-path,', '-d', dest='data_path', default='data/', help='folder where data are kept', metavar='')
parser.add_argument('--input-file,', '-I', dest='input_file', help='input file, a list of words for which the trained model is used', metavar='')
parser.add_argument('--output-file,', '-O', dest='output_file', help='output file', metavar='')
args = parser.parse_args()

if args.supervised:
    assert not args.ILP and not args.load
if args.load:
    print 'Loading model from %s...' %args.load
    model = cPickle.load(open(args.load, 'r'))
    print 'Done.'
else:
    m = MC(**vars(args))
    m.read_all_data()

    if not args.ILP:
        m.run(reread=False)
	model = m
    else:
        for i in range(args.iter):
            m.clear_caches()
            m.run(reread=False)
            m.clear_caches()
            if i == 0: ilp = ILP(m, alpha=args.alpha, beta=args.beta)
            ilp.run()
            ilp.parse()
            ilp.evaluate()
            m.update_pruner(ilp.pruner)
	model = ilp

if args.save:
    print 'Saving model to %s...' %args.save
    if args.ILP:
        ilp.model = None    # gurobi model has to be skipped for pickling
        cPickle.dump(ilp, open(args.save, 'w'))
    else: cPickle.dump(m, open(args.save, 'w'))
    print 'Done.'

if args.input_file:
    if not args.output_file: args.output_file = args.input_file + '.out'
    print 'Producing segmentations for the input file %s...' %args.input_file
    words = set([line.rstrip() for line in codecs.open(args.input_file, 'r', 'utf8')])
    if type(model) == ILP:
        model.seeds = words
        model.run()
        model.parse(wordset=words, out_file=args.output_file)
    else:
    	assert type(model) == MC
    	model.write_segments_to_file(wordset=words, out_file=args.output_file)
    print 'Done'
