import pickle
import logging

from arglib import add_argument
from dev_misc import logger
from ILP import ILP as ILP
from model import MC
import random


def parse_args():
    add_argument('--lang', help='language')  # TODO add cfg
    add_argument('--top_affixes', '-A', type=int, help='top K frequent affixes to use')
    add_argument('--top_words', '-W', default=5000, type=int, help='top K words to use')
    add_argument('--compounding', action='store_true', help='use compounding, default False')
    add_argument('--sibling', action='store_true', help='use sibling feature, default False')
    add_argument('--supervised', action='store_true', help='flag to use supervised')
    add_argument('--ILP', action='store_true', help='ILP mode, default False')
    add_argument('--seed', default=1234, type=int, help='random seed, default 1234')
    add_argument('--alpha', '-a', default=0.001, type=float, help='alpha value for ILP')
    add_argument('--beta', '-b', default=1.0, type=float, help='beta value for ILP')
    add_argument('--debug', action='store_true', help='debug mode, default false')
    add_argument('--load', '-l', help='file to load the model from')
    add_argument('--save', '-s', help='file to save the model to')  # TODO saving should happen every iteration.
    add_argument('--iter', default=5, type=int, help='number of ILP iterations')
    add_argument('--data_path,', '-d', dest='data_path', default='data/', help='folder where data are kept')
    # TODO this is for testing?
    add_argument('--input_file,', '-I', help='input file, a list of words for which the trained model is used')
    add_argument('--output_file,', '-O', help='output file')
    add_argument('--reg_l2', default=1.0, type=float, help='regularization hyperparameter for l2 loss')
    add_argument('--wv_dim', default=200, type=int, help='word vector dimensionality')
    args = parser.parse_args()

    random.seed(args.seed)
    return args


if __name__ == "__main__":
    args = parse_args()

    create_logger(arg.log_dir)

    if args.supervised:
        assert not args.ILP and not args.load
    if args.load:
        logging.info('Loading model from %s...' % args.load)
        model = pickle.load(open(args.load, 'r'))
        logging.info('Done loading')
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
        print 'Saving model to %s...' % args.save
        if args.ILP:
            ilp.model = None    # gurobi model has to be skipped for pickling
            cPickle.dump(ilp, open(args.save, 'w'))
        else: cPickle.dump(m, open(args.save, 'w'))
        print 'Done.'

    if args.input_file:
        if not args.output_file: args.output_file = args.input_file + '.out'
        print 'Producing segmentations for the input file %s...' % args.input_file
        words = set([line.rstrip() for line in Path(args.input_file).open('r', encoding='utf8')])
        if type(model) == ILP:
            model.seeds = words
            model.run()
            model.parse(wordset=words, out_file=args.output_file)
        else:
            assert type(model) == MC
            model.write_segments_to_file(wordset=words, out_file=args.output_file)
        print 'Done'
