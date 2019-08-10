import logging
import pickle
import random

from arglib import parser
from dev_misc import Map, create_logger
from mf.configs import registry
from mf.training.manager import Manager
# from ILP import ILP as ILP
# from model import MC


def parse_args():
    parser.add_argument('--lang', dtype=str, help='language')  # TODO add cfg
    parser.add_argument('--top_affixes', '-A', dtype=int, help='top K frequent affixes to use')
    parser.add_argument('--top_words', '-W', default=5000, dtype=int, help='top K words to use')
    parser.add_argument('--compounding', default=False, dtype=bool, help='use compounding, default False')
    parser.add_argument('--sibling', default=False, dtype=bool, help='use sibling feature, default False')
    parser.add_argument('--gold_affixes', default=False, dtype=bool, help='use gold affixes')
    parser.add_argument('--supervised', default=False, dtype=bool, help='flag to use supervised')
    parser.add_argument('--ILP', default=False, dtype=bool, help='ILP mode, default False')
    parser.add_argument('--seed', default=1234, dtype=int, help='random seed, default 1234')
    parser.add_argument('--alpha', '-a', default=0.001, dtype=float, help='alpha value for ILP')
    parser.add_argument('--beta', '-b', default=1.0, dtype=float, help='beta value for ILP')
    parser.add_argument('--debug', default=False, dtype=bool, help='debug mode, default false')
    parser.add_argument('--strict_wv', default=True, dtype=bool, help='strict wv mode that does not allow oov')
    parser.add_argument('--inductive', default=True, dtype=bool, help='inductive mode')
    parser.add_argument('--use_word_vectors', default=False, dtype=bool, help='use word vectors')
    parser.add_argument('--load', '-l', help='file to load the model from')
    parser.add_argument('--save', '-s', help='file to save the model to')  # TODO saving should happen every iteration.
    parser.add_argument('--iteration', default=5, dtype=int, help='number of ILP iterations')
    parser.add_argument('--data_path,', '-d', default='data/',
                        help='folder where data are kept')  # TODO this is for testing?
    parser.add_argument('--input_file,', '-I', help='input file, a list of words for which the trained model is used')
    parser.add_argument('--output_file,', '-O', help='output file')
    parser.add_argument('--reg_hyper', default=1.0, dtype=float, help='regularization hyperparameter for l2 loss')
    parser.add_argument('--learning_rate', '-lr', default=1e-1, dtype=float, help='initial learning rate')
    parser.add_argument('--default_oov', default=-0.5, dtype=float, help='default wv cosine similarity for oov')
    parser.add_argument('--wv_dim', default=200, dtype=int, help='word vector dimensionality')
    parser.add_argument('--max_epoch', default=1000, dtype=int, help='max number of epochs per iteration')
    parser.add_argument('--check_interval', '-ci', default=100, dtype=int, help='check interval')
    parser.add_cfg_registry(registry)
    args = Map(**parser.parse_args())

    random.seed(args.seed)
    if args.supervised:
        raise NotImplementedError('Supervised mode not supported yet.')
    return args


def train():
    manager = Manager()
    manager.train()


if __name__ == "__main__":
    args = parse_args()

    create_logger(args.log_dir + '/log')

    if args.supervised:
        assert not args.ILP and not args.load
    if args.load:
        raise NotImplementedError('Loading is not yet supported.')
        logging.info('Loading model from %s...' % args.load)
        model = pickle.load(open(args.load, 'r'))
        logging.info('Done loading')
    else:
        train()
        # NOTE Commented out the old way of doing it.
        # m = MC(**vars(args))
        # m.read_all_data()

        # if not args.ILP:
        #     m.run(reread=False)
        # model = m
        # else:
        #     for i in range(args.iter):
        #         m.clear_caches()
        #         m.run(reread=False)
        #         m.clear_caches()
        #         if i == 0: ilp = ILP(m, alpha=args.alpha, beta=args.beta)
        #         ilp.run()
        #         ilp.parse()
        #         ilp.evaluate()
        #         m.update_pruner(ilp.pruner)
        # model = ilp

    # NOTE Test mode is not yet supported.
    # if args.input_file:
    #     if not args.output_file: args.output_file = args.input_file + '.out'
    #     print 'Producing segmentations for the input file %s...' % args.input_file
    #     words = set([line.rstrip() for line in Path(args.input_file).open('r', encoding='utf8')])
    #     if type(model) == ILP:
    #         model.seeds = words
    #         model.run()
    #         model.parse(wordset=words, out_file=args.output_file)
    #     else:
    #         assert type(model) == MC
    #         model.write_segments_to_file(wordset=words, out_file=args.output_file)
    #     print 'Done'
