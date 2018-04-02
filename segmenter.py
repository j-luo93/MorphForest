#!/usr/bin/env python
# Author: Thamme Gowda tg@isi.edu
# Crated on: October 30, 2017
"""
Morfessor utility for morphological segmentation of bitext data.


Note:
 1. Training can be done using `morfessor-train` command line tool.
 Example::
    morfessor-train -s morf-model.src.pkl 1A-v1-train.src
 2. Use only the training data split for the morfessor training
"""
from __future__ import print_function
import traceback

def main(args):
    if (sys.stdout.encoding is None):
        print("please `export PYTHONIOENCODING=UTF-8`", file=sys.stderr)
        exit(1)
    model = args['model']
    fout = args['output']
    fin = args['input']
    lang = args['lang']
    data_dir = args['data_dir']

    import subprocess
    import codecs

    if fin is sys.stdin:
        fin = 'stdin'
    subprocess.check_output('python src/run.py %s --load %s -I %s -O %s -d %s'  %(lang, model, fin, 'tmp', data_dir), shell=True)
    # post-processing to deal with hyphens
    if fout is not sys.stdout:
        fout = codecs.open(fout, 'w', 'utf8')
    fin = codecs.open('tmp', 'r', 'utf8')
    for line in fin:
        token, segs = line.strip().split('\t')
        segs = segs.split()
        # first pass to identify intervals of consecutive hyphens
        intervals = list()
        s = None
        t = None
        for i, seg in enumerate(segs):
            if seg == '-':
                if s is None:
                    s = i
                t = i + 1
            else:
                if s is not None:
                    intervals.append((s, t))
                s = None
        if s is not None:
            intervals.append((s, t))

        # combine consecutive hyphens
        i = len(intervals) - 1
        while i >= 0:
            s, t = intervals[i]
            if t - s > 1 or s == 0:
                segs = segs[:s] + ['-' * (t - s)] + segs[t:]
            else:
                segs = segs[:s] + segs[t:]
            i -= 1
        segs = ' '.join(segs)
        out_line = '%s\t%s\n' % (token, segs)
        fout.write(out_line)
    fin.close()
    if fout is not sys.stdout:
        fout.close()

if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser("""Morfessor.""")
    parser.add_argument('lang', help='language')
    parser.add_argument('model', help='Segmenter model, one per text column (for TSV)')
    parser.add_argument('-in', '--input', help='Input file. Default=STDIN', default=sys.stdin)
    parser.add_argument('-out', '--output', help='Output file. DEFAULT=STDOUT', default=sys.stdout)
    parser.add_argument('--data_dir', '-dd', help='data directory', default='data/')
    args = vars(parser.parse_args())
    main(args)
