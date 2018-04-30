import codecs
import sys
in_name = sys.argv[1]
out_name = sys.argv[2]
seg_fin = sys.argv[3]

segments = dict()
with codecs.open(seg_fin,'r', 'utf8') as fin:
    for line in fin:
        segs = line.strip().split('\t')
        segments[segs[0]] = segs[1].split()

with codecs.open(in_name, 'r', 'utf8') as fin, codecs.open(out_name, 'w', 'utf8') as fout:
    for line in fin:
        out = ' '.join(['|'.join(segments[x]) for x in line.strip().split()])
        fout.write(out + '\n')
