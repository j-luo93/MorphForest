import codecs
import sys

vocab = set()
with codecs.open(sys.argv[1], 'r', 'utf8') as fin, codecs.open(sys.argv[2], 'w', 'utf8') as fout:
    for line in fin:
        segs = line.strip().split()
        vocab.update(segs)

    for k in vocab:
        fout.write('%s\n'%k)
    
