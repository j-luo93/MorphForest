import sys
import codecs

f_name = sys.argv[1]
f_out = sys.argv[2]
encoding = sys.argv[3]

with codecs.open(f_name, 'r', encoding=encoding, errors='strict') as fin, codecs.open(f_out, 'w', 'utf8', errors='strict') as fout:
    for line in fin:
        fout.write(line)
