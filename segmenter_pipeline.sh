corpus=$1
lang=$2
model_path=$3
data_dir=$4

vocab_file=$corpus.vocab
segmented_vocab_file=$corpus.vocab.segmented
segmented_corpus=$corpus.segmented
python get_vocab.py $corpus $vocab_file
cat $vocab_file | python segmenter.py $lang $model_path -dd $data_dir > $segmented_vocab_file
python substitute.py $corpus $segmented_corpus $segmented_vocab_file

