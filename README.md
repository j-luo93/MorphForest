# Update
* (Almost) completely rewrote the codebase.
* Migrated to Python 3.7.
* Files are expected to be in utf8 encoding. `_standardize` method is removed -- data cleaning should not be handled by this codebase.
* Use two-lettered language code instead three-lettered. For instance, use `en` instead of `eng` for English.

# MorphForest
Code for the paper [Unsupervised Learning of Morphological Forest](http://people.csail.mit.edu/j_luo/assets/publications/MorphForest.pdf) (to appear in TACL 2017).

The repo contains a baseline model that is roughly based on [this](https://github.com/karthikncode/MorphoChain).

## Installation
[Theano](http://deeplearning.net/software/theano/install.html) and [Gurobi](http://www.gurobi.com/downloads/download-license-center?utm_expid=11945996-35.HzYrTI0vR2iUbbuLy5dLJw.1&utm_referrer=http%3A%2F%2Fwww.gurobi.com%2F) are needed. You can obtain a free academic license for Gurobi solver.

You should be able to do a test run `python run.py eng -ILP -DEBUG` after proper installation.

## Data preparation
You will need to prepare three files for input before running the model, which by default are stored in the `data` folder where three sample files are also included.
* Gold segmentation file, named `gold.<lc>`. One word per line, in the format of *<word>:<segmentations>*, where morphemes are separated by hyphens, and alternative segmentations separated by spaces. See `data/gold.eng.toy` for an example.
* Word vector file, named `wv.<lc>`. One word per line, specifically one word followed by a continuous vector of float numbers which are all separated by spaces. See `data/wv.eng.toy` for an example.
* Wordlist file, named `wordlist.<lc>`. One word followed by its frequency per line, separated by space. See `data/wordlist.eng.toy` for an example.

*\<lc\>* is the language code you have to specify, usually a three-letter or two-letter string, e.g. *eng* for English. This language code is entered as an argument when you run the code (as detailed below), and is used to find the input files in the data folder.

## Running the code
Use `python run.py <lc>` to run the model, where *\<lc\>* is the afore-mentioned language code. You can add `-h` flag to see a list of settings you can change, some of which are detailed below.
* `--top-affixes` or `-a`, number of most frequent affixes to use, 100 by default
* `--top-words`, `-W`, number of most frequent words to train on, 5000 by default
* `-compounding`, flag to include compounding features
* `-sibling`, flag to include sibling features
* `-supervised`, flag to train a supervised model
* `-ILP`, flag to use the full model which is trained iteratively. Without this flag, a baseline model will be trained. The number of iteration is specified by `--iter`, 5 by default.
* `--save` and `--load` , save the model to or load the model from a specified location.
* After training or loading a model, use `--input-file` or `-I` to specify the file of words (one word per line) to segment, and `--output-file` or `-O` to store the segmentations.
* `--alpha` or `-a`, and `--beta` or `-b` are hyperparameters as introduced in the paper, 0.001 and 1.0 by default respectively. To reproduce the results as reported in Table 4, use the default values for English and Arabic. For Turkish and German, use `--beta 3.0`, as more affixes are expected for both languages.

Segmentation results will be saved in `out` folder, along with feature weights.

## Dataset
Segmentation dataset is available on my [website](http://people.csail.mit.edu/j_luo/publications/).

