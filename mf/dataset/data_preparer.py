import logging
import pathlib
from collections import Counter, defaultdict

from arglib import use_arguments_as_properties
from dev_misc import Map, log_this

from .word_vectors import DummyWordVectors, WordVectors


@use_arguments_as_properties('data_path', 'lang', 'log_dir', 'gold_affixes', 'top_affixes', 'top_words', 'inductive', 'use_word_vectors')
class DataPreparer:

    def __init__(self):
        self.word_vector_file = self.data_path + '/wv.%s' % self.lang
        self.gold_segs_file = self.data_path + '/gold.%s' % self.lang
        self.wordlist_file = self.data_path + '/wordlist.%s' % self.lang
        self.predicted_file = {'train': self.log_dir + '/pred.train.%s' % self.lang,
                               'test': self.log_dir + '/pred.test.%s' % self.lang}
        if self.gold_affixes:
            self.gold_affix_file = {'pre': self.data_path + '/gold_pre.%s' % self.lang,
                                    'suf': self.data_path + '/gold_suf.%s' % self.lang}

        self.read_wordlist()
        self.read_word_vectors()
        self.read_gold_segs()
        self._add_top_words_from_wordlist()
        self.read_affixes()
        if self.inductive:
            self._add_words_from_gold()
        # NOTE Commented out supervised mode.
        # if self.supervised:
        #     self.get_gold_parents()
        #     self.train_set = set(self.gold_parents.keys())
        # self.update_train_set()
        self.data = sorted(self.train_set)

    def __len__(self):
        return len(self.data)

    @log_this('INFO')
    def read_word_vectors(self):
        if self.use_word_vectors:
            self.wv = WordVectors(self.word_vector_file)
        else:
            self.wv = DummyWordVectors()

    @log_this('INFO')
    def read_gold_segs(self):
        self.gold_segs = dict()
        with pathlib.Path(self.gold_segs_file).open('r', encoding='utf8') as fin:
            for i, line in enumerate(fin, 1):
                segs = line.strip().split(':')
                if len(segs) != 2:
                    raise RuntimeError(f'Something is wrong with the segmentation file at line {i}.')
                self.gold_segs[segs[0]] = segs[1].split()

    @log_this('INFO')
    def read_wordlist(self):
        self.word_cnt = dict()
        with pathlib.Path(self.wordlist_file).open('r', encoding='utf8') as fin:
            for i, line in enumerate(fin, 1):
                segs = line.split()
                if len(segs) != 2:
                    raise RuntimeError(f'Something is wrong with the word count file at line {i}.')
                # Don't include very short words (length < 3)
                if len(segs[0]) >= 3:
                    self.word_cnt[segs[0]] = int(segs[1])

    @log_this('INFO')
    def _add_top_words_from_wordlist(self):
        self.train_set = set()
        cnt = Counter()
        ptr = defaultdict(list)
        for k, v in self.word_cnt.items():
            # if '-' not in k or len(k.split("'")) != 2: # ignore hyphenated words, or apostrophed words
            cnt[v] += 1
            ptr[v].append(k)
        cnt = sorted(cnt.items(), key=lambda x: x[0], reverse=True)
        if self.top_words > 0:
            i = 0
            while len(self.train_set) < self.top_words and i < len(cnt):
                self.train_set.update(ptr[cnt[i][0]])
                self.freq_thresh = cnt[i][0]
                i += 1
        logging.info('Added %d words to training set.' % (len(self.train_set)))

    @log_this('INFO')
    def _add_words_from_gold(self):
        self.train_set.update(filter(lambda w: len(w) >= 3, self.gold_segs.keys()))
        logging.info('Now %d words in training set, inductive mode.' % (len(self.train_set)))

    @log_this('INFO')
    def read_affixes(self):
        self.prefixes, self.suffixes = set(), set()
        if hasattr(self, 'gold_affix_file'):
            raise NotImplementedError('Support for gold affixes not implemented.')
        else:
            # TODO isn't this wrong?
            suf_cnt, pre_cnt = Counter(), Counter()
            for word in self.train_set:
                for pos in range(1, len(word)):
                    left, right = word[:pos], word[pos:]
                    suf_cnt[right] += 1
                    pre_cnt[left] += 1
                    # if left in self.word_cnt: suf_cnt[right] += 1
                    # if right in self.word_cnt: pre_cnt[left] += 1
            suf_cnt = sorted(suf_cnt.items(), key=lambda x: x[1], reverse=True)
            pre_cnt = sorted(pre_cnt.items(), key=lambda x: x[1], reverse=True)
            self.suffixes = set([suf for suf, cnt in suf_cnt[:self.top_affixes]])
            self.prefixes = set([pre for pre, cnt in pre_cnt[:self.top_affixes]])

    def get_batch(self):
        """This returns the entire wordlist actually."""
        return Map(wordlist=self.data,
                   num_samples=len(self.data))
    # NOTE Commented out supervised mode
    # def get_gold_parents(self):
    #     self._check_first_time_read('gold_parents')
    #     if not self.prefixes or not self.suffixes:
    #         raise RuntimeError('Should have read prefixes and suffixes before this.')
    #     self.gold_parents = dict()
    #     for child in self.gold_segs:
    #         # ignore all hyphenated words.
    #         if '-' in child:
    #             continue

    #         segmentation = self.gold_segs[child][0]  # only take the first segmentation
    #         while True:
    #             parts = segmentation.split("-")
    #             ch = "".join(parts)
    #             # print ch
    #             if len(parts) == 1:
    #                 self.gold_parents[ch] = (ch, 'STOP')
    #                 break
    #             # simple heuristic to determine the parent.
    #             scores = dict()
    #             prefix, suffix = parts[0], parts[-1]
    #             right_parent = ''.join(parts[1:])
    #             left_parent = ''.join(parts[:-1])
    #             prefix_score = self.get_similarity(ch, right_parent)
    #             suffix_score = self.get_similarity(ch, left_parent)
    #             p_score, s_score = 0.0, 0.0
    #             if prefix in self.prefixes: p_score += 1.0
    #             if suffix in self.suffixes: s_score += 1.0
    #             prefix_score += p_score
    #             suffix_score += s_score
    #             scores[(right_parent, 'PREFIX')] = prefix_score
    #             scores[(left_parent, 'SUFFIX')] = suffix_score + 0.1

    #             if right_parent in self.word_cnt and prefix in self.word_cnt:
    #                 scores[(prefix, right_parent), 'COM_RIGHT'] = 0.75 * \
    #                     self.get_similarity(right_parent, ch) + 0.25 * self.get_similarity(prefix, ch) + 1
    #             if left_parent in self.word_cnt and suffix in self.word_cnt:
    #                 scores[(left_parent, suffix), 'COM_LEFT'] = 0.25 * self.get_similarity(suffix, ch) + \
    #                     0.75 * self.get_similarity(left_parent, ch) + 1

    #             if self.transform:
    #                 if (len(left_parent) > 1 and left_parent[-1] == left_parent[-2]):
    #                     repeat_parent = left_parent[:-1]
    #                     score = self.get_similarity(ch, repeat_parent) + s_score - 0.25
    #                     scores[(repeat_parent, 'REPEAT')] = score
    #                 if left_parent[-1] == 'i':
    #                     modify_parent = left_parent[:-1] + 'y'  # only consider y -> i modification
    #                     score = self.get_similarity(ch, modify_parent) + s_score - 0.25
    #                     scores[(modify_parent, 'MODIFY')] = score
    #                 if left_parent[-1] != 'e':
    #                     delete_parent = left_parent + "e"    # only consider e deletion.
    #                     score = self.get_similarity(ch, delete_parent) + s_score - 0.25
    #                     scores[(delete_parent, 'DELETE')] = score

    #             best = max(scores.items(), key=itemgetter(1))[0]
    #             type_ = best[1]
    #             # print best, scores
    #             if type_ == 'PREFIX':
    #                 segmentation = segmentation[len(prefix) + 1:]
    #             elif type_ == 'SUFFIX':
    #                 segmentation = segmentation[:len(segmentation) - len(suffix) - 1]
    #             elif type_ == 'MODIFY':
    #                 segmentation = segmentation[:len(segmentation) - len(suffix) - 2] + 'y'
    #             elif type_ == 'REPEAT':
    #                 segmentation = segmentation[:len(segmentation) - len(suffix) - 2]
    #             elif type_ == 'DELETE':
    #                 segmentation = segmentation[:len(segmentation) - len(suffix) - 1] + 'e'
    #             elif type_ == 'COM_LEFT':
    #                 segmentation = segmentation[:-len(suffix) - 1]
    #             elif type_ == 'COM_RIGHT':
    #                 segmentation = segmentation[len(prefix) + 1:]
    #             self.gold_parents[ch] = best
    #     print('Read %d gold parents.' % (len(self.gold_parents)), file=sys.stderr)
