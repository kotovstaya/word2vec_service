import typing as tp
import gensim
import itertools
import tqdm
import numpy as np
from collections import Counter
import logging


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    handler.setFormatter(
        logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger


def preprocess_corpus(path: str,
                      limit: tp.Optional[int] = None
                      ) -> tp.List[tp.List[str]]:
    corpus = []
    with open(path, "r") as f:
        for ix, line in tqdm.tqdm(enumerate(f.readlines())):
            if limit is not None and ix == limit:
                break
            ll = gensim.utils.simple_preprocess(line)
            if len(ll) > 15:
                corpus.append(ll)
    return corpus


def get_dictionary(corpus: tp.List[tp.List[str]]):
    flatten_corpus = list(itertools.chain.from_iterable(corpus))
    dictionary = {word: index
                  for index, word in enumerate(
                    sorted(list(set(flatten_corpus))), 1)}
    dictionary["<unk>"] = 0
    invert_dictionary = {v: k for k, v in dictionary.items()}
    return dictionary, invert_dictionary


def get_word_probability_dict(corpus: tp.List[tp.List[str]]):
    flatten_corpus = list(itertools.chain.from_iterable(corpus)) + ["<unk>"]
    tmp_freq_dict = dict(Counter(flatten_corpus))
    for k in tmp_freq_dict.keys():
        tmp_freq_dict[k] = tmp_freq_dict[k]**(3/4)
    N = sum(tmp_freq_dict.values())
    return {k: v/N for k, v in tmp_freq_dict.items()}


def positive_sampling(line, window, count_in_line):
    line_length = len(line)
    rnd_ixs = np.random.choice(line_length - 1, size=count_in_line, replace=False)
    pairs = []
    step = window//2
    for rnd_ix in rnd_ixs:
        left_bound = rnd_ix - step
        right_bound = rnd_ix + step
        word = line[rnd_ix]
        for i in set(range(line_length)) & set(range(left_bound, right_bound+1)):
            context = line[i]
            if word != context and (word, context) not in pairs:
                pairs.append((word, context))
    return pairs


def negative_sampling(freq_dict, positive_pairs):
    pairs = []

    def get_negative_pair():
        return [
            np.random.choice(list(freq_dict.keys()),
                             p=list(freq_dict.values()))
            for _ in range(2)]

    pos_length = len(positive_pairs)
    neg_length = 0
    while neg_length < pos_length:
        neg_pair = get_negative_pair()
        if neg_pair not in positive_pairs and neg_pair[0] != neg_pair[1]:
            neg_length += 1
            pairs.append(neg_pair)
    return pairs
