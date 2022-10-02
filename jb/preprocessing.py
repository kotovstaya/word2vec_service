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
    handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger


def preprocess_corpus(path: str,
                      limit: tp.Optional[int] = None) -> tp.List[tp.List[str]]:
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


def get_windows(line, window) -> tp.List[tp.List[str]]:
    if len(line) < window:
        raise Exception(
            f"line size ({len(line)})" 
            f"is lower than window size ({window})")
    elif len(line) == window:
        return line
    else:
        return [line[ix:ix+window]
                for ix in range(len(line)-window)
                if len(line[ix:ix+window]) == window]


def positive_sampling(line, window, count_in_line):
    # print(f"len(line): {len(line)}")
    rnd_ixs = np.random.choice(range(window//2+1, len(line)-(window//2)-1), size=count_in_line, replace=False)
    pairs = []
    step = window//2
    for rnd_ix in rnd_ixs:
        word = line[rnd_ix]
        for i in range(step):
            context_l = line[rnd_ix + i]
            context_r = line[rnd_ix + i]
            if word != context_l and (word, context_l) not in pairs:
                pairs.append((word, context_l))
            if word != context_r and (word, context_r) not in pairs:
                pairs.append((word, context_r))
    return pairs


def negative_sampling(freq_dict, positive_pairs):
    pairs = []

    def get_negative_pair():
        return [
            np.random.choice(list(freq_dict.keys()), p=list(freq_dict.values()))
            for _ in range(2)]

    pos_length = len(positive_pairs)
    neg_length = 0
    while neg_length < pos_length:
        neg_pair = get_negative_pair()
        if neg_pair not in positive_pairs and neg_pair[0] != neg_pair[1]:
            neg_length += 1
            pairs.append(neg_pair)
    return pairs
