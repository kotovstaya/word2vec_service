import itertools
import logging
import typing as tp
from collections import Counter

import gensim
import numpy as np
import tqdm


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
    """

    :param path:
    :param limit:
    :return:
    """
    corpus = []
    step = 200
    with open(path, "r") as f:
        for ix, line in tqdm.tqdm(enumerate(f.readlines())):
            if limit is not None and ix == limit:
                break
            ll = gensim.utils.simple_preprocess(line)
            if 20 < len(ll) < step+100:
                if len(set(ll)) > 10:
                    corpus.append(ll)
            elif len(ll) > step+100:
                for el in range(0, len(ll), step):
                    tmp_line = ll[el: el+step]
                    if len(tmp_line) > 20 and len(set(tmp_line)) > 10:
                        corpus.append(tmp_line)
    print(f"corpus length: {len(corpus)}")
    return corpus


def get_dictionary(corpus: tp.List[tp.List[str]]):
    """

    :param corpus:
    :return:
    """
    flatten_corpus = list(itertools.chain.from_iterable(corpus))
    dictionary = {word: index
                  for index, word in enumerate(
                    sorted(list(set(flatten_corpus))), 1)}
    dictionary["<unk>"] = 0
    invert_dictionary = {v: k for k, v in dictionary.items()}
    return dictionary, invert_dictionary


def get_word_probability_dict(corpus: tp.List[tp.List[str]]):
    """

    :param corpus:
    :return:
    """
    flatten_corpus = list(itertools.chain.from_iterable(corpus)) + ["<unk>"]
    tmp_freq_dict = dict(Counter(flatten_corpus))
    for k in tmp_freq_dict.keys():
        tmp_freq_dict[k] = tmp_freq_dict[k]**(3/4)
    N = sum(tmp_freq_dict.values())
    return {k: v/N for k, v in tmp_freq_dict.items()}


def coocurance_martix(corpus: tp.List[tp.List[str]],
                      dictionary: tp.Dict[str, int],
                      window: int) -> np.ndarray:
    """

    :param corpus:
    :param dictionary:
    :param window:
    :return:
    """
    dict_lnegth = len(dictionary.keys())
    mtx = np.zeros(shape=(dict_lnegth, dict_lnegth))
    step = window//2
    for ix, line in tqdm.tqdm(enumerate(corpus)):
        line_length = len(line)
        words_ids = np.array([dictionary[el] for el in line])
        for word_ix in range(line_length):
            word_id = words_ids[word_ix]
            left_bound = word_ix - step
            right_bound = word_ix + step
            valid_ixs = list(
                set(range(line_length))
                & set(range(left_bound, right_bound+1)))
            context_ids = words_ids[valid_ixs]
            mtx[word_id, context_ids] = 1
    return mtx


def positive_sampling(line: tp.List[str],
                      window: int,
                      count_in_line: int):
    """

    :param line:
    :param window:
    :param count_in_line:
    :return:
    """
    line_length = len(line)
    rnd_ixs = np.random.choice(line_length - 1,
                               size=count_in_line,
                               replace=False)
    pairs = []
    step = window//2
    for rnd_ix in rnd_ixs:
        left_bound = rnd_ix - step
        right_bound = rnd_ix + step
        word = line[rnd_ix]
        for i in set(
                range(line_length)) & set(range(left_bound, right_bound+1)):
            context = line[i]
            if word != context and (word, context) not in pairs:
                pairs.append((word, context))
    return pairs


def negative_sampling(freq_dict: tp.Dict[str, float],
                      positive_pairs: tp.List[tp.Tuple[str, str]]):
    """

    :param freq_dict:
    :param positive_pairs:
    :return:
    """
    pairs = []

    def get_negative_pair():
        """

        :return:
        """
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


def negative_coocurance_sampling(freq_dict: tp.Dict[str, float],
                                 vocab: tp.Dict[str, int],
                                 N: int):
    """

    :param freq_dict:
    :param vocab:
    :param N:
    :return:
    """
    pairs = [
        [
            vocab[
                np.random.choice(
                    list(freq_dict.keys()), p=list(freq_dict.values())
                )
            ]
            for _ in range(2)]
        for _ in range(N)
    ]
    return pairs
