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
                      limit: tp.Optional[int] = None,
                      step: int = 200,
                      min_value: int = 20,
                      ) -> tp.List[tp.List[str]]:
    """
    Create a corpus of proper size.
    Sometimes it happens that line of some file is very long.
    So we need to split a long line into chunks of the specific size

    Args
        :param path: file where each line is a .py file with code
        :param limit: if you test something and don't want to get all lines
        :param step: valid size for string's split
        :param min_value: if length of line is less than this value than skip this line
    :return: List of prepared list of words
    """
    logger = get_logger("preprocess_corpus")
    corpus = []
    step_width = 100
    with open(path, "r") as f:
        for ix, line in tqdm.tqdm(enumerate(f.readlines())):
            if limit is not None and ix == limit:
                break
            ll = gensim.utils.simple_preprocess(line)
            if min_value < len(ll) < step + step_width:
                if len(set(ll)) > int(min_value//2):
                    corpus.append(ll)
            elif len(ll) > step + step_width:
                for el in range(0, len(ll), step):
                    tmp_line = ll[el: el+step]
                    if len(tmp_line) > min_value and len(set(tmp_line)) > int(min_value//2):
                        corpus.append(tmp_line)
    logger.info(f"corpus length: {len(corpus)}")
    return corpus


def get_dictionary(corpus: tp.List[tp.List[str]]):
    """
    For each word in a corpus get id.
    Return certain mapping and inverse mapping

    Args:
        :param corpus: List of prepared list of words
    :return: pair: mapping from a word to a word index, inverse mapping for the first one
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
    For each word in a corpus get frequency of this word in the corpus.
    Return certain mapping and inverse mapping

    Args:
        :param corpus: List of prepared list of words
    :return: pair: mapping from a word to a frequency of this word in the
            corpus, inverse mapping for the first one
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
    To speed up the learning process, we should create a matrix of the
    occurrence of words in the context.
    If the words met in the context of a certain size, then put 1

    Args:
        :param corpus: List of prepared list of words
        :param dictionary: mapping from a word to a word index
        :param window: what is a width of our context size
    :return: matrix. Where mtx[i,j] == 1 if i and j were in the same context.
            Otherwise 0
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
                      count_in_line: int) -> tp.List[tp.Tuple[str, str]]:
    """
    In a line, select several points (words) to define the input word.
    Get pairs within the context of these words.

    Args:
        :param line: list of prepared words
        :param window: what is a width of our context size
        :param count_in_line: how many samples should we get from one line in corpus
    :return: List of pairs from the same context
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


def get_negative_pair(freq_dict: tp.Dict[str, float]):
    """
    get a pair of negative words.

    Args:
        :param freq_dict: mapping from a word to a frequency of this word in the corpus
    :return: the pair of negative words based on statistics
    """
    return [
        np.random.choice(list(freq_dict.keys()),
                         p=list(freq_dict.values()))
        for _ in range(2)]


def negative_sampling(freq_dict: tp.Dict[str, float],
                      positive_pairs: tp.List[tp.Tuple[str, str]]):
    """
    Randomly, based on statistics, get a list of pairs of words from the corpus

    Args:
        :param freq_dict: mapping from a word to a frequency of this word in the corpus
        :param positive_pairs: List of words' pairs that were in the same context
    :return: List of words' pairs based on statistics
    """
    pairs = []

    pos_length = len(positive_pairs)
    neg_length = 0
    while neg_length < pos_length:
        neg_pair = get_negative_pair(freq_dict)
        if neg_pair not in positive_pairs and neg_pair[0] != neg_pair[1]:
            neg_length += 1
            pairs.append(neg_pair)
    return pairs


def negative_coocurance_sampling(freq_dict: tp.Dict[str, float],
                                 vocab: tp.Dict[str, int],
                                 N: int):
    """
    yeat another implementation for obtaining negative samples.
    :param freq_dict: mapping from a word to a frequency of this word in the corpus
    :param vocab: mapping from a word to a word index
    :param N: size of these pairs
    :return:
    """
    pairs = [[vocab[el] for el in get_negative_pair(freq_dict)] for _ in range(N)]
    return pairs
