import typing as tp
import gensim
import itertools
from collections import Counter


def preprocess_corpus(path: str,
                      limit: tp.Optional[int] = None) -> tp.List[tp.List[str]]:
    corpus = []
    with open(path, "r") as f:
        for ix, line in enumerate(f.readlines()):
            if limit is not None and ix == limit:
                break
            corpus.append(gensim.utils.simple_preprocess(line))
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
