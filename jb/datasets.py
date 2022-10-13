import typing as tp

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing import (
    positive_sampling,
    negative_sampling,
    negative_coocurance_sampling,
    get_logger
)


class VanillaWord2VecDataset(Dataset):

    def __init__(self,
                 corpus: tp.List[tp.List[str]],
                 dictionary: tp.Dict[str, int],
                 window: int,
                 count_in_line: int):
        """

        :param corpus:
        :param dictionary:
        :param window:
        :param count_in_line:
        """
        super().__init__()
        self.logger = get_logger(VanillaWord2VecDataset.__name__)
        self.corpus = corpus
        self.window = window
        self.dictionary = dictionary
        self.count_in_line = count_in_line

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, idx) -> tp.Tuple[torch.LongTensor, torch.LongTensor]:
        line = self.corpus[idx]
        pos_samples = positive_sampling(line, self.window, self.count_in_line)
        if not len(pos_samples):
            print(f"pos_samples: {pos_samples}")
            print(line)

        words = np.array([self.dictionary[el[0]] for el in pos_samples])
        contexts = np.array([self.dictionary[el[1]] for el in pos_samples])

        ixs = np.array(range(words.shape[0]))
        np.random.shuffle(ixs)

        return (
            torch.LongTensor(words[ixs][:, None]),
            torch.LongTensor(contexts[ixs][:, None]),
        )


class VanillaWord2VecWithCoocuranceDataset(Dataset):
    def __init__(self, corpus: tp.List[tp.List[str]],
                 coocurance: np.array, count_in_line: int):
        self.logger = get_logger(VanillaWord2VecWithCoocuranceDataset.__name__)
        (self.positive_index_rows,
         self.positive_index_cols) = np.where(coocurance == 1)
        self.count_in_line = count_in_line
        self.corpus = corpus

    def __len__(self):
        return int(len(self.corpus)*0.01)

    def __getitem__(self, item: int) -> tp.Tuple[torch.LongTensor,
                                                 torch.LongTensor]:
        ixs = np.random.choice(
            range(self.positive_index_rows.shape[0]),
            size=self.count_in_line,
            replace=False)

        return (
            torch.LongTensor(self.positive_index_rows[ixs][:, None]),
            torch.LongTensor(self.positive_index_cols[ixs][:, None]),
        )


class NegativeSamplingWord2VecDataset(Dataset):
    def __init__(self,
                 corpus: tp.List[tp.List[str]],
                 freq_dictionary: tp.Dict[str, float],
                 dictionary: tp.Dict[str, int],
                 window: int,
                 count_in_line: int):
        super().__init__()
        self.logger = get_logger(NegativeSamplingWord2VecDataset.__name__)
        self.corpus = corpus
        self.freq_dictionary = freq_dictionary
        self.dictionary = dictionary
        self.window = window
        self.count_in_line = count_in_line

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, idx) -> tp.Tuple[torch.LongTensor,
                                           torch.LongTensor,
                                           torch.FloatTensor]:
        line = self.corpus[idx]
        if len(line) - self.count_in_line < 12:
            count = 3
        else:
            count = self.count_in_line
        pos_samples = positive_sampling(line, self.window, count)
        neg_samples = negative_sampling(self.freq_dictionary, pos_samples)

        X = np.zeros(shape=(len(pos_samples) + len(neg_samples), 2))
        y = np.zeros(shape=(len(pos_samples) + len(neg_samples),))
        for ix, pair in enumerate(pos_samples):
            X[ix, 0] = self.dictionary[pair[0]]
            X[ix, 1] = self.dictionary[pair[1]]

        for ix, pair in enumerate(neg_samples, len(pos_samples)):
            X[ix, 0] = self.dictionary[pair[0]]
            X[ix, 1] = self.dictionary[pair[1]]

        y[:len(pos_samples)] = 1
        y[len(pos_samples):] = 0

        ixs = np.array(range(X.shape[0]))
        np.random.shuffle(ixs)

        return (
            torch.LongTensor(X[ixs, 0][:, None]),
            torch.LongTensor(X[ixs, 1][:, None]),
            torch.FloatTensor(y[ixs, None])
        )


class NegativeSamplingWord2VecWithCoocuranceDataset(Dataset):
    def __init__(self,
                 corpus: tp.List[tp.List[str]],
                 coocurance: np.array,
                 freq_dictionary: tp.Dict[str, float],
                 vocab: tp.Dict[str, int],
                 count_in_line: int):
        self.logger = get_logger(
            NegativeSamplingWord2VecWithCoocuranceDataset.__name__)
        (self.positive_index_rows,
         self.positive_index_cols) = np.where(coocurance == 1)
        self.count_in_line = count_in_line
        self.corpus = corpus
        self.freq_dictionary = freq_dictionary
        self.vocab = vocab

    def __len__(self):
        return int(len(self.corpus) * 0.01)

    def __getitem__(self, item: int) -> tp.Tuple[
                        torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
        pos_ixs = np.random.choice(
            range(self.positive_index_rows.shape[0]),
            size=self.count_in_line,
            replace=False)

        neg_samples = negative_coocurance_sampling(self.freq_dictionary,
                                                   self.vocab,
                                                   self.count_in_line)

        X_word_ids = np.concatenate(
            [
                self.positive_index_rows[pos_ixs],
                np.asarray([el[0] for el in neg_samples])
            ],
            axis=0)

        X_context_ids = np.concatenate(
            [
                self.positive_index_cols[pos_ixs],
                np.asarray([el[1] for el in neg_samples])
            ],
            axis=0)

        pos_y = np.ones(shape=(self.count_in_line,))
        neg_y = np.zeros(shape=(self.count_in_line,))
        y = np.concatenate([pos_y, neg_y], axis=0)

        ixs = np.array(range(y.shape[0]))
        np.random.shuffle(ixs)

        return (
            torch.LongTensor(X_word_ids[ixs][:, None]),
            torch.LongTensor(X_context_ids[ixs][:, None]),
            torch.FloatTensor(y[ixs][:, None]),
        )
