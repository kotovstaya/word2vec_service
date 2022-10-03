import typing as tp
import numpy as np
import torch
from torch.utils.data import Dataset
from preprocessing import positive_sampling, negative_sampling


class NegativeSamplingWord2VecDataset(Dataset):
    def __init__(self,
                 corpus: tp.List[tp.List[str]],
                 freq_dictionary: tp.Dict[str, float],
                 dictionary: tp.Dict[str, int],
                 window: int,
                 count_in_line: int):
        super().__init__()
        self.corpus = corpus
        self.freq_dictionary = freq_dictionary
        self.dictionary = dictionary
        self.window = window
        self.count_in_line = count_in_line

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, idx) -> tp.Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
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
