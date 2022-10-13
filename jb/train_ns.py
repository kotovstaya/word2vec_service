import os

import click
import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from datasets import (NegativeSamplingWord2VecDataset,
                      NegativeSamplingWord2VecWithCoocuranceDataset)
from models import NegativeSamplingWord2Vec
from preprocessing import (
    preprocess_corpus,
    get_dictionary,
    get_word_probability_dict,
    coocurance_martix)
from train_base import BaseTrainer


class CustomNegativeSamplingTrainer(BaseTrainer):
    """

    """
    def __init__(self,
                 dataloader: DataLoader,
                 vocab_size: int,
                 emb_dim: int,
                 lr: float,
                 epochs: int):
        super().__init__(epochs)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.dataloader = dataloader
        self.model = self._init_model(self.vocab_size, self.emb_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def _init_model(self, vocab_size: int, emb_dim: int):
        self.logger.warning("init model")
        return NegativeSamplingWord2Vec(vocab_size, emb_dim)

    def train_model(self):
        self.logger.warning("start training")
        for epoch in range(self.epochs):
            running_loss = 0.0
            counter = 0
            for i, (words, contexts, labels) in tqdm.tqdm(
                                                    enumerate(self.dataloader)):
                self.optimizer.zero_grad()

                outputs = self.model.forward(words, contexts)
                loss = self.criterion(outputs, labels[:,0])
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                counter += 1
            self.logger.warning(
                f"[{epoch + 1}] loss: {running_loss / counter:.3f}")

    def save_model(self, fpath):
        self.logger.warning("save model")
        torch.save(self.model, fpath)


@click.group()
def cli():
    pass


@click.command()
@click.option('--window-size', default=5)
@click.option('--count-in-line', default=8)
@click.option('--batch-size', default=512)
@click.option('--emb-dim', default=150)
@click.option('--epochs', default=30)
def simple_ns(window_size: int,
              count_in_line: int,
              batch_size: int,
              emb_dim: int,
              epochs: int,):
    model_fpath = os.path.join(root_folder, "simple_ns_word2vec_model")
    ds = NegativeSamplingWord2VecDataset(
        corpus=corpus,
        freq_dictionary=freq_dict,
        dictionary=dictionary,
        window=window_size,
        count_in_line=count_in_line,
    )

    def my_collate(batch):
        return (
            torch.concat([el[0] for el in batch], dim=0),
            torch.concat([el[1] for el in batch], dim=0),
            torch.concat([el[2] for el in batch], dim=0)
        )

    dataloader = DataLoader(ds,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=my_collate,
                            num_workers=0)
    ct = CustomNegativeSamplingTrainer(
        dataloader=dataloader,
        vocab_size=len(dictionary.keys()),
        emb_dim=emb_dim,
        lr=1e-3,
        epochs=epochs
    )

    ct.train_model()
    ct.save_model(model_fpath)


@click.option('--count-in-line', default=256)
@click.option('--batch-size', default=1)
@click.option('--emb-dim', default=150)
@click.option('--epochs', default=30)
@click.command()
def coocurance_ns(count_in_line: int,
                  batch_size: int,
                  emb_dim: int,
                  epochs: int):
    model_fpath = os.path.join(root_folder, "coocurance_ns_word2vec_model")
    try:
        mtx = np.load(os.path.join(root_folder, "coocurance_matrix.npy"))
    except Exception as ex:
        print(ex)
        mtx = coocurance_martix(corpus, dictionary, window=5)
        N = mtx.shape[0]
        mtx[range(N), range(N)] = -1
        np.save(os.path.join(root_folder,"coocurance_matrix.npy"), mtx)
        mtx = np.load(os.path.join(root_folder, "coocurance_matrix.npy"))

    ds = NegativeSamplingWord2VecWithCoocuranceDataset(
        corpus=corpus,
        coocurance=mtx,
        count_in_line=count_in_line,
        vocab=dictionary,
        freq_dictionary=freq_dict,
    )

    def my_collate(batch):
        return batch[0][0], batch[0][1], batch[0][2]

    dataloader = DataLoader(ds,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=my_collate,
                            num_workers=0)
    ct = CustomNegativeSamplingTrainer(
        dataloader=dataloader,
        vocab_size=len(dictionary.keys()),
        emb_dim=emb_dim,
        lr=1e-2,
        epochs=epochs
    )
    ct.train_model()
    ct.save_model(model_fpath)


cli.add_command(simple_ns)
cli.add_command(coocurance_ns)


if __name__ == "__main__":
    root_folder = "./../data/"
    raw_corpus_fpath = os.path.join(root_folder, "raw_dataset.txt")

    corpus = preprocess_corpus(raw_corpus_fpath, limit=None)
    dictionary, _ = get_dictionary(corpus)
    freq_dict = get_word_probability_dict(corpus)

    cli()
