import os
import typing as tp

import click
from gensim.models.word2vec import Word2Vec

from preprocessing import preprocess_corpus
from train_base import BaseTrainer


class FromBoxTrainer(BaseTrainer):
    def __init__(
            self,
            vector_size: int,
            window: int,
            min_count: int,
            workers: int,
            sg: int,
            compute_loss: bool,
            progress_per: int,
            report_delay: int,
            epochs: int,
            corpus: tp.List[tp.List[str]]):
        super().__init__(epochs)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.compute_loss = compute_loss
        self.progress_per = progress_per
        self.report_delay = report_delay
        self.corpus = corpus

        self.model = self._init_model()

    def _init_model(self):
        model = Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            compute_loss=self.compute_loss)
        model.build_vocab(self.corpus, progress_per=self.progress_per)
        return model

    def train_model(self):
        self.model.train(self.corpus,
                         total_examples=self.model.corpus_count,
                         report_delay=self.report_delay,
                         epochs=self.epochs)

    def save_model(self, fpath):
        self.model.save(fpath)


@click.group()
def cli():
    pass


@click.command()
@click.option('--window-size', default=5)
@click.option('--emb-dim', default=150)
@click.option('--epochs', default=30)
@click.option('--sg', default=1)
def from_box(window_size: int,
             emb_dim: int,
             epochs: int,
             sg: int):
    model_save_fpath = os.path.join(root_folder, "from_box_word2vec_model")

    fbt = FromBoxTrainer(
        vector_size=emb_dim,
        window=window_size,
        min_count=50,
        workers=5,
        sg=sg,
        compute_loss=True,
        progress_per=1,
        report_delay=1,
        epochs=epochs,
        corpus=corpus
    )

    fbt.train_model()
    fbt.save_model(model_save_fpath)


cli.add_command(from_box)


if __name__ == "__main__":
    root_folder = "./../data/"
    raw_corpus_fpath = os.path.join(root_folder, "raw_dataset.txt")
    corpus = preprocess_corpus(raw_corpus_fpath, limit=None)

    cli()
