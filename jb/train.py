import os
import typing as tp
from gensim.models.word2vec import Word2Vec
from torch.utils.data import DataLoader
from models import NegativeSamplingWord2Vec, VanillaWord2Vec
from datasets import NegativeSamplingWord2VecDataset
import torch
import tqdm
from torch import nn
import torch.optim as optim
from preprocessing import (
    preprocess_corpus,
    get_dictionary,
    get_word_probability_dict,
    get_logger)


class BaseTrainer:
    def __init__(self, epochs):
        self.epochs = epochs

    def _init_model(self, *args, **kwargs):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError

    def save_model(self, fpath):
        raise NotImplementedError


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


class CustomVanillaTrainer(BaseTrainer):
    def __init__(self,
                 dataloader: DataLoader,
                 vocab_size: int,
                 emb_dim: int,
                 lr: float,
                 epochs: int):
        self.logger = get_logger(CustomVanillaTrainer.__name__)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.epochs = epochs
        self.dataloader = dataloader
        self.model = self._init_model(self.vocab_size, self.emb_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def _init_model(self, vocab_size, emb_dim):
        self.logger.warning("init model")
        return VanillaWord2Vec(vocab_size, emb_dim)

    def train_model(self):
        self.logger.warning("start training")
        for epoch in range(self.epochs):
            running_loss = 0.0
            counter = 0
            for i, (words, contexts, labels) in tqdm.tqdm(enumerate(self.dataloader)):
                self.optimizer.zero_grad()

                outputs = self.model.forward(words)
                loss = self.criterion(outputs, labels[:,0])
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                counter += 1
            self.logger.warning(f'[{epoch + 1}] loss: {running_loss / counter:.3f}')

    def save_model(self, fpath):
        self.logger.warning("save model")
        torch.save(self.model, fpath)


class CustomNegativeSamplingTrainer(BaseTrainer):
    def __init__(self,
                 dataloader: DataLoader,
                 vocab_size: int,
                 emb_dim: int,
                 lr: float,
                 epochs: int):
        self.logger = get_logger(CustomNegativeSamplingTrainer.__name__)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.epochs = epochs
        self.dataloader = dataloader
        self.model = self._init_model(self.vocab_size, self.emb_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def _init_model(self, vocab_size, emb_dim):
        self.logger.warning("init model")
        return NegativeSamplingWord2Vec(vocab_size, emb_dim)

    def train_model(self):
        self.logger.warning("start training")
        for epoch in range(self.epochs):
            running_loss = 0.0
            counter = 0
            for i, (words, contexts, labels) in tqdm.tqdm(enumerate(self.dataloader)):
                self.optimizer.zero_grad()

                outputs = self.model.forward(words, contexts)
                loss = self.criterion(outputs, labels[:,0])
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                counter += 1
            self.logger.warning(f'[{epoch + 1}] loss: {running_loss / counter:.3f}')

    def save_model(self, fpath):
        self.logger.warning("save model")
        torch.save(self.model, fpath)


if __name__ == "__main__":
    root_folder = "./../data/"
    model_save_fpath = os.path.join(root_folder, "word2vec_model")
    raw_corpus_fpath = os.path.join(root_folder, "raw_dataset.txt")
    custom_model_save_fpath = os.path.join(root_folder, "custom_word2vec_model")

    corpus = preprocess_corpus(raw_corpus_fpath, limit=None)
    dictionary, _ = get_dictionary(corpus)
    freq_dict = get_word_probability_dict(corpus)

    # fbt = FromBoxTrainer(
    #     vector_size=150,
    #     window=5,
    #     min_count=50,
    #     workers=5,
    #     sg=1,
    #     compute_loss=True,
    #     progress_per=1,
    #     report_delay=1,
    #     epochs=50
    # )
    #
    # fbt.train_model()
    # fbt.save_model(model_save_fpath)

    ds = NegativeSamplingWord2VecDataset(
        corpus=corpus,
        freq_dictionary=freq_dict,
        dictionary=dictionary,
        window=5,
        count_in_line=8,
    )

    def my_collate(batch):
        a = [el[0] for el in batch]
        b = [el[1] for el in batch]
        c = [el[2] for el in batch]
        return (
            torch.concat(a, dim=0),
            torch.concat(b, dim=0),
            torch.concat(c, dim=0)
        )

    dataloader = DataLoader(ds, batch_size=30, shuffle=True, collate_fn=my_collate, num_workers=0)
    ct = CustomNegativeSamplingTrainer(
        dataloader=dataloader,
        vocab_size=len(dictionary.keys()),
        emb_dim=100,
        lr=1e-4,
        epochs=30
    )
    ct.train_model()
    ct.save_model(custom_model_save_fpath)
