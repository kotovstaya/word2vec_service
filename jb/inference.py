import os
import typing as tp

import numpy as np
import torch
from gensim.models.word2vec import Word2Vec

import preprocessing


class BaseInference:
    def __init__(self, model_fpath: str):
        self.model_fpath = model_fpath
        self.model = None

    def load_model(self):
        raise NotImplementedError

    def predict_topn(self, word: str, topn: tp.Optional[int] = 10):
        raise NotImplementedError


class FromBoxInference(BaseInference):
    def __init__(self, model_fpath: str):
        super().__init__(model_fpath)

    def load_model(self):
        self.model = Word2Vec.load(self.model_fpath)

    def predict_topn(self, word: str, topn: tp.Optional[int] = 10):
        candidates = self.model.wv.most_similar(word, topn=topn)
        return [[el[0], np.round(el[1], decimals=3)] for el in candidates]


class CustomNSInference(BaseInference):
    def __init__(self,
                 model_fpath: str,
                 vocab: tp.Dict[str, int],
                 inverse_vocab: tp.Dict[int, str]):
        super().__init__(model_fpath)
        self.vocab = vocab
        self.inverse_vocab = inverse_vocab

    def load_model(self):
        self.model = torch.load(self.model_fpath)
        self.model.eval()

    def predict_topn(self, word: str, topn: tp.Optional[int] = 10):
        word_idx = self.vocab[word]
        embs_shape = self.model.get_embeddings().shape[0]
        words = np.zeros(shape=(embs_shape,))
        words[:] = word_idx
        contexts = np.array(range(embs_shape))

        words = torch.LongTensor(words)[:, None]
        contexts = torch.LongTensor(contexts)[:, None]
        with torch.no_grad():
            preds = self.model.forward(words, contexts)
            preds = preds.detach().numpy()
        return sorted(
            [(self.inverse_vocab[idx], el) for idx, el in enumerate(preds)],
            key=lambda x: x[1]
        )[::-1][:topn]


class CustomVanillaInference(BaseInference):
    def __init__(self,
                 model_fpath: str,
                 vocab: tp.Dict[str, int],
                 inverse_vocab: tp.Dict[int, str]):
        super().__init__(model_fpath)
        self.vocab = vocab
        self.inverse_vocab = inverse_vocab

    def load_model(self):
        self.model = torch.load(self.model_fpath)
        self.model.eval()

    def predict_topn(self, word: str, topn: tp.Optional[int] = 10):
        word_idx = self.vocab[word]
        words = torch.LongTensor([word_idx])[None, :]
        with torch.no_grad():
            preds = self.model.forward(words)
            preds = preds.detach().numpy()[0,:]
        return sorted(
            [(self.inverse_vocab[idx], el) for idx, el in enumerate(preds)],
            key=lambda x: x[1]
        )[::-1][:topn]


if __name__ == "__main__":
    root_folder = "./../data/"
    model_fpath = os.path.join(root_folder, "custom_word2vec_model")
    raw_corpus_fpath = os.path.join(root_folder, "raw_dataset.txt")
    corpus = preprocessing.preprocess_corpus(raw_corpus_fpath, limit=None)
    vocab, invers_vocab = preprocessing.get_dictionary(corpus)
    inferencer = CustomNSInference(model_fpath, vocab, invers_vocab)
    inferencer.load_model()
    probs = inferencer.predict_topn("function")

