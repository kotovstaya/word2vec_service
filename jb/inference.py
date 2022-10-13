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
        return self.model.wv.most_similar(word, topn=topn)


class CustomInference(BaseInference):
    def __init__(self, model_fpath: str, vocab: tp.Dict[str, int]):
        super().__init__(model_fpath)
        self.vocab = vocab

    def load_model(self):
        self.model = torch.load(self.model_fpath)
        self.model.eval()

    def predict_topn(self, word: str, topn: tp.Optional[int] = 10):
        word_idx = self.vocab[word]
        embs_shape = self.model.get_embeddings().shape[0]
        words = np.zeros(shape=(embs_shape,))
        words[:] = word_idx
        contexts = np.array(range(embs_shape))

        words = torch.LongTensor(words)
        contexts = torch.LongTensor(contexts)
        preds = self.model.forward(words, contexts)
        preds = preds.detach().numpy()
        return preds


if __name__ == "__main__":
    root_folder = "./../data/"
    model_fpath = os.path.join(root_folder, "custom_word2vec_model")
    raw_corpus_fpath = os.path.join(root_folder, "raw_dataset.txt")
    corpus = preprocessing.preprocess_corpus(raw_corpus_fpath, limit=None)
    vocab, _ = preprocessing.get_dictionary(corpus)
    inferencer = CustomInference(model_fpath, vocab)
    inferencer.load_model()
    probs = inferencer.predict_topn("function")
    print(probs.shape)

