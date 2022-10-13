import torch
from torch import nn


class VanillaWord2Vec(nn.Module):
    """

    """
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.lookup = nn.Embedding(num_embeddings=self.vocab_size,
                                   embedding_dim=self.emb_dim, max_norm=1.)
        self.fc = nn.Linear(self.emb_dim, vocab_size)
        # self.softmax = nn.Softmax()

    def forward(self, word_vector):
        wv = self.lookup(word_vector)[:, 0, :]
        out = self.fc(wv)
        return out

    def get_embeddings(self):
        return self.lookup.weight.detach().numpy()


class NegativeSamplingWord2Vec(nn.Module):
    """

    """
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.lookup = nn.Embedding(num_embeddings=self.vocab_size,
                                   embedding_dim=self.emb_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, word_vector, context_vector):
        wv = self.lookup(word_vector)
        cv = self.lookup(context_vector)
        out = torch.bmm(wv, cv.permute(0, 2, 1)).squeeze().sigmoid()
        return out

    def get_embeddings(self):
        return self.lookup.weight.detach().numpy()
