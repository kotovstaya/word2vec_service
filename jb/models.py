import torch
from torch import nn


class CustomWord2Vec(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.lookup = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, word_vector, context_vector):
        wv = self.lookup(word_vector)[:, 0, :]
        cv = self.lookup(context_vector)[:, 0, :]
        out = self.sigmoid(torch.diagonal(torch.matmul(wv, cv.T)))
        return out

    def get_embeddings(self):
        return self.lookup.weight.detach().numpy()


