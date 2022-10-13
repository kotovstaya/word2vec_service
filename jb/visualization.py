import os
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import torch
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
from umap import UMAP

from preprocessing import get_dictionary, preprocess_corpus


def get_word2vec_embeddings_as_array(corpus_path: str,
                                     model_path: str) -> np.array:
    """

    :param corpus_path:
    :param model_path:
    :return:
    """
    model = Word2Vec.load(model_path)

    corpus = preprocess_corpus(corpus_path, limit=None)
    dict, invert_dict = get_dictionary(corpus)

    valid_words = []
    for ix, word in enumerate(dict.keys()):
        try:
            val = model.wv[word]
            valid_words.append(word)
        except:
            pass
    embeddings = np.zeros(shape=(len(valid_words), 150))
    for ix, word in enumerate(valid_words):
        try:
            embeddings[ix] = model.wv[word]
        except:
            print(f"word: {word}")
    return embeddings


def get_custom_word2vec_embeddings_as_array(model_path: str) -> np.array:
    """

    :param model_path:
    :return:
    """
    model = torch.load(model_path)
    print(model.get_embeddings()[0, :10])
    embeddings = model.get_embeddings()
    return embeddings


def get_reducer_embeddings(reducer_class: tp.Any, embeddings: np.array,
                           n_components: int) -> np.array:
    """

    :param reducer_class:
    :param embeddings:
    :param n_components:
    :return:
    """
    reducer = reducer_class(n_components=n_components)
    return reducer.fit_transform(embeddings)


def get_tsne_embeddings(embeddings: np.array, n_components: int) -> np.array:
    return get_reducer_embeddings(TSNE, embeddings, n_components)


def get_umap_embeddings(embeddings: np.array, n_components: int) -> np.array:
    return get_reducer_embeddings(UMAP, embeddings, n_components)


def plot_representation(embeddings: np.array, title: str,
                        filepath: str) -> None:
    """

    :param embeddings:
    :param title:
    :param filepath:
    :return:
    """
    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.scatter(embeddings[:, 0], embeddings[:, 1])
    plt.title(title)
    plt.xlabel("first dimension")
    plt.ylabel("second dimension")
    plt.savefig(filepath)


if __name__ == "__main__":
    root_folder = "./../data/"
    raw_corpus_fpath = os.path.join(root_folder, "raw_dataset.txt")
    model_fpath = os.path.join(root_folder, "from_box_word2vec_model")
    tsne_plot_fpath = os.path.join(root_folder, "from_box_tsne_2d_plot.png")
    umap_plot_fpath = os.path.join(root_folder, "from_box_umap_2d_plot.png")

    w2v_embeddings = get_word2vec_embeddings_as_array(raw_corpus_fpath,
                                                      model_fpath)

    embeddings_tsn_2d = get_tsne_embeddings(w2v_embeddings, n_components=2)
    embeddings_umap_2d = get_umap_embeddings(w2v_embeddings, n_components=2)
    plot_representation(embeddings_tsn_2d, "TSNE embeddings", tsne_plot_fpath)
    plot_representation(embeddings_umap_2d, "UMAP embeddings", umap_plot_fpath)