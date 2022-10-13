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
    Get Word2Vec model from gensim and get corpus of files.
    Extract embeddings for every word in vocabulary in on 2D array

    Args:
        :param corpus_path: where is file with all documents
        :param model_path: where is the trained model
    :return: array with words' embeddings
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
    Get Word2Vec custom model from
    Extract embeddings for every word in vocabulary in on 2D array

    Args:
        :param model_path: where is the trained model
    :return:
    """
    model = torch.load(model_path)
    print(model.get_embeddings()[0, :10])
    embeddings = model.get_embeddings()
    return embeddings


def get_reducer_embeddings(reducer_class: tp.Any, embeddings: np.array,
                           n_components: int) -> np.array:
    """
    For visualization purposes we should train a model that can help reduce
    dimensions of our embeddings to a smaller space.

    Args:
        :param reducer_class: class instance for reduction
        :param embeddings: array with embeddings in a N-dimension space
        :param n_components: what a dimension size will take our embeddings after reduction
    :return: trained model
    """
    reducer = reducer_class(n_components=n_components)
    return reducer.fit_transform(embeddings)


def get_tsne_embeddings(embeddings: np.array, n_components: int) -> np.array:
    """
    implement get_reducer_embeddings function with TSNE model

    Args:
        :param embeddings: array with embeddings in a N-dimension space
        :param n_components: what a dimension size will take our embeddings after reduction
    :return: trained model
    """
    return get_reducer_embeddings(TSNE, embeddings, n_components)


def get_umap_embeddings(embeddings: np.array, n_components: int) -> np.array:
    """
    implement get_reducer_embeddings function with UMAP model

    Args:
        :param embeddings: array with embeddings in a N-dimension space
        :param n_components: what a dimension size will take our embeddings after reduction
    :return: trained model
    """
    return get_reducer_embeddings(UMAP, embeddings, n_components)


def plot_representation(embeddings: np.array, title: str,
                        filepath: str) -> None:
    """
    Plot 2d embeddings

    Args:
        :param embeddings: array with 2d embeddings
        :param title: string with name of the plot
        :param filepath: where save this plot
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