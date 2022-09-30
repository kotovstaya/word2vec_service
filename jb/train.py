import os
from preprocessing import preprocess_corpus
from gensim.models.word2vec import Word2Vec

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def train_model(
        vector_size: int,
        window: int,
        min_count: int,
        workers: int,
        sg: int,
        compute_loss: bool,
        progress_per: int,
        report_delay: int,
        epochs: int):
    model = Word2Vec(vector_size=vector_size,
                     window=window,
                     min_count=min_count,
                     workers=workers,
                     sg=sg,
                     compute_loss=compute_loss)
    model.build_vocab(corpus, progress_per=progress_per)
    model.train(corpus, total_examples=model.corpus_count, report_delay=report_delay, epochs=epochs)
    return model


def save_model(model, fpath):
    model.save(fpath)


if __name__ == "__main__":
    root_folder = "./../data/"
    model_save_fpath = os.path.join(root_folder, "word2vec_model")
    raw_corpus_fpath = os.path.join(root_folder, "raw_dataset.txt")

    corpus = preprocess_corpus(raw_corpus_fpath, limit=None)

    w2v = train_model(
        vector_size=150,
        window=5,
        min_count=50,
        workers=5,
        sg=1,
        compute_loss=True,
        progress_per=1,
        report_delay=1,
        epochs=50
    )
    save_model(w2v, model_save_fpath)