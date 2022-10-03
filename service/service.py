import os
from fastapi import FastAPI
import jb.inference as inference
import jb.preprocessing as preprocessing

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "hello"}


@app.get("/frombox/{item}")
async def frombox_get_top_n(item: str):
    root_folder = "./../data/"
    model_fpath = os.path.join(root_folder, "word2vec_model")
    inferencer = inference.FromBoxInference(model_fpath)
    inferencer.load_model()
    similarity = inferencer.predict_topn(item)

    return {"message": similarity}


@app.get("/custom/{item}")
async def custom_get_top_n(item: str):
    root_folder = "./../data/"
    model_fpath = os.path.join(root_folder, "custom_word2vec_model")
    raw_corpus_fpath = os.path.join(root_folder, "raw_dataset.txt")
    corpus = preprocessing.preprocess_corpus(raw_corpus_fpath, limit=None)
    vocab, _ = preprocessing.get_dictionary(corpus)
    inferencer = inference.CustomInference(model_fpath, vocab)
    inferencer.load_model()
    similarity = inferencer.predict_topn(item)

    return {"message": similarity}
