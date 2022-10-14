import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sys

sys.path.insert(1, './../jb')

import inference
import preprocessing

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODELS = {
    "box_inferencer": None,
    "vanilla_inferencer": None,
    "ns_inferencer": None
}


def update_model_dict(model_fpath, vanilla_model_fpath, ns_model_fpath, vocab, inverse_vocab):
    global MODELS
    if MODELS["box_inferencer"] is None:
        MODELS["box_inferencer"] = inference.FromBoxInference(model_fpath)
        MODELS["box_inferencer"].load_model()

        MODELS["vanilla_inferencer"] = inference.CustomVanillaInference(
            vanilla_model_fpath, vocab, inverse_vocab)
        MODELS["vanilla_inferencer"].load_model()

        MODELS["ns_inferencer"] = inference.CustomNSInference(ns_model_fpath,
                                                              vocab,
                                                              inverse_vocab)
        MODELS["ns_inferencer"].load_model()


def get_predictions(search_word):
    global MODELS
    similarity_gensim = MODELS["box_inferencer"].predict_topn(search_word)
    vanilla_similarity = MODELS["vanilla_inferencer"].predict_topn(search_word)
    ns_similarity = MODELS["ns_inferencer"].predict_topn(search_word)
    return similarity_gensim, vanilla_similarity, ns_similarity


@app.get("/")
async def root():
    return {"message": "Hello"}


@app.get("/predictions/{item}", response_class=HTMLResponse)
async def root(request: Request, item: str):
    global MODELS

    search_word = item

    root_folder = "./../data/"
    raw_corpus_fpath = os.path.join(root_folder, "raw_dataset.txt")
    model_fpath = os.path.join(root_folder, "from_box_word2vec_model")
    vanilla_model_fpath = os.path.join(root_folder,
                                       "coocurance_vanilla_word2vec_model.pt")
    ns_model_fpath = os.path.join(root_folder, "coocurance_ns_word2vec_model")

    corpus = preprocessing.preprocess_corpus(raw_corpus_fpath, limit=None)
    vocab, inverse_vocab = preprocessing.get_dictionary(corpus)
    update_model_dict(model_fpath,
                      vanilla_model_fpath,
                      ns_model_fpath,
                      vocab,
                      inverse_vocab)

    similarity_gensim, vanilla_similarity, ns_similarity = get_predictions(
        search_word)

    return templates.TemplateResponse("main.html",
                                      {
                                          "request": request,
                                          "search_word": search_word,
                                          "similarity_gensim": similarity_gensim,
                                          "vanilla_similarity": vanilla_similarity,
                                          "ns_similarity": ns_similarity
                                      })
