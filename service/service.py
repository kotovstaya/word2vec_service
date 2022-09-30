import os
from fastapi import FastAPI
from gensim.models.word2vec import Word2Vec

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "hello"}


@app.get("/entities/{entity}")
async def get_top_n(entity):
    root_folder = "./../data/"
    model_fpath = os.path.join(root_folder, "word2vec_model")
    model = Word2Vec.load(model_fpath)
    similarity = model.wv.most_similar(entity, topn=10)
    return {"message": similarity}