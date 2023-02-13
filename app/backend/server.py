from typing import List

import time

import numpy as np
import uvicorn
from app_utils import (
    create_context_embedding,
    get_keyword,
    load_model,
    load_retriever,
    load_sbert,
    load_small_stt_model,
    predict_stt,
    split_passages,
    summarize_fid,
)
from fastapi import FastAPI, File
from omegaconf import OmegaConf
from pydantic import BaseModel

"""
Mecab 설치
    python3 -m pip install konlpy
    sudo apt-get install curl git
    bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
(참조: https://konlpy.org/en/latest/install/)
"""
app = FastAPI()
conf = OmegaConf.load("./config.yaml")


class SummarizeRequest(BaseModel):
    keywords: list
    emb_name: str


class SummarizeResponse(BaseModel):
    summarization: list


@app.on_event("startup")
def startup_event():
    print("Start Boost2Note Server")
    load_model(model_type="fid")
    load_model(model_type="sbert")
    load_retriever()
    print("FiD model loaded")
    load_small_stt_model()
    print("Whisper model loaded")
    load_sbert()
    print("success to load model")


@app.on_event("shutdown")
def shutdown_envent():
    print("Shutdown Boost2Note")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/stt")
async def get_passages(files: List[bytes] = File()) -> List:
    print("stt started")
    results = []
    keywords = set()
    passages = set()
    for file in files:
        result = predict_stt(file)
        result = split_passages(result)

        if passages and keywords:
            passages.update(result)
            keywords.update(get_keyword(result, device=conf.device))
        else:
            passages = set(result)
            keywords = get_keyword(result, device=conf.device)

        results.append(result)
    emb_name = str(time.time())
    create_context_embedding(list(passages), renew_emb=True, emb_name=emb_name)
    print("stt finished")
    return [results, list(keywords), emb_name]


@app.post("/summarize")
async def summarize_text(request: SummarizeRequest, response_model=SummarizeResponse):
    print("summarize started")
    request_dict = dict(request)
    keywords = np.array(request_dict["keywords"])
    emb_name = request_dict["emb_name"]
    keyword_list = [keywords.tolist()]
    inputs = [keywords]

    outputs, top_docs = summarize_fid(inputs, debug=False, renew_emb=False, emb_name=emb_name)

    top_docs = {doc for docs in top_docs for doc in docs}

    results = [keyword_list, outputs, list(top_docs)]
    print("summarize finished")
    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30002)
