from typing import List

import numpy as np
import uvicorn
from app_utils import (
    create_context_embedding,
    get_keyword,
    load_model,
    load_retriever,
    load_sbert,
    load_stt_model,
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


class Keywords(BaseModel):
    keywords: list


class SummarizeResponse(BaseModel):
    summarization: list


@app.on_event("startup")
def startup_event():
    print("Start Boost2Note Server")
    load_model(model_type="fid")
    load_model(model_type="sbert")
    load_retriever()
    print("FiD model loaded")
    load_stt_model()
    load_sbert()
    print("success to loading whisper model")


@app.on_event("shutdown")
def shutdown_envent():
    print("Shutdown Boost2Note")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/stt/")
async def get_passages(files: List[bytes] = File()) -> List:
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
    create_context_embedding(list(passages), renew_emb=True)
    return [results, list(keywords)]


@app.post("/summarize")
async def summarize_text(keywords: Keywords, response_model=SummarizeResponse):
    sample_num = 3
    dict_keys = dict(keywords)
    keywords = np.array(dict_keys["keywords"])
    print(keywords)
    inputs = [keywords]
    for _ in range(sample_num - 1):
        inputs.append(keywords[np.random.choice(len(keywords), len(keywords) - 1, replace=False)])
    outputs = summarize_fid(inputs, debug=True, renew_emb=False)
    return outputs


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
