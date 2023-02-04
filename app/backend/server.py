from typing import List

import uvicorn
from api import CSR
from app_utils.cache_load import load_model, load_retriever
from app_utils.inference import summarize_fid
from fastapi import APIRouter, FastAPI, File
from pydantic import BaseModel

app = FastAPI()
stt_router = APIRouter(prefix="/stt")


class Keywords(BaseModel):
    keywords: list


@app.on_event("startup")
def startup_event():
    print("Start Boost2Note Server")
    load_model(model_type="fid")
    load_retriever()
    print("FiD model loaded")


@app.on_event("shutdown")
def shutdown_envent():
    print("Shutdown Boost2Note")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/stt/")
async def get_stt(files: List[bytes] = File(...)):
    result = CSR(files[0])
    return result


@app.post("/save/")
async def save_stt_text(files: List[str]):
    print(files)
    return {"test": "STT"}


@app.post("/summarize")
async def summarize_text(keywords: Keywords):
    dict_keys = dict(keywords)
    print(dict_keys)
    output = summarize_fid(dict_keys["keywords"], debug=True, renew_emb=False)
    print(output)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
