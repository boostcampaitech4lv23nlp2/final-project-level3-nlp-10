from typing import List

import uvicorn
from api import CSR
from fastapi import APIRouter, FastAPI, File
from keybert import keybert_keyword

app = FastAPI()
stt_router = APIRouter(prefix="/stt")


@app.on_event("startup")
def startup_event():
    print("Start Boost2Note Server")


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


@app.post("/keyword/")
async def get_keyword(text: str):
    keywords = keybert_keyword(text, 5)
    return keywords


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
