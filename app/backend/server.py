from typing import List

import uvicorn
from app_utils import load_model, load_retriever, load_stt_model, summarize_fid
from fastapi import FastAPI, File

app = FastAPI()


@app.on_event("startup")
def startup_event():
    print("Start Boost2Note Server")
    load_model(model_type="fid")
    load_retriever()
    print("FiD model loaded")
    summarize_fid(["앙팡", "두유", "서울우유"])


@app.on_event("shutdown")
def shutdown_envent():
    print("Shutdown Boost2Note")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/stt/")
async def get_stt(files: List[bytes] = File(...)):
    model = load_stt_model()
    result = model.transcribe("./data/피어세션 2023-01-31 수정.mp3")
    return result


@app.post("/summarize/")
async def summarize_text(files: List[str]):
    print(files)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
