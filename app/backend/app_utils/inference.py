from typing import List

from app_utils.server_setup import load_model, load_retriever, load_tokenizer


def summarize_fid(keys: List[str]):
    (retriever_tokenizer, reader_tokenizer) = load_tokenizer()
    retriever = load_retriever()
    model = load_model(model_type="FID")
    retriever
    model

    query = retriever_tokenizer.sep_token.join(keys)
    query = retriever_tokenizer([query], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
