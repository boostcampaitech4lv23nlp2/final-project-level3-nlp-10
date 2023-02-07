import numpy as np
import torch
from app_utils.cache_load import load_model, load_retriever, load_tokenizer
from omegaconf import OmegaConf

conf = OmegaConf.load("./config.yaml")


def summarize_fid(keys: np.array, debug=False, renew_emb=True):
    (retriever_tokenizer, reader_tokenizer) = load_tokenizer()
    retriever = load_retriever()
    model = load_model(model_type="fid", device=conf.device)
    model.to(conf.device)
    queries = [retriever_tokenizer.sep_token.join(key) for key in keys]
    tokenized_query = retriever_tokenizer(
        queries, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    )

    top_docs = get_top_docs(retriever, tokenized_query)
    inputs = []
    for query, passages in zip(queries, top_docs):
        input = [f"{reader_tokenizer.bos_token}질문: {query} 문서: {doc}{reader_tokenizer.eos_token}" for doc in passages]
        inputs.append(
            reader_tokenizer(input, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        )

    input_ids = torch.stack([item["input_ids"] for item in inputs], dim=0).to(conf.device)
    attention_mask = torch.stack([item["attention_mask"] for item in inputs], dim=0).to(conf.device)
    output = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_length=512, num_beams=5, early_stopping=True
    )
    outputs = [reader_tokenizer.decode(output_slice, skip_special_tokens=True) for output_slice in output]
    # output = reader_tokenizer.decode(output, skip_special_tokens=True)
    if debug:
        for input_ids in input_ids[0]:
            print(reader_tokenizer.decode(input_ids, skip_special_tokens=True))
        print(outputs[0])
    model.to("cpu")
    print(model.device)
    return outputs


def get_top_docs(retriever, query):
    # top_doc_indices: (1, topk) tensor
    top_doc_indices = retriever.get_relevant_doc(query, k=conf.topk)
    # top_docs: (1, topk) List
    top_docs = retriever.get_passage_by_indices(top_doc_indices)
    return top_docs
