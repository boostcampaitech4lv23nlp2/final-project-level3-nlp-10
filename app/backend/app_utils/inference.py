from typing import List

from app_utils.cache_load import load_model, load_retriever, load_tokenizer
from app_utils.data_process import create_context_embedding
from omegaconf import OmegaConf

conf = OmegaConf.load("./config.yaml")


def summarize_fid(keys: List[str], debug=False, renew_emb=True):
    (retriever_tokenizer, reader_tokenizer) = load_tokenizer()
    retriever = load_retriever()
    model = load_model(model_type="fid")
    query = retriever_tokenizer.sep_token.join(keys)
    tokenized_query = retriever_tokenizer(
        [query], padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    )

    top_docs = get_top_docs(retriever, tokenized_query, renew_emb=renew_emb)[0]

    input = [f"{reader_tokenizer.bos_token}질문: {query} 문서: {doc}{reader_tokenizer.eos_token}" for doc in top_docs]
    tokenized_input = reader_tokenizer(
        input, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    )

    output = model.generate(
        input_ids=tokenized_input["input_ids"].unsqueeze(0),
        attention_mask=tokenized_input["attention_mask"].unsqueeze(0),
        max_length=512,
        num_beams=5,
        early_stopping=True,
    )[0]
    output = reader_tokenizer.decode(output, skip_special_tokens=True)
    if debug:
        for input_ids in tokenized_input["input_ids"]:
            print(reader_tokenizer.decode(input_ids, skip_special_tokens=True))
        print(output)
    return output


def get_top_docs(retriever, query, renew_emb=True):
    create_context_embedding("resources/meeting_records/output.json", renew_emb=renew_emb)
    # top_doc_indices: (1, topk) tensor
    top_doc_indices = retriever.get_relevant_doc(query, k=conf.topk)
    # top_docs: (1, topk) List
    top_docs = retriever.get_passage_by_indices(top_doc_indices)
    return top_docs
