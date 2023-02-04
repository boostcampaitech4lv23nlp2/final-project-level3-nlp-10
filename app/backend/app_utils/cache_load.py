from functools import lru_cache

import whisper
from app_utils.key_bert import KeywordBert
from model import DPRContextEncoder, DPRQuestionEncoder, FiD, FiD_DPR
from omegaconf import OmegaConf
from transformers import AutoTokenizer

conf = OmegaConf.load("./config.yaml")


@lru_cache
def load_model(model_type):
    if model_type == "fid":
        return FiD.from_pretrained(conf.fid.model_path)
    if model_type == "sbert":
        return KeywordBert()


@lru_cache
def load_retriever():
    q_encoder = DPRQuestionEncoder.from_pretrained("EJueon/keyword_dpr_question_encoder")
    p_encoder = DPRContextEncoder.from_pretrained("EJueon/keyword_dpr_context_encoder")
    tokenizer = load_tokenizer()[0]
    retriver = FiD_DPR(
        conf=conf, q_encoder=q_encoder, p_encoder=p_encoder, tokenizer=tokenizer, emb_save_path=conf.emb_save_path
    )
    return retriver


@lru_cache
def load_tokenizer():
    retriever_tokenizer = AutoTokenizer.from_pretrained(conf.fid.retriever_tokenizer)
    reader_tokenizer = AutoTokenizer.from_pretrained(conf.fid.reader_tokenizer)

    return (retriever_tokenizer, reader_tokenizer)


@lru_cache
def load_stt_model():
    return whisper.load_model("large")


@lru_cache
def load_sbert():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("jhgan/ko-sroberta-multitask")
