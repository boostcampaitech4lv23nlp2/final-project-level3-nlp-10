from typing import List, Tuple

# import gzip
import os
import pickle
import random
import re

import kss
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_metric
from transformers import (
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


# pickle load 후 ids, dialogues, summaries로 분리
def load_pickle_data(path: str) -> Tuple[List[str], List[List[str]], List[str]]:
    # with gzip.open(path, "rb") as f:
    #     data = pickle.load(f)
    with open(path, "rb") as f:
        data = pickle.load(f)

    ids, dialogues, summaries, queries = [], [], [], []

    ids.extend(data["ids"])

    for idx in range(len(data)):
        passage = data["passage"][idx]
        query = "<sep>".join(data["keybert"][idx])
        queries.append(query)

        if data["source"][idx] == "dacon":  # dacon
            passage = re.sub(r"\([^)]*\)", "", passage)  # 괄호문 제거
            dialogues.append(passage.split("화자]"))
        elif "speech" in data["ids"][idx]:  # report-speech
            passage_split = re.split(r"\([^)]*\)|\<[^>]*\>", passage)  # 화자와 발화문 분리 "(화자) , <화자> 제거"
            splits = []
            for i in range(len(passage_split)):
                sentence = passage_split[i].strip(" ")  # 양 쪽 공백 제거
                if sentence != "":  # 빈 문자열 무시
                    splits.append(sentence)
            dialogues.append(splits)
        else:  # report-minute, broadcast
            passage_split = re.split(r"\n", passage)  # 발화문별 분리
            splits = []
            for i in range(len(passage_split)):
                sentence = passage_split[i]
                sentence = re.sub(r"^.*]", "", sentence)  # 화자] 제거
                sentence = sentence.strip('" ')
                if sentence != "":  # 빈 문자열 무시
                    splits.append(sentence)
            dialogues.append(splits)

    summaries.extend([summary["summary1"] for summary in data["annotation"]])

    return ids, dialogues, summaries, queries


def preprocess(dialogue_data):
    for dialogues in dialogue_data:
        for i, d in enumerate(dialogues):
            dialogues[i] = re.sub(" +", " ", d)

    for idx in range(len(dialogue_data)):
        dialogue_data[idx] = "<sep>".join(dialogue_data[idx])

    return dialogue_data


def add_s_token(summary_data):
    for sample_idx in range(len(summary_data)):
        # dialogue_data[sample_idx] = f"<s>{dialogue_data[sample_idx]}</s>"
        summary_data[sample_idx] = f"<s>{summary_data[sample_idx]}</s>"
    return summary_data


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["passage"], max_length=max_input_length, truncation=True
    )  # padding은 data_collator에서 생성할 것

    # tokenizer for targets
    with tokenizer.as_target_tokenizer():  # kobart의 경우, 동일한 tokenizer를 사용
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them. (label_pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(kss.split_sentences(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(kss.split_sentences(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def summarization(text):
    tokenized_text = tokenizer(text, return_tensors="pt", truncation=False).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            tokenized_text["input_ids"],
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            max_length=512,
            # top_p = 0.7,
            # top_k = 20,
            num_beams=20,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    seed_everything(42)
    train_dataset_path = "../data/train_query_v10_keybert.pickle"
    valid_dataset_path = "../data/valid_query_v10_keybert.pickle"
    train_ids, train_dialogues, train_summaries, train_queries = load_pickle_data(train_dataset_path)
    valid_ids, valid_dialogues, valid_summaries, valid_queries = load_pickle_data(valid_dataset_path)

    # tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
    # model = BartForConditionalGeneration.from_pretrained("hyunwoongko/kobart")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")
    model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2").to("cuda")

    # print(train_dialogues[:3])
    train_dialogues = preprocess(train_dialogues)
    valid_dialogues = preprocess(valid_dialogues)

    if tokenizer.name_or_path == "gogamza/kobart-base-v2":
        train_summaries = add_s_token(train_summaries)
        valid_summaries = add_s_token(valid_summaries)

    final_train_dialogues = [f"<s>{train_queries[i]}<sep>{train_dialogues[i]}</s>" for i in range(len(train_ids))]
    final_valid_dialogues = [f"<s>{valid_queries[i]}<sep>{valid_dialogues[i]}</s>" for i in range(len(valid_ids))]

    # print(train_dialogues[:3])
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"passage": final_train_dialogues, "summary": train_summaries}),
            "valid": Dataset.from_dict({"passage": final_valid_dialogues, "summary": valid_summaries}),
        }
    )

    tokenizer.add_special_tokens({"sep_token": "<sep>"})
    added_token_num = 1

    max_input_length = 1024
    max_target_length = 1024

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)

    batch_size = 8
    args = Seq2SeqTrainingArguments(
        output_dir="gogamza_v11_sep_data_1024_keyword_linear",
        run_name="gogamza_v11_sep_data_1024_keyword_linear",
        evaluation_strategy="steps",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        save_steps=1000,
        eval_steps=1000,
        lr_scheduler_type="linear",  # "cosine_with_restarts"
        # fp16=True,  # Use mixed precision
        # fp16_opt_level="O1",  # mixed precision mode
        report_to="wandb",
        # dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rouge1",
        # data_sampler로 랜덤으로 데이터 들어가는지? bucketing?
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric = load_metric("rouge")
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model()  # model_output/pytorch_model.bin, config.json
