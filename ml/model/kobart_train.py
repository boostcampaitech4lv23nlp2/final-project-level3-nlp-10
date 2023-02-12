from typing import List, Tuple

import os
import pickle
import random
import re

import kss
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_metric
from omegaconf import OmegaConf
from transformers import (
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


class KoBART:
    def __init__(self, config_path="../config/kobart_config.yaml"):
        self.config = OmegaConf.load(config_path)
        self.train_ids, self.train_dialogues, self.train_summaries, self.train_queries = load_pickle_data(
            self.config.path.train_dataset_path
        )
        self.valid_ids, self.valid_dialogues, self.valid_summaries, self.valid_queries = load_pickle_data(
            self.config.path.valid_dataset_path
        )
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.config.path.model_name_or_path)
        self.model = BartForConditionalGeneration.from_pretrained(self.config.path.model_name_or_path).to("cuda")
        self.metric = load_metric("rouge")
        seed_everything(self.config.others.seed)

    def preprocess(self, dialogue_data):
        for dialogues in dialogue_data:
            for i, d in enumerate(dialogues):
                dialogues[i] = re.sub(" +", " ", d)

        for idx in range(len(dialogue_data)):
            dialogue_data[idx] = "<sep>".join(dialogue_data[idx])

        return dialogue_data

    def add_s_token(self, summary_data):
        for sample_idx in range(len(summary_data)):
            # dialogue_data[sample_idx] = f"<s>{dialogue_data[sample_idx]}</s>"
            summary_data[sample_idx] = f"<s>{summary_data[sample_idx]}</s>"
        return summary_data

    def preprocess_function(self, examples):
        model_inputs = self.tokenizer(
            examples["passage"], max_length=self.config.train.max_input_length, truncation=True
        )  # padding은 data_collator에서 생성

        # tokenizer for targets
        with self.tokenizer.as_target_tokenizer():  # kobart의 경우, 동일한 tokenizer를 사용
            labels = self.tokenizer(
                examples["summary"], max_length=self.config.train.max_target_length, truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them. (label_pad_token_id)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(kss.split_sentences(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(kss.split_sentences(label.strip())) for label in decoded_labels]

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def train(self):
        self.train_dialogues = self.preprocess(self.train_dialogues)
        self.valid_dialogues = self.preprocess(self.valid_dialogues)

        if self.tokenizer.name_or_path == "gogamza/kobart-base-v2":
            self.train_summaries = self.add_s_token(self.train_summaries)
            self.valid_summaries = self.add_s_token(self.valid_summaries)

        final_train_dialogues = [
            f"<s>{self.train_queries[i]}<sep>{self.train_dialogues[i]}</s>" for i in range(len(self.train_ids))
        ]
        final_valid_dialogues = [
            f"<s>{self.valid_queries[i]}<sep>{self.valid_dialogues[i]}</s>" for i in range(len(self.valid_ids))
        ]

        dataset = DatasetDict(
            {
                "train": Dataset.from_dict({"passage": final_train_dialogues, "summary": self.train_summaries}),
                "valid": Dataset.from_dict({"passage": final_valid_dialogues, "summary": self.valid_summaries}),
            }
        )

        self.tokenizer.add_special_tokens({"sep_token": "<sep>"})
        added_token_num = 1

        tokenized_datasets = dataset.map(self.preprocess_function, batched=True)

        self.model.resize_token_embeddings(self.tokenizer.vocab_size + added_token_num)

        args = Seq2SeqTrainingArguments(
            output_dir=self.config.wandb.project_name,
            run_name=self.config.wandb.project_name,
            evaluation_strategy=self.config.train.evaluation_strategy,
            learning_rate=self.config.train.learning_rate,
            per_device_train_batch_size=self.config.train.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.train.per_device_eval_batch_size,
            save_total_limit=self.config.train.save_total_limit,
            num_train_epochs=self.config.train.num_train_epochs,
            predict_with_generate=True,
            save_steps=1000,
            eval_steps=1000,
            lr_scheduler_type=self.config.train.lr_scheduler_type,
            fp16=self.config.train.fp16,
            # fp16_opt_level="O1",  # mixed precision mode
            report_to="wandb",
            load_best_model_at_end=True,
            metric_for_best_model=self.config.train.metric_for_best_model,
        )
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["valid"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        torch.cuda.empty_cache()
        trainer.train()
        trainer.save_model()


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def load_pickle_data(path: str) -> Tuple[List[str], List[List[str]], List[str], List[str]]:
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
