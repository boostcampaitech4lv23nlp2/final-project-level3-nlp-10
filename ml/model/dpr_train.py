import random
import sys
from pprint import pprint

import numpy as np
import torch
import wandb
from datasets import load_dataset
from model.models import BertEncoder, DenseRetrieval

# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, TrainingArguments

# model_name = "klue/bert-base"
seed = 12345
num_neg = 7
test_sample = -1
num_train_epochs = 5
learning_rate = 5e-5
batch_size = 4

project_name = "Final_DPR_KLUE"
entity_name = "boost2end"


def train(model_name="klue/bert-base"):
    train_dataset = load_dataset("klue", "mrc")["train"]
    if test_sample > 0:
        train_dataset = train_dataset[:test_sample]

    wandb.login()
    wandb.init(project=project_name, entity=entity_name, name=f"{model_name}/epoch{num_train_epochs}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    p_encoder = BertEncoder.from_pretrained(model_name).to(args.device)
    q_encoder = BertEncoder.from_pretrained(model_name).to(args.device)
    wandb.watch((p_encoder, q_encoder))
    retriever = DenseRetrieval(
        args=args, dataset=train_dataset, num_neg=num_neg, tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder
    )
    retriever.train()

    p_encoder.save_pretrained(f"./saved_models/{model_name.replace('/', '_')}/p_encoder")
    q_encoder.save_pretrained(f"./saved_models/{model_name.replace('/', '_')}/q_encoder")

    query = "제주도 시청의 주소는 뭐야?"
    results = retriever.get_relevant_doc(query=query, k=5)
    print(f"[Search Query] {query}\n")

    indices = results.tolist()
    for i, idx in enumerate(indices):
        print(f"Top-{i + 1}th Passage (Index {idx})")
        pprint(retriever.dataset["context"][idx])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        train(model_name)
    else:
        train()
