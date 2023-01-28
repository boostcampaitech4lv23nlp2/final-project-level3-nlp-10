import sys
from pprint import pprint

import wandb
from data.make_dataset import MRCDataset
from model.models import BertEncoder

# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, TrainingArguments
from utils.dpr import DenseRetrieval

project_name = "Final_DPR_KLUE"
entity_name = "boost2end"


def train(conf, model_name="klue/bert-base"):
    # train_dataset = load_dataset("klue", "mrc")["train"]
    train_dataset = MRCDataset("train")
    test_sample = conf.dpr.train.test_sample
    if test_sample > 0:
        train_dataset = train_dataset[:test_sample]

    wandb.login()
    wandb.init(project=project_name, entity=entity_name, name=f"{model_name}/epoch{conf.dpr.train.num_train_epochs}")

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=conf.dpr.train.learning_rate,
        per_device_train_batch_size=conf.common.batch_size,
        per_device_eval_batch_size=conf.common.batch_size,
        num_train_epochs=conf.dpr.train.num_train_epochs,
        weight_decay=0.01,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    p_encoder = BertEncoder.from_pretrained(model_name).to(args.device)
    q_encoder = BertEncoder.from_pretrained(model_name).to(args.device)
    wandb.watch((p_encoder, q_encoder))
    retriever = DenseRetrieval(
        args=args,
        dataset=train_dataset,
        num_neg=conf.dpr.train.neg_num,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
    )
    retriever.train()

    # encoder 저장
    p_encoder.save_pretrained(f"./saved_models/{model_name.replace('/', '_')}/p_encoder")
    q_encoder.save_pretrained(f"./saved_models/{model_name.replace('/', '_')}/q_encoder")

    # 테스트
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
