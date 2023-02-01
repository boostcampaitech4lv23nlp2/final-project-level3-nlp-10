import os

import wandb
from data.make_dataset import ProjectDataset
from model.models import BertEncoder

# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, TrainingArguments
from utils.dpr import DenseRetrieval

project_name = "Final_DPR_KLUE"
entity_name = "boost2end"


def collate_fn(batch):
    return batch


def train(conf, emb_type="train"):
    # train_dataset = load_dataset("klue", "mrc")["train"]
    reader_tokenizer = AutoTokenizer.from_pretrained(conf.fid.encoder_tokenizer)
    train_dataset = ProjectDataset(conf.common.dataset_path, "train", reader_tokenizer.sep_token)
    valid_dataset = ProjectDataset(conf.common.dataset_path, "valid", reader_tokenizer.sep_token)

    wandb.login()
    wandb.init(
        project=project_name, entity=entity_name, name=f"{conf.dpr.model_name}/epoch{conf.common.num_train_epochs}"
    )

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=conf.common.learning_rate,
        per_device_train_batch_size=conf.common.batch_size,
        per_device_eval_batch_size=conf.common.batch_size,
        num_train_epochs=conf.common.num_train_epochs,
        weight_decay=0.01,
    )
    tokenizer = AutoTokenizer.from_pretrained(conf.dpr.model_name)
    p_encoder = BertEncoder.from_pretrained(conf.dpr.model_name).to(args.device)
    q_encoder = BertEncoder.from_pretrained(conf.dpr.model_name).to(args.device)
    wandb.watch((p_encoder, q_encoder))

    if conf.dpr.emb_type == "train":
        emb_path = conf.dpr.emb_train_path
        emb_dataset = train_dataset
    elif conf.dpr.emb_type == "valid":
        emb_path = conf.dpr.emb_valid_path
        emb_dataset = valid_dataset
    elif conf.dpr.emb_type == "all":
        emb_path = conf.dpr.emb_all_path
        # TODO dataset for all

    retriever = DenseRetrieval(
        args=args,
        dataset=emb_dataset,
        num_neg=conf.dpr.train.neg_num,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        emb_save_path=emb_path,
    )
    retriever.set_passage_dataloader()
    retriever.create_passage_embeddings()

    retriever.train()

    # encoder
    encoder_save_path = f"model/saved_models/dpr/project/{conf.dpr.model_name.replace('/', '_')}/"
    p_encoder.save_pretrained(os.path.join(encoder_save_path, "p_encoder"))
    q_encoder.save_pretrained(os.path.join(encoder_save_path, "q_encoder"))
    print(f"saved at {encoder_save_path}")
