import torch
import wandb
from data.make_dataset import MRCDataset
from model.models import BertEncoder, FiD
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments
from utils.dpr import DenseRetrieval


def collate_fn(batch):
    return batch


def fid_train(conf):
    train_dataset = MRCDataset("train")
    valid_dataset = MRCDataset("validation")
    train_dataloader = DataLoader(train_dataset, batch_size=conf.common.batch_size, shuffle=True, collate_fn=collate_fn)
    train_dataloader
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=conf.common.batch_size, shuffle=False, collate_fn=collate_fn
    )
    valid_dataloader
    wandb.login()
    wandb.init(
        project=conf.wandb.project_name, entity=conf.wandb.entity_name, name=f"epoch{conf.common.num_train_epochs}"
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

    q_encoder = BertEncoder.from_pretrained(conf.fid.q_encoder_path)
    p_encoder = BertEncoder.from_pretrained(conf.fid.p_encoder_path)
    tokenizer = AutoTokenizer.from_pretrained(conf.fid.encoder_tokenizer)
    reader_tokenizer = AutoTokenizer.from_pretrained(conf.fid.reader_model)
    retriever = DenseRetrieval(
        args=args,
        dataset=train_dataset,
        num_neg=-1,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        emb_save_path=conf.fid.emb_save_path,
        eval=conf.dpr.do_eval,
    )
    retriever.set_passage_dataloader()
    retriever.create_passage_embeddings()
    model = FiD.from_path(
        config=conf, args=args, reader_model_path=conf.fid.reader_model, passage_dataset=valid_dataset
    )

    # Train
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=conf.common.learning_rate)
    for epoch in range(1, conf.common.num_train_epochs):
        total_loss = 0

        # Train
        for data in tqdm(train_dataloader, desc=f"train {epoch} epoch"):
            optimizer.zero_grad()
            questions = [batch["question"] for batch in data]
            answers = [batch["answers"][0] for batch in data]
            tokenized_queries = tokenizer(
                questions, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
            )
            top_doc_indices = retriever.get_relevant_doc(
                tokenized_queries, k=conf.fid.topk
            )  # (batch_size, topk)형태의 tensor
            top_docs = retriever.get_passage_by_indices(top_doc_indices)

            inputs = []
            labels = []
            for query, passages, answer in zip(questions, top_docs, answers):
                cur_input = [
                    f"{reader_tokenizer.cls_token}질문: {query} 문서: {p}{reader_tokenizer.eos_token}" for p in passages
                ]
                # BartEncoder의 embed_positions에서 input_ids.size를 처리하는데, 이때 최대 길이가 512
                inputs.append(
                    reader_tokenizer(
                        cur_input, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
                    )
                )
                labels.append(
                    reader_tokenizer(
                        answer + reader_tokenizer.eos_token,
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                )

            input_ids = torch.stack([item["input_ids"] for item in inputs], dim=0).to(args.device)
            attention_mask = torch.stack([item["attention_mask"] for item in inputs], dim=0).to(args.device)
            labels = torch.stack([item["input_ids"] for item in labels]).to(args.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs["loss_fn"]
            wandb.log({"train_loss": outputs["loss_fn"]})
            outputs["loss_fn"].backward()
            optimizer.step()
        mean_loss = total_loss / len(train_dataloader)
        wandb.log({"train_mean_loss": mean_loss})

        # Validation
        valid_total_loss = 0
        for data in tqdm(valid_dataloader, desc=f"valid {epoch} epoch"):
            questions = [batch["question"] for batch in data]
            answers = [batch["answers"][0] for batch in data]
            tokenized_queries = tokenizer(
                questions, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
            )
            top_doc_indices = retriever.get_relevant_doc(
                tokenized_queries, k=conf.fid.topk
            )  # (batch_size, topk)형태의 tensor
            top_docs = retriever.get_passage_by_indices(top_doc_indices)

            inputs = []
            labels = []
            for query, passages, answer in zip(questions, top_docs, answers):
                cur_input = [
                    f"{reader_tokenizer.cls_token}질문: {query} 문서: {p}{reader_tokenizer.eos_token}" for p in passages
                ]
                # BartEncoder의 embed_positions에서 input_ids.size를 처리하는데, 이때 최대 길이가 512
                inputs.append(
                    reader_tokenizer(
                        cur_input, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
                    )
                )
                labels.append(
                    reader_tokenizer(
                        answer + reader_tokenizer.eos_token,
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                )

            input_ids = torch.stack([item["input_ids"] for item in inputs], dim=0).to(args.device)
            attention_mask = torch.stack([item["attention_mask"] for item in inputs], dim=0).to(args.device)
            labels = torch.stack([item["input_ids"] for item in labels]).to(args.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            wandb.log({"valid_loss": outputs["loss_fn"]})
            valid_total_loss += outputs["loss_fn"]
        valid_mean_loss = valid_total_loss / len(valid_dataloader)
        wandb.log({"valid_mean_loss": valid_mean_loss})


def model_train(conf, model, dataloader):
    model.to()
