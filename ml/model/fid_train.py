import torch
import wandb
from data.make_dataset import ProjectDataset
from model.models import BertEncoder, FiD
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments
from utils.dpr import DenseRetrieval


def collate_fn(batch):
    return batch


def fid_train(conf):
    tokenizer = AutoTokenizer.from_pretrained(conf.fid.encoder_tokenizer)
    train_dataset = ProjectDataset(conf.common.dataset_path, "train", tokenizer.sep_token)
    valid_dataset = ProjectDataset(conf.common.dataset_path, "valid", tokenizer.sep_token)
    train_dataloader = DataLoader(
        train_dataset, batch_size=conf.common.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=conf.common.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True
    )
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
    reader_tokenizer = AutoTokenizer.from_pretrained(conf.fid.reader_model)

    if conf.dpr.emb_type == "train":
        emb_path = conf.dpr.emb_train_path
        emb_dataset = train_dataset
    elif conf.dpr.emb_type == "valid":
        emb_path = conf.dpr.emb_valid_path
        emb_dataset = valid_dataset
    elif conf.dpr.emb_type == "all":
        emb_path = conf.dpr.emb_all_path
        # TODO dataset for all
    else:
        raise Exception("Unsupported emb_type in config.yaml")
    retriever = DenseRetrieval(
        args=args,
        dataset=emb_dataset,
        num_neg=-1,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        emb_save_path=emb_path,
    )
    # retriever.set_passage_dataloader()
    retriever.create_passage_embeddings()
    quit()
    model = FiD.from_path(
        config=conf,
        args=args,
        reader_model_path=conf.fid.reader_model,
    )
    # Train
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=conf.common.learning_rate)
    for epoch in range(1, conf.common.num_train_epochs):
        total_loss = 0

        # Train
        model.to(args.device)
        model.train()
        for data in tqdm(train_dataloader, desc=f"train {epoch} epoch"):
            optimizer.zero_grad()
            questions = [batch["question"] for batch in data]
            answers = [batch["answer"] for batch in data]
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
            wandb.log({"train_loss": outputs["loss_fn"]})
            outputs["loss_fn"].backward()
            optimizer.step()
            total_loss += outputs["loss_fn"].detach().cpu()  # gpu에 있는 상태로 total_loss에 더하면 gpu 메모리에 누적됨
        mean_loss = total_loss / len(train_dataloader)
        wandb.log({"train_mean_loss": mean_loss})

        # Validation
        del input_ids
        del attention_mask
        del labels
        torch.cuda.empty_cache()
        valid_device = args.device
        valid_total_loss = 0
        model.to(valid_device)
        model.eval()
        with torch.no_grad():
            for data in tqdm(valid_dataloader, desc=f"valid {epoch} epoch"):
                questions = [batch["question"] for batch in data]
                answers = [batch["answer"] for batch in data]
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

                input_ids = torch.stack([item["input_ids"] for item in inputs], dim=0).to(valid_device)
                attention_mask = torch.stack([item["attention_mask"] for item in inputs], dim=0).to(valid_device)
                labels = torch.stack([item["input_ids"] for item in labels]).to(valid_device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                wandb.log({"valid_loss": outputs["loss_fn"]})
                valid_total_loss += outputs["loss_fn"].detach().cpu()
            valid_mean_loss = valid_total_loss / len(valid_dataloader)
            wandb.log({"valid_mean_loss": valid_mean_loss})

    save_path = "./saved_models/fid.pt"
    torch.save(model, save_path)
    print(f"saved at {save_path}")
