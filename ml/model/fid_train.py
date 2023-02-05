import os

import torch
import wandb
from data.make_dataset import ProjectDataset
from datasets import load_metric
from model.models import FiD, FiDT5
from model.retriever import DPRContextEncoder, DPRQuestionEncoder
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, TrainingArguments
from utils.dpr import DenseRetrieval


def collate_fn(batch):
    return batch


def fid_train(conf, emb_type="train", t5=False):
    tokenizer = AutoTokenizer.from_pretrained(conf.fid.encoder_tokenizer)
    train_dataset = ProjectDataset(conf.common.dataset_path, "train", tokenizer.sep_token, conf.common.gzip)
    valid_dataset = ProjectDataset(conf.common.dataset_path, "valid", tokenizer.sep_token, conf.common.gzip)
    train_dataloader = DataLoader(
        train_dataset, batch_size=conf.common.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=conf.common.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True
    )
    wandb.login()
    wandb.init(project=conf.wandb.project_name, entity=conf.wandb.entity_name, name="mecab_newdpr_15")

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=conf.common.learning_rate,
        per_device_train_batch_size=conf.common.batch_size,
        per_device_eval_batch_size=conf.common.batch_size,
        num_train_epochs=conf.common.num_train_epochs,
        weight_decay=0.01,
    )

    q_encoder = DPRQuestionEncoder.from_pretrained("EJueon/keyword_dpr_question_encoder")
    p_encoder = DPRContextEncoder.from_pretrained("EJueon/keyword_dpr_context_encoder")

    reader_tokenizer = AutoTokenizer.from_pretrained(conf.fid.reader_model)

    if emb_type == "train":
        print("use emb train")
        emb_path = conf.dpr.emb_train_path
        emb_dataset = train_dataset
    elif emb_type == "valid":
        print("use emb valid")
        emb_path = conf.dpr.emb_valid_path
        emb_dataset = valid_dataset
    elif emb_type == "all":
        print("use emb all")
        emb_path = conf.dpr.emb_all_path
        emb_dataset = ProjectDataset(conf.common.dataset_path, emb_type, tokenizer.sep_token, conf.common.gzip)
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

    model = set_model(conf, t5)

    blue_metric = load_metric("bleu")
    # blue_metric = None
    rouge_metric = load_metric("rouge")

    os.makedirs(conf.fid.model_save_path, exist_ok=True)
    # Train
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=conf.common.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=0)
    predicted_list = []
    best_score = 0
    for epoch in range(1, conf.common.num_train_epochs + 1):
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
                    f"{reader_tokenizer.bos_token}질문: {query} 문서: {p}{reader_tokenizer.eos_token}" for p in passages
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
            labels = torch.stack([item["input_ids"] for item in labels]).squeeze(dim=1).to(args.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            wandb.log({"train/loss": outputs["loss"]})
            outputs["loss"].backward()
            optimizer.step()
            calculate_metrics(
                bleu_metric=blue_metric,
                rouge_metric=rouge_metric,
                tokenizer=reader_tokenizer,
                logits=outputs["logits"],
                labels=labels,
            )
            total_loss += outputs["loss"].detach().cpu()  # gpu에 있는 상태로 total_loss에 더하면 gpu 메모리에 누적됨

        mean_loss = total_loss / len(train_dataloader)
        wandb.log({"train/mean_loss": mean_loss})
        log_metrics(bleu_metric=blue_metric, rouge_metric=rouge_metric, run_type="train")

        # Validation
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
                        f"{reader_tokenizer.bos_token}질문: {query} 문서: {p}{reader_tokenizer.eos_token}" for p in passages
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
                labels = torch.stack([item["input_ids"] for item in labels]).squeeze(dim=1).to(valid_device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                wandb.log({"valid/loss": outputs["loss"]})
                calculate_metrics(
                    bleu_metric=blue_metric,
                    rouge_metric=rouge_metric,
                    tokenizer=reader_tokenizer,
                    logits=outputs["logits"],
                    labels=labels,
                )
                valid_total_loss += outputs["loss"].detach().cpu()
            valid_mean_loss = valid_total_loss / len(valid_dataloader)
            wandb.log({"valid/mean_loss": valid_mean_loss})
            bleu_score, rouge_score = log_metrics(bleu_metric=blue_metric, rouge_metric=rouge_metric, run_type="valid")
        cur_score = bleu_score * 0.5
        if rouge_score:
            cur_score += (rouge_score["rouge1"].mid.fmeasure + rouge_score["rouge2"].mid.fmeasure) * 0.5
        cur_score *= 100
        wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})
        scheduler.step(cur_score)
        if best_score < cur_score:
            best_score = cur_score
            model.save_pretrained(os.path.join(conf.fid.model_save_path, "pretrained_best"))
            print(f"new hight score with {cur_score}. save model at {conf.fid.model_save_path}")
        print_summarize_example(model, reader_tokenizer, input_ids, attention_mask, labels, predicted_list)

    model.save_pretrained(os.path.join(conf.fid.model_save_path, "pretrained_last"))
    print(f"saved at {conf.fid.model_save_path}")


def calculate_metrics(
    tokenizer,
    logits,
    labels,
    bleu_metric=None,
    rouge_metric=None,
):
    preds = torch.argmax(logits, dim=-1)
    preds = [tokenizer.decode(pred, skip_special_tokens=True).split() for pred in preds]
    labels = [[tokenizer.decode(label, skip_special_tokens=True).split()] for label in labels]
    if bleu_metric is not None:
        bleu_metric.add_batch(predictions=preds, references=labels)
    if rouge_metric is not None:
        rouge_metric.add_batch(predictions=preds, references=labels)


def log_metrics(bleu_metric=None, rouge_metric=None, run_type="train"):
    log = {}
    bleu_score = 0
    rouge_scores = None
    if bleu_metric is not None:
        bleu_score = bleu_metric.compute()["bleu"]
        log[f"{run_type}/train_bleu"] = bleu_score
    if rouge_metric is not None:
        rouge_scores = rouge_metric.compute()
        rouge1 = rouge_scores["rouge1"].mid
        rouge2 = rouge_scores["rouge2"].mid
        log.update(
            {
                f"{run_type}/train_rouge1_precision": rouge1.precision,
                f"{run_type}/train_rouge1_recall": rouge1.recall,
                f"{run_type}/train_rouge1_f-score": rouge1.fmeasure,
                f"{run_type}/train_rouge2_precision": rouge2.precision,
                f"{run_type}/train_rouge2_recall": rouge2.recall,
                f"{run_type}/train_rouge2_f-score": rouge2.fmeasure,
            }
        )
    wandb.log(log)
    print()
    print(rouge_scores)
    print()
    return bleu_score, rouge_scores


def print_summarize_example(model, reader_tokenizer, input_ids, attention_mask, labels, predicted_list):
    print()
    print("question")
    print(input_ids[0].size())
    print(reader_tokenizer.decode(input_ids[0][0], skip_special_tokens=True))  # decoder에는 1차원 텐서 입력
    print("predicted")
    # generate에는 (batch_size, topk, max_seq) 크기의 3차원 텐서 입력
    predicted_list.append(
        reader_tokenizer.decode(
            model.generate(input_ids[0].unsqueeze(0), attention_mask[0].unsqueeze(0), max_length=70)[0],
            skip_special_tokens=True,
        )
    )
    for pred in predicted_list:
        print(pred)
    print("answer")
    print(reader_tokenizer.decode(labels[0], skip_special_tokens=True))
    print()


def set_model(conf, t5):
    if t5:
        reader_model = conf.fid.t5_reader_model
        basemodel_config = AutoConfig.from_pretrained(reader_model)
        if conf.fid.continue_learning:
            model = FiDT5.from_pretrained(reader_model)
            print("\n" + "-" * 30)
            print("start learning from checkpoint")
            print("-" * 30 + "\n")
        else:
            model = FiDT5.from_path(model_config=basemodel_config)
            model.load_pretrained_params(reader_model)
    else:
        reader_model = conf.fid.reader_model
        if conf.fid.continue_learning:
            model = FiD.from_pretrained(conf.fid.checkpoint_path)
            print("\n" + "-" * 30)
            print("start learning from checkpoint")
            print("-" * 30 + "\n")
        else:
            basemodel_config = AutoConfig.from_pretrained(reader_model)
            model = FiD.from_path(model_config=basemodel_config)
            model.load_pretrained_params(conf.fid.reader_model)

    return model
