from typing import List

import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup


class DenseRetrieval:
    def __init__(
        self,
        tokenizer,
        p_encoder,
        q_encoder,
        emb_save_path="data/MRC_dataset/",
        args=None,
        dataset=None,
        num_neg=-1,
    ):
        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg
        self.emb_save_path = emb_save_path

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        # train인 경우
        if args and dataset:
            self.passages = dataset.context
            if num_neg > 0:
                self.prepare_in_batch_negative(num_neg=num_neg)
            print(f"Lengths of unique contexts : {len(self.passages)}")
            self.ids = list(range(len(self.passages)))
        # inference인 경우
        else:
            del self.p_encoder

    def create_passage_embeddings(self):
        pickle_name = "dense_embedding.bin"
        emb_path = os.path.join(self.emb_save_path, pickle_name)

        if os.path.isfile(emb_path):
            with open(emb_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            if self.passages is None:
                raise Exception("passage embedding 생성 시에는 클래스 생성 시 args, dataset이 필요합니다.")
            tokenized_embs = self.tokenizer(self.passages, padding="max_length", truncation=True, return_tensors="pt")
            tokenized_embs = tokenized_embs.to("cuda")
            self.p_encoder.to("cuda")

            self.p_embedding = torch.zeros(tokenized_embs["input_ids"].size(0), 768)  # Bert_encoder만을 위한 상수!
            for i in tqdm(
                range(0, len(tokenized_embs["input_ids"]), self.args.per_device_eval_batch_size),
                desc="buliding passage embeddings",
            ):
                end = (
                    i + self.args.per_device_eval_batch_size
                    if i + self.args.per_device_eval_batch_size < len(tokenized_embs["input_ids"])
                    else None
                )
                input_ids = tokenized_embs["input_ids"][i:end]
                attention_mask = tokenized_embs["attention_mask"][i:end]
                token_type_ids = tokenized_embs["token_type_ids"][i:end]
                self.p_embedding[i:end] = (
                    self.p_encoder(input_ids, attention_mask, token_type_ids).pooler_output.detach().cpu()
                )

            with open(emb_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print(f"embedding pickle saved at {emb_path}.")

    def get_passage_by_indices(self, top_docs):
        """
        Args:
            top_docs (torch.Tensor): (batch_size, topk) 형태의 tensor

        Returns:
            List[List[str]]: (batch_size, topk) 형태의
        """
        return [[self.passages[idx] for idx in indices] for indices in top_docs]

    def prepare_in_batch_negative(self, dataset=None, num_neg=2, tokenizer=None):
        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.
        corpus = np.array(list(set(self.passages)))
        p_with_neg = []
        print(f"corpus len {len(corpus)}")
        print(f"passages len {len(self.passages)}")
        cnt = 0
        pbar = tqdm(self.passages, desc="in-batch negative sampling")
        for c in pbar:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if c not in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break
                else:
                    cnt += 1
                    pbar.set_postfix_str(f"failed matching cnt : {cnt}")

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(dataset.question, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        p_seqs = tokenizer(p_with_neg, padding="max_length", max_length=512, truncation=True, return_tensors="pt")

        max_len = p_seqs["input_ids"].size(-1)
        self.max_len = max_len
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, num_neg + 1, max_len)
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, num_neg + 1, max_len)

        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )

        self.train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size
        )

        self.set_passage_dataloader()

    def set_passage_dataloader(self):
        valid_seqs = self.tokenizer(
            self.dataset.context, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        )
        passage_dataset = TensorDataset(
            valid_seqs["input_ids"], valid_seqs["attention_mask"], valid_seqs["token_type_ids"]
        )
        # self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size)
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=1)

    def train(self, args=None):
        if args is None:
            args = self.args

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                total_loss = 0
                for batch in tepoch:
                    self.p_encoder.train()
                    self.q_encoder.train()

                    p_inputs = {
                        "input_ids": batch[0].view(-1, self.max_len).to(args.device),
                        "attention_mask": batch[1].view(-1, self.max_len).to(args.device),
                        "token_type_ids": batch[2].view(-1, self.max_len).to(args.device),
                    }

                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device),
                    }

                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_batch_size = int(p_inputs["input_ids"].size()[0] / (self.num_neg + 1))
                    q_batch_size = q_inputs["input_ids"].size()[0]
                    p_outputs = p_outputs.view(p_batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(q_batch_size, 1, -1)
                    sim_scores = torch.bmm(
                        q_outputs, torch.transpose(p_outputs, 1, 2)
                    ).squeeze()  # (batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(q_batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    targets = torch.zeros(q_batch_size).long()  # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    total_loss += loss
                    wandb.log({"train/loss": loss})
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs
                wandb.log({"train/loss_mean": total_loss / len(tepoch)})

    def get_relevant_doc(self, tokenized_query, k=1):
        """
        args:
            tokenized_query : (batch_size, seq_len) 크기의 tokenized 된 query (BatchEncoding)
            k : topk
            args : arguments
            p_encoder : BertEncoder로 구성된 passage_encoder
            q_encoder : BertEncoder로 구성된 query_encoder

        return
            (batch_size, topk) 형태의 torch.Tensor

        """
        device = self.args.device if self.args else "cpu"
        self.q_encoder.to(device)
        self.q_encoder.eval()
        with torch.no_grad():
            tokenized_query = tokenized_query.to(device)
            q_emb = self.q_encoder(**tokenized_query).pooler_output.to("cpu")  # (num_query=1, emb_dim)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(self.p_embedding, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=-1, descending=True)  # rank: (batch_size, passage 전체 개수)
        if rank.size(0) > 1:
            rank = rank.squeeze()
        return rank[:, :k]

    def measure_performance(self, passages: List[str], questions: List[str], topk: int):
        """mesaure performance of retriever. passages should be the same passages that were used to build self.p_embedding

        Args:
            passages (List[str]): passage list
            questions (List[str]): question list
            topk (int): top-k개의 document를 뽑아낼 때 뽑아낼 document 개수
        """
        matched = 0
        for question, answer_passage in tqdm(
            zip(questions, passages), total=len(questions), desc="calculating retrieval performance"
        ):
            tokenized_question = self.tokenizer(
                question, max_length=512, truncation=True, padding="max_length", return_tensors="pt"
            )
            retrieved_doc_indices = self.get_relevant_doc(tokenized_question, topk)[0]
            retrieved_passages = [passages[idx] for idx in retrieved_doc_indices]
            if answer_passage in retrieved_passages:
                matched += 1
        percentage = matched / len(questions)
        print(f"successful retrieval percentage : {percentage}")
        return percentage
