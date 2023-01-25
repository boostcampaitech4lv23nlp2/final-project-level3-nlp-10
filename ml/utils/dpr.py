from typing import List, Optional, Tuple

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
        args,
        dataset,
        num_neg,
        tokenizer,
        p_encoder,
        q_encoder,
        data_path="data/MRC_dataset/",
        passage_path="data/MRC_dataset/wikipedia_documents.json",
        eval=False,
    ):

        """
        현재 MRC_dataset에 맞춰져 있음(특히 passage들을 읽는 부분)
        """
        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg
        self.data_path = data_path

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        self.passages = dataset["context"]

        print(f"Lengths of unique contexts : {len(self.passages)}")
        self.ids = list(range(len(self.passages)))

        if not eval:
            self.prepare_in_batch_negative(num_neg=num_neg)

    def create_passage_embeddings(self):
        pickle_name = "dense_embedding.bin"
        emb_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emb_path):
            with open(emb_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            tokenized_embs = self.tokenizer(self.passages, padding="max_length", truncation=True, return_tensors="pt")
            tokenized_embs = tokenized_embs.to("cuda")
            self.p_encoder.to("cuda")

            self.p_embedding = torch.zeros(tokenized_embs["input_ids"].size(0), 768)  # Bert_encoder만을 위한 상수!
            for i in tqdm(range(0, len(tokenized_embs["input_ids"]), self.args.per_device_eval_batch_size)):
                end = (
                    i + self.args.per_device_eval_batch_size
                    if i + self.args.per_device_eval_batch_size < len(tokenized_embs["input_ids"])
                    else None
                )
                input_ids = tokenized_embs["input_ids"][i:end]
                attention_mask = tokenized_embs["attention_mask"][i:end]
                token_type_ids = tokenized_embs["token_type_ids"][i:end]
                self.p_embedding[i:end] = self.p_encoder(input_ids, attention_mask, token_type_ids).detach().cpu()

            with open(emb_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("embedding pickle saved.")

    def prepare_in_batch_negative(self, dataset=None, num_neg=2, tokenizer=None):
        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.
        corpus = np.array(list(set([example for example in dataset["context"]])))
        p_with_neg = []

        cnt = 0
        pbar = tqdm(dataset["context"], desc="in-batch negative sampling")
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
        print(f"total {len(p_with_neg)} passages for p_seq")
        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(dataset["question"], padding="max_length", truncation=True, return_tensors="pt")
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors="pt")

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

        valid_seqs = tokenizer(dataset["context"], padding="max_length", truncation=True, return_tensors="pt")
        passage_dataset = TensorDataset(
            valid_seqs["input_ids"], valid_seqs["attention_mask"], valid_seqs["token_type_ids"]
        )
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size)

    def set_passage_dataloader(self):
        valid_seqs = self.tokenizer(self.dataset["context"], padding="max_length", truncation=True, return_tensors="pt")
        passage_dataset = TensorDataset(
            valid_seqs["input_ids"], valid_seqs["attention_mask"], valid_seqs["token_type_ids"]
        )
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size)

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

    def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None):

        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors="pt").to(
                args.device
            )
            q_emb = q_encoder(**q_seqs_val).to("cpu")  # (num_query=1, emb_dim)

            p_embs = []
            for batch in self.passage_dataloader:

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
                p_emb = p_encoder(**p_inputs).to("cpu")
                p_embs.append(p_emb)

        p_embs = torch.stack(p_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)  # (num_passage, emb_dim)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        return rank[:k]

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform(queries)
        assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices
