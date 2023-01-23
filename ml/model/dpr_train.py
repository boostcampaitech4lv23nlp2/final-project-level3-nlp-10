import random

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

# from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

model_name = "klue/bert-base"
seed = 12345
num_neg = 1
test_sample = 137
num_train_epochs = 1
learning_rate = 5e-5
batch_size = 4


class DenseRetrieval:
    def __init__(self, args, dataset, num_neg, tokenizer, p_encoder, q_encoder):

        """
        학습과 추론에 사용될 여러 셋업을 마쳐봅시다.
        """
        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        self.prepare_in_batch_negative(num_neg=num_neg)

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

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

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


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        pooled_output = outputs[1]
        return pooled_output


def train():
    train_dataset = load_dataset("klue", "mrc")["train"]
    if test_sample > 0:
        train_dataset = train_dataset[:test_sample]

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
    retriever = DenseRetrieval(
        args=args, dataset=train_dataset, num_neg=num_neg, tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder
    )
    retriever.train()

    p_encoder.save_pretrained("./p_encoder")
    q_encoder.save_pretrained("./q_encoder")


if __name__ == "__main__":
    train()
