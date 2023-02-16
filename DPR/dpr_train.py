import random

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm, trange
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
num_neg = 3


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        pooled_output = outputs[1]

        return pooled_output


def train(args, dataset, p_model, q_model):

    # Dataloader
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        {
            "params": [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in q_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Start training!
    global_step = 0

    p_model.zero_grad()
    q_model.zero_grad()
    torch.cuda.empty_cache()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            p_model.train()
            q_model.train()

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            p_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}

            q_inputs = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}

            p_outputs = p_model(**p_inputs)  # (batch_size, emb_dim)
            q_outputs = q_model(**q_inputs)  # (batch_size, emb_dim)

            # Calculate similarity score & loss
            sim_scores = torch.matmul(
                q_outputs, torch.transpose(p_outputs, 0, 1)
            )  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)

            # target: position of positive samples = diagonal element
            targets = torch.arange(0, args.per_device_train_batch_size).long()
            if torch.cuda.is_available():
                targets = targets.to("cuda")

            sim_scores = F.log_softmax(sim_scores, dim=1)

            loss = F.nll_loss(sim_scores, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()
            q_model.zero_grad()
            p_model.zero_grad()
            global_step += 1

            torch.cuda.empty_cache()

    return p_model, q_model


def to_cuda(batch):
    return tuple(t.cuda() for t in batch)


def main():
    dataset = load_dataset("klue", "mrc")
    # corpus = list(set([example["context"] for example in dataset["train"]]))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # sample_idx = np.random.choice(range(len(dataset['train'])), 128)
    training_dataset = dataset["train"][:]

    q_seqs = tokenizer(training_dataset["question"], padding="max_length", truncation=True, return_tensors="pt")
    p_seqs = tokenizer(training_dataset["context"], padding="max_length", truncation=True, return_tensors="pt")

    train_dataset = TensorDataset(
        p_seqs["input_ids"],
        p_seqs["attention_mask"],
        p_seqs["token_type_ids"],
        q_seqs["input_ids"],
        q_seqs["attention_mask"],
        q_seqs["token_type_ids"],
    )

    p_encoder = BertEncoder.from_pretrained(model_name)
    q_encoder = BertEncoder.from_pretrained(model_name)

    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
    )
    p_encoder, q_encoder = train(args, train_dataset, p_encoder, q_encoder)

    p_encoder.save_pretrained("./p_encoder")
    q_encoder.save_pretrained("./q_encoder")

    """
    with torch.no_grad():
        p_encoder.eval()
        q_encoder.eval()

    q_seqs_val = tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
    q_emb = q_encoder(**q_seqs_val).to('cpu')  #(num_query, emb_dim)

    p_embs = []
    for p in valid_corpus:
        p = tokenizer(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
        p_emb = p_encoder(**p).to('cpu').numpy()
        p_embs.append(p_emb)

    p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)
    """


if __name__ == "__main__":
    main()
