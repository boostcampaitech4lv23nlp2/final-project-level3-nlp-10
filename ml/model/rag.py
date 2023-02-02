import os

import datasets
import numpy as np
import torch
import wandb
from model.retrieval import DPRQuestionEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoTokenizer,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
    Trainer,
    get_cosine_schedule_with_warmup,
)

bleu_metric = datasets.load_metric("bleu")
rouge_metric = datasets.load_metric("rouge")


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class RagDataset(Dataset):
    def __init__(self, source_dataset, target_dataset, tokenizer, max_length, return_tensors="pt"):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.dataset_size = len(self.source_dataset)
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors
        self.max_length = max_length

    def __len__(self):
        return self.dataset_size

    def sample_data(self, query):
        # keyword sampling
        num = np.random.randint(2, len(query))
        query = np.array(query)
        np.random.shuffle(query)
        q_sample = "[SEP]".join(query[:num])
        return q_sample

    def __getitem__(self, idx):
        source_tokenizer = (
            self.tokenizer.question_encoder if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer
        )
        source_tokenizer.padding_side = "right"
        target_tokenizer = self.tokenizer.generator if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer
        target_tokenizer.padding_side = "right"
        source_inputs = source_tokenizer(
            self.sample_data(self.source_dataset[idx]),
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors=self.return_tensors,
        )
        target_inputs = target_tokenizer(
            self.target_dataset[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=self.return_tensors,
        )

        source_ids = source_inputs["input_ids"].squeeze().cuda()
        target_ids = target_inputs["input_ids"].squeeze().cuda()
        src_mask = source_inputs["attention_mask"].squeeze().cuda()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

    def collate_fn(self, batch):

        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
        batch = {
            "input_ids": input_ids,
            "attention_mask": masks,
            "decoder_input_ids": target_ids,
        }
        return batch


class RagTrainer(Trainer):
    def __init__(
        self,
        conf,
        q_encoder_model_name_or_path: str,
        generator_model_name_or_path: str,
        train_dataset=None,
        val_dataset=None,
        max_len=512,
        index_name: str = "custom",
        passages_path: str = "./test",
        index_path: str = "./data/test.faiss",
        use_dummy_dataset: bool = False,
        is_monitoring=False,
        save_period=1,
    ):
        self.conf = conf
        self.is_monitoring = is_monitoring
        self.max_len = max_len
        self.save_dir = self.conf.output_dir
        self.set_tokenizer(q_encoder_model_name_or_path, generator_model_name_or_path)
        self.model = RagSequenceForGeneration.from_pretrained_question_encoder_generator(
            q_encoder_model_name_or_path, generator_model_name_or_path
        )
        self.model.rag.question_encoder = DPRQuestionEncoder.from_pretrained(q_encoder_model_name_or_path)
        self.model.config.use_dummy_dataset = use_dummy_dataset
        self.model.config.index_name = index_name
        self.model.config.passages_path = passages_path
        self.model.config.index_path = index_path

        self.retriever = RagRetriever(self.model.config, self.question_encoder_tokenizer, self.generator_tokenizer)
        self.model.set_retriever(self.retriever)
        self.train_dataloader = self.prepare_data(
            train_dataset[0], train_dataset[1], batch_size=self.conf.per_device_train_batch_size
        )
        self.val_dataloader = None
        if val_dataset:
            self.val_dataloader = self.prepare_data(
                val_dataset[0], val_dataset[1], batch_size=self.conf.per_device_eval_batch_size
            )
        if self.is_monitoring:
            wandb.init(project="rag", entity="boost2end")
        self.set_optimizers(self.conf)
        self.save_period = save_period

    def set_optimizers(self, conf):
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": conf.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=conf.learning_rate, eps=conf.adam_epsilon)
        t_total = len(self.train_dataloader) // self.conf.gradient_accumulation_steps * self.conf.num_train_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.conf.warmup_steps, num_training_steps=t_total, num_cycles=1
        )

    def set_tokenizer(self, q_encoder_model_name_or_path: str, generator_model_name_or_path: str):
        self.question_encoder_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name_or_path)
        self.tokenizer = RagTokenizer(self.question_encoder_tokenizer, self.generator_tokenizer)

    def prepare_data(self, query_dataset, summary_dataset, tokenizer: RagTokenizer = None, batch_size=1):
        if tokenizer is None:
            tokenizer = self.tokenizer
        rag_dataset = RagDataset(
            source_dataset=query_dataset, target_dataset=summary_dataset, tokenizer=tokenizer, max_length=self.max_len
        )
        dataloader = DataLoader(rag_dataset, batch_size=batch_size)  # , collate_fn=rag_dataset.collate_fn)

        return dataloader

    def save(self, filename="temp_"):
        """save current model"""
        filename = f"rag_{filename}"
        filepath = os.path.join(self.save_dir, filename)
        self.model.save_pretrained(filepath)
        return filepath

    def train(self):
        self.model.cuda()
        self.model.train()
        self.model.zero_grad()
        torch.cuda.empty_cache()

        for epoch in range(int(self.conf.num_train_epochs)):
            with tqdm(self.train_dataloader, unit="batch", desc=f"{epoch}th Train: ") as tepoch:
                for batch in tepoch:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["decoder_input_ids"],
                    )
                    loss = torch.mean(outputs.loss)

                    loss.backward()
                    tepoch.set_postfix(loss=f"{str(loss.item())}")
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    torch.cuda.empty_cache()
                    if self.is_monitoring:
                        wandb.log({"epoch": epoch, "Train loss": loss, "lr": self.optimizer.param_groups[0]["lr"]})
            self.eval()
            if epoch % self.save_period == 0:
                filename = f"epoch_{epoch}"
                self.save(filename)

    def eval(self):
        with torch.no_grad():
            self.model.eval()
            self.model.zero_grad()
            bleus = []
            rouge1_p, rouge1_r, rouge1_f = [], [], []
            rouge2_p, rouge2_r, rouge2_f = [], [], []
            for i, batch in enumerate(self.val_dataloader):
                generated = self.model.generate(batch["input_ids"], max_length=256)
                bleu = bleu_metric.compute(predictions=generated, references=[batch["decoder_input_ids"]])
                rouge = rouge_metric.compute(predictions=generated, references=[batch["decoder_input_ids"]])

                bleus.append(bleu["bleu"])
                rouge1_p.append(rouge["rouge1"].mid.precision)
                rouge1_r.append(rouge["rouge1"].mid.recall)
                rouge1_f.append(rouge["rouge1"].mid.fmeasure)
                rouge2_p.append(rouge["rouge2"].mid.precision)
                rouge2_r.append(rouge["rouge2"].mid.recall)
                rouge2_f.append(rouge["rouge2"].mid.fmeasure)
                if i % 100 == 0:
                    print(f"{i}:")
                    print(generated)
                    print(f'{self.tokenizer.question_encoder.batch_decode(batch["input_ids"])}')
                    print(f"{self.tokenizer.generator.batch_decode(generated, skip_special_tokens=True)[0]}")

            wandb.log(
                {
                    "bleu": sum(bleus) / len(bleus),
                    "rouge1_p": sum(rouge1_p) / len(rouge1_p),
                    "rouge1_r": sum(rouge1_r) / len(rouge1_r),
                    "rouge1_f": sum(rouge1_f) / len(rouge1_f),
                    "rouge2_p": sum(rouge2_p) / len(rouge2_p),
                    "rouge2_r": sum(rouge2_r) / len(rouge2_r),
                    "rouge2_f": sum(rouge2_f) / len(rouge2_f),
                }
            )
