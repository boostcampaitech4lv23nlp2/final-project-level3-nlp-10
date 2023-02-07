from typing import Callable, Iterable, List, Optional, Tuple

import os

import datasets
import numpy as np
import torch
import wandb
from model.retrieval import DPRQuestionEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoTokenizer,
    BartForConditionalGeneration,
    RagRetriever,
    RagTokenForGeneration,
    RagTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    get_cosine_schedule_with_warmup,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.rag.modeling_rag import RetrievAugLMMarginOutput

rouge_metric = datasets.load_metric("rouge")


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


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

    def map_decoder_inputs(self, context):
        return f"[BOS]{context}[EOS]"

    def __getitem__(self, idx):
        source_tokenizer = (
            self.tokenizer.question_encoder if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer
        )
        source_tokenizer.padding_side = "right"
        target_tokenizer = self.tokenizer.generator if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer
        target_tokenizer.padding_side = "right"
        source_inputs = source_tokenizer(
            self.sample_data(self.source_dataset[idx]),
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors=self.return_tensors,
        )

        target_inputs = target_tokenizer(
            self.map_decoder_inputs(self.target_dataset[idx]),
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


class Rag(RagTokenForGeneration):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,
        **kwargs,
    ):
        super().__init__(config, question_encoder, generator, retriever, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.LongTensor] = None,
        doc_scores: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_retrieved: Optional[bool] = None,
        do_marginalize: Optional[bool] = None,
        reduce_loss: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        n_docs: Optional[int] = None,
        **kwargs,  # needs kwargs for generation
    ) -> RetrievAugLMMarginOutput:
        r"""
        do_marginalize (`bool`, *optional*):
            If `True`, the logits are marginalized over all documents by making use of
            `torch.nn.functional.log_softmax`.
        reduce_loss (`bool`, *optional*):
            Only relevant if `labels` is passed. If `True`, the NLL loss is reduced using the `torch.Tensor.sum`
            operation.
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Legacy dictionary, which is required so that model can use *generate()* function.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, RagRetriever, RagTokenForGeneration
        >>> import torch
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
        >>> retriever = RagRetriever.from_pretrained(
        ...     "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
        ... )
        >>> # initialize with RagRetriever to do everything in one forward call
        >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
        >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
        >>> targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
        >>> input_ids = inputs["input_ids"]
        >>> labels = targets["input_ids"]
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> # or use retriever separately
        >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)
        >>> # 1. Encode
        >>> question_hidden_states = model.question_encoder(input_ids)[0]
        >>> # 2. Retrieve
        >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
        >>> doc_scores = torch.bmm(
        ...     question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
        ... ).squeeze(1)
        >>> # 3. Forward to generator
        >>> outputs = model(
        ...     context_input_ids=docs_dict["context_input_ids"],
        ...     context_attention_mask=docs_dict["context_attention_mask"],
        ...     doc_scores=doc_scores,
        ...     decoder_input_ids=labels,
        ... )
        >>> # or directly generate
        >>> generated = model.generate(
        ...     context_input_ids=docs_dict["context_input_ids"],
        ...     context_attention_mask=docs_dict["context_attention_mask"],
        ...     doc_scores=doc_scores,
        ... )
        >>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
        ```"""
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_marginalize = do_marginalize if do_marginalize is not None else self.config.do_marginalize
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False

        outputs = self.rag(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            n_docs=n_docs,
        )

        loss = None
        logits = outputs.logits

        if labels is not None:
            assert decoder_input_ids is not None
            loss = self.get_nll(
                outputs.logits,
                outputs.doc_scores,
                labels,
                reduce_loss=reduce_loss,
                epsilon=self.config.label_smoothing,
                n_docs=n_docs,
            )
        if do_marginalize:
            logits = self.marginalize(logits, outputs.doc_scores, n_docs)

        return RetrievAugLMMarginOutput(
            loss=loss,
            logits=logits,
            doc_scores=outputs.doc_scores,
            past_key_values=outputs.past_key_values,
            context_input_ids=outputs.context_input_ids,
            context_attention_mask=outputs.context_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
            generator_cross_attentions=outputs.generator_cross_attentions,
        )

    def marginalize(self, seq_logits, doc_scores, n_docs=None):

        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # RAG-token marginalization
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )

        doc_logprobs = torch.log_softmax(doc_scores, dim=1)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        return torch.logsumexp(log_prob_sum, dim=1)

    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        # shift tokens left
        target = torch.cat([target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1)

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        rag_logprobs = self.marginalize(seq_logits, doc_scores, n_docs)
        target = target.unsqueeze(-1)
        assert target.dim() == rag_logprobs.dim()

        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits
        ll, smooth_obj = _mask_pads(ll, smooth_obj)
        ll = ll.sum(1)  # sum over tokens
        smooth_obj = smooth_obj.sum(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss


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
        n_docs=1,
    ):
        self.conf = conf
        self.is_monitoring = is_monitoring
        self.max_len = max_len
        self.save_dir = self.conf.output_dir
        self.set_tokenizer(q_encoder_model_name_or_path, generator_model_name_or_path)
        self.model = Rag.from_pretrained_question_encoder_generator(
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
        self.n_docs = n_docs

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
        self.generator_tokenizer = AutoTokenizer.from_pretrained("alaggung/bart-r3f")
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
        for epoch in range(int(self.conf.num_train_epochs)):
            self.eval()
            self.model.train()
            self.model.zero_grad()
            torch.cuda.empty_cache()
            with tqdm(self.train_dataloader, unit="batch", desc=f"{epoch}th Train: ") as tepoch:
                for batch in tepoch:
                    generator = self.model.rag.generator
                    target_ids = batch["decoder_input_ids"]
                    if isinstance(self.model.generator, T5ForConditionalGeneration):
                        decoder_start_token_id = generator.config.decoder_start_token_id
                        decoder_input_ids = (
                            torch.cat(
                                [
                                    torch.tensor([[decoder_start_token_id]] * target_ids.shape[0]).to(target_ids),
                                    target_ids,
                                ],
                                dim=1,
                            )
                            if target_ids.shape[0] < self.target_lens["train"]
                            else generator._shift_right(target_ids)
                        )
                    elif isinstance(generator, BartForConditionalGeneration):
                        decoder_input_ids = target_ids
                    lm_labels = decoder_input_ids
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        decoder_input_ids=batch["decoder_input_ids"],
                        labels=lm_labels,
                        n_docs=self.n_docs,
                        reduce_loss=True,
                        use_cache=False,
                        do_marginalize=True,
                    )
                    loss = outputs.loss
                    loss.backward()
                    tepoch.set_postfix(loss=f"{str(loss.item())}")
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()

                    if self.is_monitoring:
                        wandb.log({"epoch": epoch, "Train loss": loss, "lr": self.optimizer.param_groups[0]["lr"]})
                    torch.cuda.empty_cache()
                    del batch
            if epoch % self.save_period == 0:
                filename = f"epoch_{epoch}"
                self.save(filename)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def generate(self, inputs):
        source_tokenizer = (
            self.tokenizer.question_encoder if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer
        )
        source_inputs = source_tokenizer.prepare_seq2seq_batch(
            inputs,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.model.cuda()
        with torch.no_grad():
            generated_ids = self.model.generate(
                source_inputs["input_ids"].cuda(),
                attention_mask=source_inputs["attention_mask"].cuda(),
                do_deduplication=False,  # rag specific parameter
                use_cache=True,
                min_length=1,
                max_length=512,
            )
            preds: List[str] = self.ids_to_clean_text(generated_ids=generated_ids)
        return preds

    def eval(self):
        self.model.cuda()
        with torch.no_grad():
            self.model.eval()
            self.model.zero_grad()
            rouge1_p, rouge1_r, rouge1_f = [], [], []
            rouge2_p, rouge2_r, rouge2_f = [], [], []
            rougeL_p, rougeL_r, rougeL_f = [], [], []
            for i, batch in enumerate(tqdm(self.val_dataloader)):
                generated_ids = self.model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    do_deduplication=False,  # rag specific parameter
                    use_cache=True,
                    min_length=1,
                    max_length=512,
                )
                preds: List[str] = self.ids_to_clean_text(generated_ids=generated_ids)
                target: List[str] = self.ids_to_clean_text(batch["decoder_input_ids"])

                # generated_tokens = [
                #     self.tokenizer.generator.convert_ids_to_tokens(g, skip_special_tokens=True) for g in generated_ids
                # ]
                rouge = rouge_metric.compute(predictions=[preds], references=[[target]])

                rouge1_p.append(rouge["rouge1"].mid.precision)
                rouge1_r.append(rouge["rouge1"].mid.recall)
                rouge1_f.append(rouge["rouge1"].mid.fmeasure)
                rouge2_p.append(rouge["rouge2"].mid.precision)
                rouge2_r.append(rouge["rouge2"].mid.recall)
                rouge2_f.append(rouge["rouge2"].mid.fmeasure)
                rougeL_p.append(rouge["rougeL"].mid.precision)
                rougeL_r.append(rouge["rougeL"].mid.recall)
                rougeL_f.append(rouge["rougeL"].mid.fmeasure)
                if i % 100 == 0:
                    print(f"{i}:")
                    keywords: List[str] = self.ids_to_clean_text(batch["input_ids"])
                    print(f"keywords: {keywords[0]}")
                    print(f"target: {target[0]}")
                    print(f"prediction : {preds[0]}")
                del batch
            if self.is_monitoring:
                wandb.log(
                    {
                        "rouge1_p": sum(rouge1_p) / len(rouge1_p),
                        "rouge1_r": sum(rouge1_r) / len(rouge1_r),
                        "rouge1_f": sum(rouge1_f) / len(rouge1_f),
                        "rouge2_p": sum(rouge2_p) / len(rouge2_p),
                        "rouge2_r": sum(rouge2_r) / len(rouge2_r),
                        "rouge2_f": sum(rouge2_f) / len(rouge2_f),
                        "rougeL_f": sum(rougeL_f) / len(rougeL_f),
                    }
                )
