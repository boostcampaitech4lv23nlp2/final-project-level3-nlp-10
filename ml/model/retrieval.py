from typing import List, NoReturn, Optional, Tuple, Union

import os
import pickle
from dataclasses import dataclass

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import wandb

# from datasets import Dataset
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup  # , get_linear_schedule_with_warmup
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from transformers.models.dpr import DPRPretrainedContextEncoder, DPRPretrainedQuestionEncoder
from transformers.utils import ModelOutput
from utils import timer


class DPRConfig(PretrainedConfig):
    model_tyspe = "dpr"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        projection_dim: int = 0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.projection_dim = projection_dim
        self.position_embedding_type = position_embedding_type


class DPREncoder(BertPreTrainedModel):
    def __init__(self, config: DPRConfig):
        super(DPREncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[BaseModelOutputWithPooling, Tuple[Tensor, ...]]:

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # pooled_output = outputs[1]
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            return (sequence_output, pooled_output) + outputs[2:]
        # return pooled_output
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def embeddings_size(self) -> int:
        return self.bert.config.hidden_size


@dataclass
class DPRQuestionEncoderOutput(ModelOutput):
    """
    Class for outputs of [`DPRQuestionEncoder`].
    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, embeddings_size)`):
            The DPR encoder outputs the *pooler_output* that corresponds to the question representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed questions for nearest neighbors queries with context embeddings.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    pooler_output: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class DPRQuestionEncoder(DPRPretrainedQuestionEncoder):
    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.config = config
        self.question_encoder = DPREncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[DPRQuestionEncoderOutput, Tuple[Tensor, ...]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device) if input_ids is None else (input_ids != self.config.pad_token_id)
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        outputs = self.question_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs[1:]
        return DPRQuestionEncoderOutput(
            pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


@dataclass
class DPRContextEncoderOutput(ModelOutput):
    """
    Class for outputs of [`DPRQuestionEncoder`].
    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, embeddings_size)`):
            The DPR encoder outputs the *pooler_output* that corresponds to the context representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed contexts for nearest neighbors queries with questions embeddings.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    pooler_output: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class DPRContextEncoder(DPRPretrainedContextEncoder):
    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.config = config
        self.ctx_encoder = DPREncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[DPRContextEncoderOutput, Tuple[Tensor, ...]]:
        r"""
        Return:
        Examples:
        ```python
        >>> from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
        >>> tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        >>> model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
        >>> embeddings = model(input_ids).pooler_output
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device) if input_ids is None else (input_ids != self.config.pad_token_id)
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        outputs = self.ctx_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs[1:]
        return DPRContextEncoderOutput(
            pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class DenseRetrieval:
    def __init__(
        self,
        conf,
        num_neg,
        tokenizer,
        p_encoder,
        q_encoder,
        save_period: int = 1,
        eval_function=None,
        train_query_dataset=None,
        train_context_dataset=None,
        val_query_dataset=None,
        val_context_dataset=None,
        data_path="./data/ckpt",
        is_bm25=False,
        is_monitoring=False,
    ):
        self.conf = conf
        self.device = self.conf.device

        self.train_query_dataset = train_query_dataset
        self.train_context_dataset = train_context_dataset
        self.val_query_dataset = val_query_dataset
        self.val_context_dataset = val_context_dataset

        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.save_period = save_period
        self.data_path = data_path
        self.loss = torch.nn.NLLLoss()

        self.is_monitoring = is_monitoring
        if self.is_monitoring:
            wandb.init(project="dpr", entity="boost2end")
            wandb.watch((self.q_encoder, self.p_encoder), criterion=self.loss, log="all", log_freq=10)

        if self.train_query_dataset and self.train_context_dataset:  # for only train
            self.train_dataloader, self.train_passage_dataloader = self.prepare_in_batch_negative(
                query_dataset=self.train_query_dataset,
                context_dataset=self.train_context_dataset,
                is_bm25=is_bm25,
                is_train=True,
            )
        else:
            self.train_dataloader, self.train_passage_dataloader = None, None
        if self.val_query_dataset and self.val_context_dataset:  # for only test(or validation)
            self.val_dataloader, self.val_passage_dataloader = self.prepare_in_batch_negative(
                query_dataset=self.val_query_dataset,
                context_dataset=self.val_context_dataset,
                is_bm25=is_bm25,
                is_train=False,
            )
        else:
            self.val_dataloader, self.train_passage_dataloader = None, None
        self.set_optimziers(conf)
        # self.get_dense_embedding(reset=True)
        # self.indexer_path = self.build_faiss()

    def prepare_in_batch_negative(
        self, query_dataset, context_dataset, tokenizer=None, is_bm25: bool = False, is_train=False
    ):

        if tokenizer is None:
            tokenizer = self.tokenizer

        corpus = np.array(list(set([example for example in context_dataset])))
        p_with_neg = []

        for c in tqdm(context_dataset, desc="in_batch_negative"):
            while True:
                neg_idxs = np.random.randint(len(corpus), size=self.num_neg)
                if c not in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break
        q_seqs = tokenizer(query_dataset, padding="max_length", truncation=True, return_tensors="pt")
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors="pt")

        max_len = p_seqs["input_ids"].size(-1)

        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, self.num_neg + 1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, self.num_neg + 1, max_len)
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, self.num_neg + 1, max_len)

        _dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )

        batch_size = self.conf.per_device_train_batch_size if is_train else self.conf.per_device_eval_batch_size

        dataloader = DataLoader(_dataset, shuffle=is_train, batch_size=batch_size)

        valid_seqs = tokenizer(context_dataset, padding="max_length", truncation=True, return_tensors="pt")
        passage_dataset = TensorDataset(
            valid_seqs["input_ids"], valid_seqs["attention_mask"], valid_seqs["token_type_ids"]
        )

        passage_dataloader = DataLoader(passage_dataset, batch_size=batch_size)
        return dataloader, passage_dataloader

    def cache_passage_dense_vector(self, dataloader: DataLoader, encoder):
        embs = []
        for batch in dataloader:
            batch = tuple(t.to(self.conf.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}

            emb = encoder(**inputs).pooler_output.to("cpu")
            embs.append(emb)
            del inputs
        return embs

    def save(self, filename="temp_"):
        """save current model"""
        q_filename = f"q_encoder_{filename}"
        filepath = os.path.join(self.data_path, q_filename)
        # torch.save(self.q_encoder.state_dict(), filepath)
        self.q_encoder.save_pretrained(filepath)
        p_filename = f"p_encoder_{filename}"
        filepath = os.path.join(self.data_path, p_filename)
        self.p_encoder.save_pretrained(filepath)
        # torch.save(self.p_encoder.state_dict(), filepath)
        return filepath

    def set_optimziers(self, conf):
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": conf.weight_decay,
            },
            {
                "params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": conf.weight_decay,
            },
            {
                "params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=conf.learning_rate, eps=conf.adam_epsilon)
        t_total = len(self.train_dataloader) // self.conf.gradient_accumulation_steps * self.conf.num_train_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.conf.warmup_steps, num_training_steps=t_total
        )

    def reshape_batch_inputs(self, batch, is_train=True):
        batch_size = batch[0].shape[0]

        p_inputs = {
            "input_ids": batch[0].reshape(batch_size * (self.num_neg + 1), -1).to(self.device),
            "attention_mask": batch[1].reshape(batch_size * (self.num_neg + 1), -1).to(self.device),
            "token_type_ids": batch[2].reshape(batch_size * (self.num_neg + 1), -1).to(self.device),
        }

        q_inputs = {
            "input_ids": batch[3].to(self.device),
            "attention_mask": batch[4].to(self.device),
            "token_type_ids": batch[5].to(self.device),
        }
        return p_inputs, q_inputs

    def train(self):
        """
        Training Loop
        """

        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        for epoch in range(int(self.conf.num_train_epochs)):
            # if self.val_dataloader:
            # self.eval(epoch)
            with tqdm(self.train_dataloader, unit="batch", desc=f"{epoch}th Train: ") as tepoch:
                for batch in tepoch:

                    self.p_encoder.train()
                    self.q_encoder.train()

                    batch_size = batch[0].shape[0]
                    targets = torch.zeros(batch_size).long()
                    targets = targets.to(self.device)

                    p_inputs, q_inputs = self.reshape_batch_inputs(batch)

                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.pooler_output.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.pooler_output.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(
                        q_outputs, torch.transpose(p_outputs, 1, 2)
                    ).squeeze()  # (batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)
                    loss = self.loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    if self.is_monitoring:
                        wandb.log({"epoch": epoch, "Train loss": loss, "lr": self.optimizer.param_groups[0]["lr"]})
                    torch.cuda.empty_cache()
                    del p_inputs, q_inputs, batch

            if epoch % self.save_period == 0:
                filename = f"epoch_{epoch}"
                self.save(filename)

    def eval(self, epoch):
        """
        Evaluation Loop
        """

        self.p_encoder.eval()
        self.q_encoder.eval()

        torch.cuda.empty_cache()

        with tqdm(self.val_dataloader, unit="batch", desc=f"{epoch}th Eval: ") as tepoch:
            mean_loss = []
            for batch in tepoch:
                batch_size = batch[0].shape[0]
                targets = torch.zeros(batch_size).long()
                targets = targets.to(self.device)
                p_inputs, q_inputs = self.reshape_batch_inputs(batch, is_train=False)

                p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)
                # Calculate similarity score & loss
                p_outputs = p_outputs.pooler_output.view(batch_size, self.num_neg + 1, -1)
                q_outputs = q_outputs.pooler_output.view(batch_size, 1, -1)

                sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  # (batch_size, 1)
                sim_scores = sim_scores.view(batch_size, -1)
                sim_scores = F.log_softmax(sim_scores, dim=1)
                loss = self.loss(sim_scores, targets)
                mean_loss.append(loss.item())
                tepoch.set_postfix(loss=f"{str(loss.item())}")

                if self.is_monitoring:
                    wandb.log({"epoch": epoch, "Eval loss": loss})

                self.p_encoder.zero_grad()
                self.q_encoder.zero_grad()
                torch.cuda.empty_cache()
                del p_inputs, q_inputs
            if self.is_monitoring:
                wandb.log({"Eval loss mean": sum(mean_loss) / len(mean_loss)})

    def get_dense_embedding(
        self, p_encoder=None, passage_dataloader=None, pickle_name="dense_embedding.bin", reset=False
    ) -> NoReturn:

        """
        Summary:
            Generate Passage Embedding
            Save Embedding to pickle
        """
        if p_encoder is None:
            p_encoder = self.p_encoder

        if passage_dataloader is None:
            passage_dataloader = self.val_passage_dataloader

        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path) and reset is False:
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            # print("Embedding pickle load.")

        else:
            print("Build passage embedding")
            with torch.no_grad():
                p_encoder.eval()
                self.p_embedding = self.cache_passage_dense_vector(passage_dataloader, p_encoder)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")

    def build_faiss(self, num_clusters=64) -> NoReturn:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.
        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"dense_faiss_clusters{num_clusters}.faiss"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = np.array(self.p_embedding)
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(quantizer, quantizer.d, num_clusters, faiss.METRIC_L2)
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")
        return indexer_path

    def get_relevant_doc(
        self, queries: List, k: Optional[int] = 1, p_encoder=None, q_encoder=None, passage_dataloader=None, reset=False
    ) -> Tuple[List, List]:

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        if passage_dataloader is None:
            passage_dataloader = self.val_passage_dataloader

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors="pt").to(
                self.conf.device
            )  # [num_query, model_maxlen]
            q_emb = q_encoder(**q_seqs_val).pooler_output.to("cpu")  # [num_query, emb_dim]
            if reset is True:
                self.get_dense_embedding()
            # p_embs = self.cache_passage_dense_vector(passage_dataloader, p_encoder)

        p_embs = torch.stack(self.p_embedding, dim=0).view(
            len(passage_dataloader.dataset), -1
        )  # [num_passage, emb_dim]
        dot_prod_scores = torch.matmul(
            q_emb, torch.transpose(p_embs, 0, 1)
        )  # [num_query, emb_dim] @ [emb_dim, num_passage]
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()  # [num_query, num_passage]
        return rank[:k]


class SparseRetrieval:
    def __init__(self, tokenizer, dataset, data_path) -> NoReturn:

        self.contexts = dataset
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        self.tokenizer = tokenizer
        self.data_path = data_path

        self.tfidfv = TfidfVectorizer(
            tokenizer=self.tokenizer,
            ngram_range=(1, 2),
            max_features=50000,
        )
        self.get_sparse_embedding()  # generate by get_sparse_embedding()
        self.build_faiss()

    def get_sparse_embedding(self, pickle_name="sparse_embedding.bin", tfidfv_name="tfidv.bin") -> NoReturn:

        """
        Summary:
            Generate Passage Embedding
            Save TFIDF & Embedding to pickle
        """
        # emd_path = os.path.join(self.data_path, pickle_name)
        # tfidfv_path = os.path.join(self.data_path, tfidfv_name)
        self.p_embedding = self.tfidfv.fit_transform(self.contexts)
        # if os.path.isfile(emd_path):
        #     with open(emd_path, "rb") as file:
        #         self.p_embedding = pickle.load(file)
        #     print("Embedding pickle load.")
        # else:
        #     print("Build passage embedding")
        #     self.p_embedding = self.tfidfv.fit_transform(self.contexts)
        #     print(self.p_embedding.shape)
        #     with open(emd_path, "wb") as file:
        #         pickle.dump(self.p_embedding, file)
        #     print("Embedding pickle saved.")

    def build_faiss(self, num_clusters=64) -> NoReturn:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.
        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(quantizer, quantizer.d, num_clusters, faiss.METRIC_L2)
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def get_relevant_doc(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:

        query_vec = self.tfidfv.transform(queries)
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

    def get_relevant_doc_faiss(self, queries: [str], k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform(queries)
        assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]


class BM25(SparseRetrieval):
    def __init__(
        self,
        tokenizer,
        dataset,
        data_path: Optional[str] = "../data",
    ):
        super().__init__(tokenizer=tokenizer, dataset=dataset, data_path=data_path)

        self.contexts = dataset
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.get_sparse_embedding()

    def get_sparse_embedding(self):
        with timer("bm25 building"):
            self.p_embedding = BM25Okapi(self.contexts, tokenizer=self.tokenizer)

    def get_relevant_doc(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        with timer("transform"):
            tokenized_queries = [self.tokenizer(query) for query in queries]

        with timer("exhaustive search"):
            result = np.array(
                [self.p_embedding.get_scores(tokenized_query) for tokenized_query in tqdm(tokenized_queries)]
            )

        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices
