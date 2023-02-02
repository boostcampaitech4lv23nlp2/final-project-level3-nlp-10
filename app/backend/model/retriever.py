from typing import Optional, Tuple, Union

import os
import pickle
from dataclasses import dataclass

import numpy as np

# import faiss
import torch
import torch.nn.functional as F

# from datasets import Dataset
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from transformers.models.dpr import DPRPretrainedContextEncoder, DPRPretrainedQuestionEncoder

# from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
# from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import ModelOutput


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


class FiD_DenseRetrieval:
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

    def create_passage_embeddings(self, renew_emb=False):
        pickle_name = "dense_embedding.bin"
        os.makedirs(self.emb_save_path, exist_ok=True)
        emb_path = os.path.join(self.emb_save_path, pickle_name)

        if os.path.isfile(emb_path) and renew_emb is False:
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
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

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
