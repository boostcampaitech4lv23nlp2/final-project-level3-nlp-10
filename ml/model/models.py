import torch
import torch.nn as nn
from transformers import (
    AutoModelForSeq2SeqLM,
    BartConfig,
    BartForConditionalGeneration,
    BertModel,
    BertPreTrainedModel,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Stack


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        return pooled_output


class FiDEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # passage_length = input_ids.size(-1)
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0].view(bsz, self.n_passages * passage_length, -1),
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

        return encoder_outputs


class FiD(BartForConditionalGeneration):
    def __init__(self, model_config: BartConfig):
        if model_config:
            super().__init__(model_config)
        else:
            super().__init__()
        self.wrap_encoder(self.model.encoder)

    def load_pretrained_params(self, basemodel_name):
        basemodel = AutoModelForSeq2SeqLM.from_pretrained(basemodel_name)
        self.model.encoder.encoder.load_state_dict(basemodel.get_encoder().state_dict())
        self.model.decoder.load_state_dict(basemodel.get_decoder().state_dict())
        self.lm_head.load_state_dict(basemodel.lm_head.state_dict())
        print(f"loaded {basemodel_name} parameters.")

    def wrap_encoder(self, encoder):
        self.model.encoder = FiDEncoder(encoder)

    @classmethod
    def from_path(cls, model_config):

        return cls(model_config=model_config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        """

        Args:
            input_ids (torch.Tensor): (batch_size, topk, seq_max_len) 형태의 Tensor
            attention_mask (torch.Tensor): (batch_size, topk, seq_max_len) 형태의 Tensor
            labels (torch.Tensor): (batch_size, seq_max_len) 형태의 summarization Tensor

        Return:
            {
                logit (torch.Tensor):
                    (batch_size, max_seq_len, vocab_size) 크기의 logit
                loss_fn:
                    logit과 label간의 loss_fn
                last_hidden_state (torch.Tensor):
                    (batch_size, max_seq_len, hidden_dim) 크기의 tensor
            }
        """
        if input_ids is not None:
            if input_ids.ndim == 3:
                self.model.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask is not None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def generate(self, input_ids, attention_mask, max_length=256):
        self.model.encoder.n_passages = input_ids.size(1)
        return super().generate(
            inputs=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
        )


class FiDT5Encoder(T5Stack):
    def __init__(self, encoder, model_config):
        super().__init__(model_config)
        self.encoder = encoder
        # 보조
        # self.main_input_name = encoder.main_input_name

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # passage_length = input_ids.size(-1)
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # encoder_outputs = (encoder_outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + encoder_outputs[1:]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=encoder_outputs[0].view(bsz, self.n_passages * passage_length, -1),
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class FiDT5(T5ForConditionalGeneration):
    def __init__(self, model_config: T5Config):
        if model_config:
            super().__init__(model_config)
        else:
            super().__init__()
        self.wrap_encoder(self.encoder, model_config)

    def load_pretrained_params(self, basemodel_name):
        basemodel = T5ForConditionalGeneration.from_pretrained(basemodel_name)
        self.encoder.encoder.load_state_dict(basemodel.get_encoder().state_dict())
        self.decoder.load_state_dict(basemodel.get_decoder().state_dict())
        self.lm_head.load_state_dict(basemodel.lm_head.state_dict())
        print(f"loaded {basemodel_name} parameters.")

    def wrap_encoder(self, encoder, model_config):
        self.encoder = FiDT5Encoder(encoder, model_config)

    @classmethod
    def from_path(cls, model_config):

        return cls(model_config=model_config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        """

        Args:
            input_ids (torch.Tensor): (batch_size, topk, seq_max_len) 형태의 Tensor
            attention_mask (torch.Tensor): (batch_size, topk, seq_max_len) 형태의 Tensor
            labels (torch.Tensor): (batch_size, seq_max_len) 형태의 summarization Tensor

        Return:
            {
                logit (torch.Tensor):
                    (batch_size, max_seq_len, vocab_size) 크기의 logit
                loss_fn:
                    logit과 label간의 loss_fn
                last_hidden_state (torch.Tensor):
                    (batch_size, max_seq_len, hidden_dim) 크기의 tensor
            }
        """
        if input_ids is not None:
            if input_ids.ndim == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask is not None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def generate(self, input_ids, attention_mask, max_length=256):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
        )


class FiDT5Mentor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.encoder = self.model.get_encoder()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # input_ids: (batch_size, n_passages, passage_length)
        batch_size, n_passages, passage_len = input_ids.shape

        # (bs, n_passages, passage_len) -> (bs * n_passages, passage_len)
        input_ids = input_ids.reshape(-1, passage_len)
        attention_mask_tmp = attention_mask.reshape(-1, passage_len)

        # encode the question + context
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask_tmp,
        )

        # concat all the encoder hidden states
        # (bs * n_passages, passage_len, hidden_dim) -> (bs, n_passages * passage_len, hidden_dim)
        hidden_states = encoder_outputs[0]
        encoder_outputs = (hidden_states.reshape(batch_size, n_passages * passage_len, -1), *encoder_outputs[1:])
        attention_mask = attention_mask.reshape(batch_size, -1)

        # Fusion-in-Decoder
        outputs = self.model(
            input_ids=None, attention_mask=attention_mask, encoder_outputs=encoder_outputs, labels=labels
        )
        return outputs

    def generate(self, input_ids, attention_mask, max_length):
        return self.model.generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
        )
