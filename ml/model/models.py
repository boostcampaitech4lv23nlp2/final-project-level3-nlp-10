import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, BartConfig, BartForConditionalGeneration, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput


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
    def __init__(self, encoder, conf):
        super().__init__()

        self.encoder = encoder
        self.conf = conf

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # passage_length = input_ids.size(-1)
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.conf.fid.topk
        input_ids = input_ids.view(bsz * self.conf.fid.topk, passage_length)
        attention_mask = attention_mask.view(bsz * self.conf.fid.topk, passage_length)
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # facebook github에서 모든 output들을 더했음
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0].view(bsz, self.conf.fid.topk * passage_length, -1),
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

        return encoder_outputs


class FiD(BartForConditionalGeneration):
    def __init__(self, model_config: BartConfig, conf):
        super().__init__(model_config)  # self.model: BartModel, self.lm_head 생성
        self.wrap_encoder(self.model.encoder, conf)
        self.conf = conf

    def load_pretrained_params(self, basemodel_name):
        basemodel = AutoModelForSeq2SeqLM.from_pretrained(basemodel_name)
        self.model.encoder.encoder.load_state_dict(basemodel.get_encoder().state_dict())
        self.model.decoder.load_state_dict(basemodel.get_decoder().state_dict())
        self.lm_head.load_state_dict(basemodel.lm_head.state_dict())
        print(f"loaded {basemodel_name} parameters.")

    def wrap_encoder(self, encoder, conf):
        self.model.encoder = FiDEncoder(encoder, conf)

    @classmethod
    def from_path(cls, model_config, config):

        return cls(
            model_config=model_config,
            conf=config,
        )

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
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask is not None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def generate(self, input_ids, attention_mask, max_length=256):
        return super().generate(
            inputs=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
        )


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, :, 1:] = input_ids[:, :, :-1].clone()
    shifted_input_ids[:, :, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
