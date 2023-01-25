import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, BertModel, BertPreTrainedModel


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        pooled_output = outputs[1]
        return pooled_output


class DPR(nn.Module):
    def __init__(self, config, q_encoder, p_encoder):
        self.conf = config
        self.q_encoder = q_encoder
        self.p_encoder = p_encoder


class FiD(nn.Module):
    def __init__(self, config, q_encoder, p_encoder, reader):
        super().__init__()

        self.conf = config
        self.q_encoder = q_encoder
        self.p_encoder = p_encoder
        self.reader = reader

    @classmethod
    def from_path(cls, config, q_encoder_path, p_encoder_path, reader_model_path):
        q_encoder = BertEncoder.from_pretrained(q_encoder_path)
        p_encoder = BertEncoder.from_pretrained(p_encoder_path)
        reader = AutoModelForSeq2SeqLM.from_pretrained(reader_model_path)

        return cls(config, q_encoder, p_encoder, reader)
