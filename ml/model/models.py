import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BertModel, BertPreTrainedModel


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        pooled_output = outputs[1]
        return pooled_output


class FiD(nn.Module):
    def __init__(self, conf, args, reader, passage_dataset, encoder_tokenizer):
        super().__init__()
        self.conf = conf
        self.args = args
        self.r_encoder = reader.get_encoder()
        self.r_decoder = reader.get_decoder()
        self.reader = reader
        self.encoder_tokenizer = encoder_tokenizer
        self.datasets = passage_dataset

    @classmethod
    def from_path(cls, config, args, reader_model_path, passage_dataset):
        encoder_tokenizer = AutoTokenizer.from_pretrained(reader_model_path)
        reader = AutoModelForSeq2SeqLM.from_pretrained(reader_model_path)

        return cls(
            conf=config,
            args=args,
            reader=reader,
            passage_dataset=passage_dataset,
            encoder_tokenizer=encoder_tokenizer,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):
        """

        Args:
            input_ids (torch.Tensor): (batch_size, topk, seq_max_len) 형태의 Tensor
            attention_mask (torch.Tensor): (batch_size, topk, seq_max_len) 형태의 Tensor
            labels (torch.Tensor): (batch_size, 1, seq_max_len) 형태의 summarization Tensor

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

        passage_length = input_ids.size(-1)
        input_ids = input_ids.view(self.conf.common.batch_size * self.conf.fid.topk, -1)
        attention_mask = attention_mask.view(self.conf.common.batch_size * self.conf.fid.topk, -1)
        encoder_outputs = self.r_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # facebook github에서 모든 output들을 더했음
        encoder_outputs = (
            encoder_outputs[0].view(self.conf.common.batch_size, self.conf.fid.topk * passage_length, -1),
        ) + encoder_outputs[1:]
        # encoder_outptus : (batch_size, topk*seq_max_len, hidden_dim)
        encoder_outputs = encoder_outputs[0]
        # decoder_input_ids : (batch_size, topk, seq_max_len)
        decoder_input_ids = shift_tokens_right(
            labels, self.encoder_tokenizer.pad_token_id, self.reader.config.decoder_start_token_id
        )
        # decoder_output : (batch_size, seq_max_len, hidden_dim)
        decoder_output = self.r_decoder(
            input_ids=decoder_input_ids.view(-1, decoder_input_ids.size(-1)),
            encoder_hidden_states=encoder_outputs.view(self.conf.common.batch_size, -1, encoder_outputs.size(-1)),
        )["last_hidden_state"]
        lm_logits = self.reader.lm_head(decoder_output)

        # TODO facebook에서는 안하고 BartConditionalGenerate에서는 함
        # lm_logits: (batch_size, seq_max_len, vocab_size)
        lm_logits = lm_logits + self.reader.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            loss_fn = CrossEntropyLoss()
            masked_lm_loss = loss_fn(lm_logits.view(-1, self.reader.config.vocab_size), labels.view(-1))

        return {"logits": lm_logits, "loss_fn": masked_lm_loss, "last_hidden_state": decoder_output}


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
