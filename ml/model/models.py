import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BertModel, BertPreTrainedModel
from utils.dpr import DenseRetrieval


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
    def __init__(self, conf, args, q_encoder, p_encoder, passage_dataset, encoder_tokenizer):
        self.conf = conf
        self.args = args
        self.q_encoder = q_encoder
        self.p_encoder = p_encoder
        self.retriever = DenseRetrieval(
            args, passage_dataset, encoder_tokenizer, p_encoder, q_encoder, conf.emb_save_path, conf.dpr.do_eval
        )
        if conf.dpr.do_eval:
            self.retriever.set_passage_dataloader()
            self.retriever.create_passage_embeddings()

    # def forward(self, query, topk):


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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, label: torch.Tensor):
        """

        Args:
            input_ids (torch.Tensor): (topk, seq_max_len) 형태의 Tensor
            attention_mask (torch.Tensor): (topk, seq_max_len) 형태의 Tensor
            label (torch.Tensor): (1, seq_max_len) 형태의 summarization Tensor

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
        encoder_output = self.r_encoder(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]
        """encoder_output(dataclass)
            last_hidden_state : FloatTensor
                (batch_size, max_sequence_length, hidden_size) 형태의 마지막 layer의 output
            hidden_states : tuple(FloatTensor)
                embedding_layer + 모든 layer의 output을 (batch_size, sequence_length, hidden_size)형태로 제공
            attentions : tuple(FloatTensor)
                attention softmax 이후의 어텐션 가중치들.
                self-attention :head들의 가중치 평균 계산에 사용
        """
        decoder_input_ids = shift_tokens_right(
            label, self.encoder_tokenizer.pad_token_id, self.reader.config.decoder_start_token_id
        )
        # output last hidden state size애러 해결중
        decoder_output = self.r_decoder(
            input_ids=decoder_input_ids, encoder_hidden_states=encoder_output.view(1, -1, encoder_output.size(-1))
        )

        lm_logits = self.reader.lm_head(decoder_output["last_hidden_state"])
        lm_logits = lm_logits + self.reader.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if label is not None:
            loss_fn = CrossEntropyLoss()
            masked_lm_loss = loss_fn(lm_logits.view(-1, self.reader.config.vocab_size), label.view(-1))

        return {
            "logits": lm_logits,
            "loss_fn": masked_lm_loss,
            "last_hidden_state": decoder_output["last_hidden_state"],
        }


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
