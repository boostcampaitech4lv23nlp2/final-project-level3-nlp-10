import torch
import torch.nn as nn
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
    def __init__(self, conf, args, q_encoder, p_encoder, reader, passage_dataset, dpr_tokenizer, encoder_tokenizer):
        super().__init__()

        self.conf = conf
        self.args = args
        self.q_encoder = q_encoder
        self.p_encoder = p_encoder
        self.r_encoder = reader.get_encoder()
        self.r_decoder = reader.get_decoder()
        self.reader = reader
        self.r_embedding = list(reader.base_model.children())[0]
        self.encoder_tokenizer = encoder_tokenizer
        self.datasets = passage_dataset
        self.retriever = DenseRetrieval(
            args=args,
            dataset=passage_dataset,
            num_neg=-1,
            tokenizer=dpr_tokenizer,
            p_encoder=p_encoder,
            q_encoder=q_encoder,
            emb_save_path=conf.fid.emb_save_path,
            eval=conf.dpr.do_eval,
        )
        self.retriever.set_passage_dataloader()
        self.retriever.create_passage_embeddings()
        """
        if conf.dpr.do_eval:
            self.dpr = DPR(
                config,
                args,
                q_encoder,
                p_encoder,
                passage_dataset,
                encoder_tokenizer
            )
        """

    @classmethod
    def from_path(cls, config, args, q_encoder_path, p_encoder_path, reader_model_path, passage_dataset):
        q_encoder = BertEncoder.from_pretrained(q_encoder_path)
        p_encoder = BertEncoder.from_pretrained(p_encoder_path)
        dpr_tokenizer = AutoTokenizer.from_pretrained(config.fid.encoder_tokenizer)
        encoder_tokenizer = AutoTokenizer.from_pretrained(reader_model_path)
        reader = AutoModelForSeq2SeqLM.from_pretrained(reader_model_path)

        return cls(
            conf=config,
            args=args,
            q_encoder=q_encoder,
            p_encoder=p_encoder,
            reader=reader,
            passage_dataset=passage_dataset,
            dpr_tokenizer=dpr_tokenizer,
            encoder_tokenizer=encoder_tokenizer,
        )

    def forward(self, query, labels=None):
        doc_top_indices = self.retriever.get_relevant_doc(query, k=self.conf.fid.topk)
        inputs = []
        for idx in doc_top_indices:
            inputs.append(f"질문:{query} 문서:{self.datasets['context'][idx]}")

        tokenized_inputs = torch.zeros((self.conf.fid.topk, 1, 1024), dtype=torch.int8)
        concat_input = torch.zeros(0)
        for i, input in enumerate(inputs):
            # TODO encoder의 max_length 상수로 사용함
            tokenized_input = self.encoder_tokenizer(
                input, padding="max_length", max_length=1024, truncation=True, return_tensors="pt"
            )
            tokenized_inputs[i] = tokenized_input["input_ids"].clone().type(torch.int8)
            tokenized_input = tokenized_input.to(self.args.device)
            self.r_encoder.to("cpu")
            tokenized_input.to("cpu")
            # TODO last_hidden_state를 쓰는게 맞는가?
            encoder_output = self.r_encoder(tokenized_input["input_ids"].unsqueeze(0)).last_hidden_state
            encoder_output = encoder_output[:, 0, :].squeeze()
            concat_input = torch.cat([concat_input, encoder_output.detach().cpu()])

        print(tokenized_inputs.size())
        print(tokenized_inputs.dtype)
        print()
        print(tokenized_inputs[0])
        if labels is None:
            decoder_input_ids = shift_tokens_right(
                tokenized_inputs, self.encoder_tokenizer.pad_token_id, self.reader.config.decoder_start_token_id
            )
        else:
            decoder_input_ids = shift_tokens_right(
                labels, self.encoder_tokenizer.pad_token_id, self.reader.config.decoder_start_token_id
            )
        output = self.r_decoder(input_ids=decoder_input_ids, encoder_hidden_states=concat_input.unsqueeze(0))
        print(output.size())
        print(type(output))


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

    print(shifted_input_ids.size())
    print(shifted_input_ids.dtype)
    return shifted_input_ids
