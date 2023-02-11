import json

import torch
import torch.nn.functional as F
from app_utils.cache_load import load_retriever, load_sbert


def convert2context(json_path):
    """
        stt의 결과물로 받은 output.json 파일을 모델이 사용할 수 있는 context로 변경합니다.
    Args:
        json_path (str): output.json 파일 경로

    Returns:
        List[str]: 발화문 단위로 나눈 str 리스트. 한 사람이 연속적으로 말한 발화문은 연결되어 하나의 str을 구성한다.
    """
    with open(json_path, "r") as f:
        speech = json.load(f)
    # result만 있고 없고 분기
    last_label = -1
    context = []
    text = ""
    for segment in speech["segments"]:
        label = segment["speaker"]["label"]
        if last_label == label:
            text += " " + segment["textEdited"]
        else:
            context.append(text)
            text = segment["textEdited"]
            last_label = label
    return context


def create_context_embedding(text_list: list, renew_emb=True, emb_name=None):
    """meeting record를 통해 새로운 embedding vector들을 계산하여 저장합니다.

    Args:
        record_path (str): 회의 기록이 담겨있는 json 파일(현재 Naver Clova Speech의 output에 맞춰져있음)
    """
    retriever = load_retriever()
    retriever.passages = text_list
    retriever.create_passage_embeddings(renew_emb=renew_emb, emb_name=emb_name)


def get_sentence_embedding(model, text: list, batch_size: int = 16) -> torch.Tensor:
    embeddings = []
    n_batch = len(text) // batch_size + 1
    for i in range(n_batch):
        with torch.no_grad():
            embedding = model.encode(
                sentences=text[batch_size * i : batch_size * (i + 1)],
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
                device="cuda:0",
            )
            embeddings.append(embedding.cpu())
    return torch.cat(embeddings).squeeze()


def split_passages(segments, threshold=0.5):
    if len(segments) <= 1:
        return segments[0]["text"]
    passages = []
    for segment in segments:
        passages.append(segment["text"])
    model = load_sbert()
    embedding = get_sentence_embedding(model=model, text=passages, batch_size=16)
    emb_len = len(embedding)
    sim_matrix = torch.zeros((emb_len, emb_len), dtype=torch.float32)

    for i in range(emb_len):
        sim = F.cosine_similarity(embedding[[i]], embedding)
        sim_matrix[i] += sim
    sim_matrix = sim_matrix.numpy()
    # print(sim_matrix)
    splits = [-1] * len(sim_matrix)
    for j in range(emb_len):
        for i in range(j, emb_len):
            if sim_matrix[j][i] > threshold and splits[i] == -1:
                splits[i] = j

    re_passages = []
    split = splits[0]
    passage = ""
    for s, p in zip(splits, passages):
        if split == s:
            passage += p
        else:
            re_passages.append(passage)
            passage = p
    re_passages.append(passage)
    return re_passages
