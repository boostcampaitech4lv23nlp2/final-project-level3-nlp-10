import json

from app_utils.cache_load import load_retriever


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


def create_context_embedding(record_path):
    """meeting record를 통해 새로운 embedding vector들을 계산하여 저장합니다.

    Args:
        record_path (str): 회의 기록이 담겨있는 json 파일(현재 Naver Clova Speech의 output에 맞춰져있음)
    """
    retriever = load_retriever()
    retriever.passages = convert2context(record_path)
    retriever.create_passage_embeddings(renew_emb=True)
