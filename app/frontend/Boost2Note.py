from typing import NamedTuple

import io
import json
import sys
from collections import deque
from pathlib import Path

import requests
import streamlit as st
import streamlit_nested_layout
from streamlit_tags import st_tags

__all__ = ["streamlit_nested_layout"]

sys.path.insert(0, ".")
queue = deque()
st.set_page_config(layout="wide")

port = 30002  # backend 서버 port와 동일하게 설정 (POST 통신 port)
address = "http://118.67.130.208"  # "http://{본인 서버 IP 주소}"로 설정

# =======
#   App
# =======


class UploadedFileRec(NamedTuple):
    """Metadata and raw bytes for an uploaded file. Immutable."""

    id: str
    name: str
    type: str
    data: bytes


class UploadedFile(io.BytesIO):
    """A mutable uploaded file.
    This class extends BytesIO, which has copy-on-write semantics when
    initialized with `bytes`.
    """

    def __init__(self, record: UploadedFileRec):
        # BytesIO's copy-on-write semantics doesn't seem to be mentioned in
        # the Python docs - possibly because it's a CPython-only optimization
        # and not guaranteed to be in other Python runtimes. But it's detailed
        # here: https://hg.python.org/cpython/rev/79a5fbe2c78f
        super(UploadedFile, self).__init__(record.data)
        self.id = record.id
        self.name = record.name
        self.type = record.type
        self.size = len(record.data)


def call_stt(con, files, contents_list):
    with con:
        for file, contents in zip(files, contents_list):
            # st.audio(file.getvalue(), format="audio/ogg")
            expander = st.expander(file.name, expanded=True)

            for content in contents:
                expander.markdown(content)


def call_annotated_stt(con, files, contents_list, annotated_texts):
    with con:
        for file, contents in zip(files, contents_list):
            # st.audio(file.getvalue(), format="audio/ogg")
            expander = st.expander(file.name, expanded=True)
            for content in contents:
                if content in annotated_texts:
                    text = f"**:blue[{content}]**"
                else:
                    text = content
                expander.markdown(text)


def call_summary(con, titles, contents):
    with con:
        for title, content in zip(titles, contents):
            if type(title) == list:
                title = ", ".join(title)
            expander = st.expander(title, expanded=True)
            expander.markdown(content)


def callback_upload():
    st.session_state["success"] = False


def callback_multiselect(placeholder):
    upload_files, stt_data = st.session_state["stt"]
    placeholder.empty()
    call_stt(placeholder.container(), uploaded_files, stt_data)


def check_files(uploaded_files):
    # session에 저장된 파일과 file uploader에 저장된 파일 유무 비교
    new_data = []
    if not uploaded_files:
        return
    if not st.session_state["stt"]:
        return
    for file, stt_data in zip(*st.session_state["stt"]):
        print(file.id)
        for uploaded_file in uploaded_files:
            if file.id == uploaded_file.id:
                new_data.append([file, stt_data])
                break
    st.session_state["stt"] = list(zip(*new_data))


if "keywords" not in st.session_state:
    st.session_state["keywords"] = None

if "filename" not in st.session_state:
    st.session_state["filename"] = ""

if "success" not in st.session_state:  # stt 및 문단화 성공에 대한 session state
    st.session_state["success"] = False
    st.session_state["stt"] = ""

if "stt_disabled" not in st.session_state:
    st.session_state["stt_disabled"] = False

if "summary_disabled" not in st.session_state:
    st.session_state["summary_disabled"] = True

if "uploaded" not in st.session_state:
    st.session_state["uploaded"] = []

if "emb_token" not in st.session_state:
    st.session_state["emb_token"] = None

if "summaries" not in st.session_state:
    st.session_state["summaries"] = []

empty1, con_stt, empty2 = st.columns([0.001, 1.0, 0.001])
empty1, con2, empty2 = st.columns([0.001, 1.0, 0.001])

result = None
with st.sidebar:
    st.title("회의 관리는 Boost2Note")
    st.caption("Made by Boost2End")
    st.markdown("---")

with con_stt:
    upload_tab, record_tab = st.tabs(["녹음본 업로드", "..."])
    with upload_tab:
        uploaded_files = st.file_uploader(
            "녹음 파일을 올려주세요.",
            key="file_uploader",
            accept_multiple_files=True,
            type=["wav", "mp4", "mp3"],
            on_change=callback_upload,
        )
        check_files(uploaded_files)  # 남아있는 파일과 불일치 파일 확인
        ex_button = st.button("예시 파일 변환하기", key="ex_button", disabled=st.session_state["stt_disabled"])
        stt_button = st.button("변환하기", key="stt_button", disabled=st.session_state["stt_disabled"])
        if ex_button:
            audio_bytes = Path("./test2.mp3").read_bytes()
            file = UploadedFile(UploadedFileRec("3000", "example.mp3", "mp3", (audio_bytes)))
            uploaded_files.append(file)
            files = []
            st.session_state["keywords"] = None
            st.session_state["emb_token"] = None
            msg = "변환 중입니다.."
            warning_placeholder = st.empty()
            warning_placeholder.warning(msg, icon="🤖")
            for uploaded_file in uploaded_files:
                sound_bytes = uploaded_file.getvalue()
                files.append(("files", (uploaded_file.name, sound_bytes, uploaded_file.type)))

            response = requests.post(f"{address}:{port}/stt", files=files)
            result = response.json()
            print("result: ", result)
            st.session_state["keywords"] = None
            warning_placeholder.empty()
            st.session_state["success"] = st.success("변환 완료")

        elif stt_button:
            if uploaded_files is not None:
                files = []
                st.session_state["stt_disabled"] = True
                st.session_state["keywords"] = None
                st.session_state["emb_token"] = None
                msg = "변환 중입니다.."
                warning_placeholder = st.empty()
                warning_placeholder.warning(msg, icon="🤖")
                for uploaded_file in uploaded_files:
                    sound_bytes = uploaded_file.getvalue()
                    files.append(("files", (uploaded_file.name, sound_bytes, uploaded_file.type)))

                response = requests.post(f"{address}:{port}/stt", files=files)
                result = response.json()
                print("result: ", result)
                st.session_state["stt_disabled"] = False
                st.session_state["keywords"] = None
                warning_placeholder.empty()
                st.session_state["success"] = st.success("변환 완료")

        stt_placeholder = st.empty()
        if st.session_state["success"]:
            if not st.session_state["keywords"]:
                print("keywords:", result[1])
                st.session_state["keywords"] = result[1]
            if not st.session_state["emb_token"]:
                print("emb_token:", result[2])
                st.session_state["emb_token"] = result[2]
            if not st.session_state["stt"]:
                print("stt:", result)
                st.session_state["stt"] = [uploaded_files, result[0]]
            uploaded_files, stt_data = st.session_state["stt"]
            print("check: ", st.session_state["stt"])
            stt_placeholder.empty()
            call_stt(stt_placeholder.container(), uploaded_files, stt_data)
    with record_tab:
        st.write("직접 녹음 기능은 업데이트 예정입니다. ")


with con2:
    if st.session_state["keywords"]:
        options = st.multiselect(
            "주요 키워드", st.session_state["keywords"], on_change=callback_multiselect, args=(stt_placeholder,)
        )
        input_keywords = st_tags(label="키워드 직접 입력", text="Press enter to add more")
        keywords_set = list(
            set(options + [i.strip(" ") for i in input_keywords if i.strip(" ") != ""])
        )  # 띄어쓰기 제거 및 중복 제거
        if len(keywords_set) > 0:
            st.session_state["summary_disabled"] = False
        else:
            st.session_state["summary_disabled"] = True

    else:
        options = st.multiselect("주요 키워드", list(queue), on_change=callback_multiselect, args=(stt_placeholder,))
    summary_button = st.button("요약하기", key="summarization", disabled=st.session_state["summary_disabled"])
    if summary_button:
        st.session_state["summary_disabled"] = True
        msg = "요약 중입니다.."
        warning_placeholder = st.empty()
        warning_placeholder.warning(msg, icon="🤖")
        response = requests.post(
            f"{address}:{port}/summarize", json={"keywords": keywords_set, "emb_name": st.session_state["emb_token"]}
        )
        json_res = json.loads(response.text)  # json_res : list
        warning_placeholder.empty()
        if len(json_res[0][0]) > 100:
            st.session_state["summaries"].append((json_res[0], ["Jail Break: HELP..."]))  # Easter Egg
        else:
            st.session_state["summaries"].append((json_res[0], json_res[1]))
        upload_files, stt_data = st.session_state["stt"]
        stt_placeholder.empty()
        call_annotated_stt(stt_placeholder.container(), uploaded_files, stt_data, json_res[2])
        st.session_state["summary_disabled"] = True

    for title, summary in reversed(st.session_state["summaries"]):
        call_summary(con2, title, summary)
