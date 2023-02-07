import json
import sys
import time
from collections import deque

import requests
import streamlit as st
import streamlit_nested_layout
from streamlit_tags import st_tags

__all__ = ["streamlit_nested_layout"]

sys.path.insert(0, ".")
queue = deque()
st.set_page_config(layout="wide")


# =======
#   App
# =======


def wait_msg(msg, wait=3, type_="warning"):
    placeholder = st.empty()
    placeholder.warning(msg, icon="🤖")
    time.sleep(wait)
    placeholder.empty()


def call_stt(con, files, contents_list):
    with con:
        for file, contents in zip(files, contents_list):
            expander = st.expander(file.name, expanded=True)
            for content in contents:
                expander.markdown(content)


def call_summary(con, titles, contents):
    with con:
        for title, content in zip(titles, contents):
            expander = st.expander(title, expanded=True)
            expander.markdown(content)


def callback_upload():
    st.session_state["success"] = False


def check_files(uploaded_files):
    new_data = []
    if not st.session_state["stt"]:
        return
    for file, stt_data in zip(*st.session_state["stt"]):
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

if "uploaded" not in st.session_state:
    st.session_state["uploaded"] = []

empty1, con_stt, empty2 = st.columns([0.001, 1.0, 0.001])
stt_placeholder = st.empty()
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
        check_files(uploaded_files)

        stt_button = st.button("변환하기", key="stt_button", disabled=st.session_state["stt_disabled"])
        if stt_button:
            if uploaded_files is not None:
                files = []
                st.session_state["stt_disabled"] = True
                for uploaded_file in uploaded_files:
                    msg = "변환 중입니다.."
                    sound_bytes = uploaded_file.getvalue()
                    files.append(("files", (uploaded_file.name, sound_bytes, uploaded_file.type)))
                response = requests.post("http://localhost:8000/stt", files=files)
                result = response.json()
                print(result)
                st.session_state["stt_disabled"] = False
                # wait_msg(msg, 3, msg)
                st.session_state["keywords"] = None
                st.session_state["success"] = st.success("변환 완료")

        if st.session_state["success"]:
            if not st.session_state["keywords"]:
                st.session_state["keywords"] = result[1]
            if not st.session_state["stt"]:
                st.session_state["stt"] = [uploaded_files, result[0]]
            upload_files, stt_data = st.session_state["stt"]
            stt_placeholder.empty()
            call_stt(stt_placeholder.container(), uploaded_files, stt_data)
            # with st.expander("STT 원본 텍스트", expanded=True):
            # for stt_data in st.session_state["stt"]:
            #     st.write(stt_data)

    with record_tab:
        st.write("record")


with con2:
    if st.session_state["keywords"]:
        options = st.multiselect(
            "주요 키워드",
            st.session_state["keywords"],
        )
        input_keywords = st_tags(label="키워드 직접 입력", text="Press enter to add more")
        keywords_set = list(set(options + [i.strip(" ") for i in input_keywords if i.strip(" ") != ""]))
        # 띄어쓰기 제거 및 중복 제거

    else:
        options = st.multiselect(
            "주요 키워드",
            list(queue),
        )

    _, con_summary, _ = st.columns([0.001, 1.0, 0.001])
    # expanders = []
    # for i in range(3):
    #     expander = st.expander("? 키워드 요약", expanded=True)
    #     # expander.markdown("asddsf")
    #     expanders.append(expander)
    with con_summary:
        if st.button("요약하기", key="summarization"):
            response = requests.post("http://localhost:8000/summarize", json={"keywords": keywords_set})
            json_res = json.loads(response.text)  # json_res : list
            titles = ["키워드 요약"] * len(json_res)  # test
            stt_placeholder.empty()
            call_summary(con_summary, titles, json_res)
            upload_files, stt_data = st.session_state["stt"]
            call_stt(stt_placeholder.container(), uploaded_files, stt_data)
