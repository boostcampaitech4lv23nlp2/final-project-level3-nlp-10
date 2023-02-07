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


def call(con, titles, contents):
    with con:
        for title, content in zip(titles, contents):
            expander = st.expander(title, expanded=True)
            expander.markdown(content)


if "keywords" not in st.session_state:
    st.session_state["keywords"] = None

if "filename" not in st.session_state:
    st.session_state["filename"] = ""

if "success" not in st.session_state:  # stt 및 문단화 성공에 대한 session state
    st.session_state["success"] = False
    st.session_state["stt"] = ""

empty1, con1, empty2 = st.columns([0.1, 1.0, 0.1])
empty1, con2, empty2 = st.columns([0.1, 1.0, 0.1])

result = None
with st.sidebar:
    st.title("회의 관리는 Boost2Note")
    st.caption("Made by Boost2End")
    st.markdown("---")

with con1:
    upload_tab, record_tab = st.tabs(["녹음본 업로드", "..."])
    with upload_tab:
        uploaded_files = st.file_uploader(
            "녹음 파일을 올려주세요.", key="file_uploader", accept_multiple_files=True, type=["wav", "mp4", "mp3"]
        )

        if st.button("변환하기", key="stt_button"):

            if uploaded_files is not None:
                for uploaded_file in uploaded_files:
                    msg = "변환 중입니다.."
                    sound_bytes = uploaded_file.getvalue()
                    files = [("files", (uploaded_file.name, sound_bytes, uploaded_file.type))]
                    response = requests.post("http://localhost:8000/stt", files=files)
                    result = response.json()
                    print(result)
                    wait_msg(msg, 3, msg)
                st.session_state["keywords"] = None
                st.session_state["success"] = st.success("변환 완료")

        if st.session_state["success"]:
            if not st.session_state["keywords"]:
                st.session_state["keywords"] = result[1]
            # stt_expanders = []
            # stt_expander = st.expander("STT 원본 텍스트", expanded=True)
            # call(con2, expanders, json_res)
            if not st.session_state["stt"]:
                st.session_state["stt"] = result[0]
            titles = ["stt 원본 텍스트"] * len(st.session_state["stt"])
            call(con1, titles, st.session_state["stt"])
            # with st.expander("STT 원본 텍스트", expanded=True):
            # for stt_data in st.session_state["stt"]:
            #     st.write(stt_data)

            # for passage in stt_data:
            #    st.write(passage)
            #    st.write(" ")

            # st.write(st.session_state["stt"])
            # _, _, _, con5, con6 = st.columns([0.2, 0.2, 0.1, 0.3, 0.2])
            # with con6:
            #     save = st.button("원본 텍스트 저장", key="save_txt")
            #     if save:
            #         response = requests.post("http://localhost:8000/save", files=st.session_state["stt"])
            #         label = response.json()
            #         st.write(f"label is {label}")

    with record_tab:
        st.write("record")

        # if file is not None:
        #     try:
        #         img = Image.open(file)
        #     except:
        #         st.error("The file you uploaded does not seem to be a valid image. Try uploading a png or jpg file.")
        # if st.session_state.get("image_url") not in ["", None]:
        #     st.warning("To use the file uploader, remove the image URL first.")


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

    con8, _ = st.columns([0.8, 0.2])
    # expanders = []
    # for i in range(3):
    #     expander = st.expander("? 키워드 요약", expanded=True)
    #     # expander.markdown("asddsf")
    #     expanders.append(expander)

    with con8:
        if st.button("요약하기", key="summarization"):
            response = requests.post("http://localhost:8000/summarize", json={"keywords": keywords_set})
            json_res = json.loads(response.text)  # json_res : list
            titles = ["키워드 요약"] * len(json_res)  # test
            call(con2, titles, json_res)
