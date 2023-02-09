import sys
import time
from collections import deque

import requests
import streamlit as st
import streamlit_nested_layout

__all__ = ["streamlit_nested_layout"]

sys.path.insert(0, ".")
queue = deque()
queue.append("nnew")
st.set_page_config(layout="wide")


# =======
#   App
# =======


def wait_msg(msg, wait=3, type_="warning"):
    placeholder = st.empty()
    placeholder.warning(msg, icon="🤖")
    time.sleep(wait)
    placeholder.empty()


if "filename" not in st.session_state:
    st.session_state["filename"] = ""
if "success" not in st.session_state:
    st.session_state["success"] = False
    st.session_state["stt"] = ""


empty1, con1, empty2 = st.columns([0.1, 1.0, 0.1])
empty1, con2, empty2 = st.columns([0.1, 1.0, 0.1])

with st.sidebar:
    st.title("회의 관리는 Boost2Note")
    st.caption("Made by Boost2End")
    st.markdown("---")

with con1:
    upload_tab, record_tab = st.tabs(["녹음본 업로드", "..."])

    with upload_tab:
        uploaded_file = st.file_uploader(
            "녹음 파일을 올려주세요. (wav, mp4, mp3)",
            key="file_uploader",
            accept_multiple_files=False,
            type=["wav", "mp4", "mp3", "png"],
        )
        print(uploaded_file)
        print(st.session_state.filename)
        if uploaded_file and uploaded_file.name != st.session_state["filename"]:
            msg = "변환 중입니다.."
            # TODO: 올리면 save 되어야 함.
            st.write(uploaded_file.name)
            sound_bytes = uploaded_file.getvalue()
            files = [("files", (uploaded_file.name, sound_bytes, uploaded_file.type))]
            response = requests.post("http://localhost:8000/stt", files=files)
            result = response.json()
            # print(type(result), result.text)
            wait_msg(msg, 3, msg)
            st.session_state["filename"] = uploaded_file.name
            st.session_state["success"] = st.success("변환 완료")

        if st.session_state["success"]:
            with st.expander("STT 원본 텍스트", expanded=True):
                if not st.session_state["stt"]:
                    st.session_state["stt"] = result["text"]
                st.write(st.session_state["stt"])
                _, _, _, con5, con6 = st.columns([0.2, 0.2, 0.1, 0.3, 0.2])
                with con5:
                    save = st.button("원본 텍스트 저장", key="save_txt")
                    if save:
                        response = requests.post("http://localhost:8000/save", files=["save"])
                        label = response.json()
                        st.write(f"label is {label}")
                with con6:
                    st.button("초기화", key="reset_txt")
        # with st.container():
        #     st.write("This is inside the container")
        # if st.session_state.get("file_uploader") is not None:
        #     st.warning("To use the Gallery, remove the uploaded image first.")
        if st.session_state.get("image_url") not in ["", None]:
            st.warning("To use the Gallery, remove the image URL first.")

    with record_tab:
        st.write("record")

        # if file is not None:
        #     try:
        #         img = Image.open(file)
        #     except:
        #         st.error("The file you uploaded does not seem to be a valid image. Try uploading a png or jpg file.")
        # if st.session_state.get("image_url") not in ["", None]:
        #     st.warning("To use the file uploader, remove the image URL first.")
    st.markdown("---")

with con2:
    options = st.multiselect("주요 키워드", list(queue), ["nnew"])
    con8, _ = st.columns([0.8, 0.2])
    with con8:
        st.button("요약하기", key="summarization")
    for i in range(3):
        with st.expander("? 키워드 요약", expanded=True):
            st.markdown(
                """
                adfsdf
            """
            )