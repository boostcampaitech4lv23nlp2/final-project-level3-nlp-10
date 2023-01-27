import sys
import time
from collections import deque

import streamlit as st

sys.path.insert(0, ".")
queue = deque()
queue.append("nnew")
st.set_page_config(layout="wide")

# =======
#   App
# =======

with st.sidebar:
    st.title("회의 관리는 Boost2Note")
    st.caption("Made by Boost2End")
    st.markdown("---")

empty1, con1, empty2 = st.columns([0.1, 1.0, 0.1])
_, con_summary, option, _ = st.columns([0.1, 0.7, 0.2, 0.1])
empty1, con3, empty2 = st.columns([0.1, 1.0, 0.1])

with con1:
    upload_tab, record_tab = st.tabs(["녹음본 업로드", "녹음하기"])
    with upload_tab:
        file = st.file_uploader("녹음본 파일을 올려주세요.", key="file_uploader", accept_multiple_files=True)

        # st.write(st.session_state.get("file_uploader"))

        with st.spinner("변환중입니다..."):
            time.sleep(5)
        st.success("변환 완료")
        with st.expander("STT 원본 텍스트", expanded=True):
            st.markdown(
                """
                adfsdf
            """
            )
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
with con3:
    options = st.multiselect("주요 키워드", list(queue), ["nnew"])
    for i in range(3):
        with st.expander("? 키워드 요약", expanded=True):
            st.markdown(
                """
                adfsdf
            """
            )
