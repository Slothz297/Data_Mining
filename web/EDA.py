import streamlit as st
import pandas as pd
from src.eda import eda

def run():
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.title("📊 Phân tích dữ liệu (EDA)")

    if st.button("⬅ Quay lại trang chính"):
        st.session_state.page = "dashboard"
        st.rerun()

    df = st.session_state.df
    st.subheader("📋 Xem trước dữ liệu")
    st.dataframe(df.head(1000), use_container_width=True, height=500)

    st.subheader("📈 Phân tích tổng quan")
    with st.spinner("Đang phân tích..."):
        result = eda(df)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Kích thước dữ liệu:**", result["shape"])
        st.write("**Dòng đầu tiên:**")
        st.dataframe(result["head"], use_container_width=True)
        st.write("**Dòng cuối:**")
        st.dataframe(result["tail"], use_container_width=True)
    with c2:
        st.write("**Thông tin cột:**")
        st.dataframe(result["info"], use_container_width=True)
        st.write("**Thiếu dữ liệu:**")
        st.dataframe(result["missing"], use_container_width=True)

    st.subheader("📉 Thống kê mô tả")
    st.dataframe(result["describe"], use_container_width=True)
