import streamlit as st
import pandas as pd

def run():
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.set_page_config(page_title="Dashboard | 3A Superstore AI", layout="wide")
    st.title("3A Superstore Dashboard")

    # Nút quay lại trang chọn dữ liệu
    if st.button("⬅️ Quay lại trang chọn dữ liệu"):
        st.session_state.page = "main"
        st.rerun()

    # Kiểm tra dữ liệu có sẵn
    df = st.session_state.df

    # Hiển thị bảng dữ liệu
    st.markdown("### 🧾 Dữ liệu hiện tại")
    st.dataframe(df.head(100), use_container_width=True, height=400)

    # Hiển thị số lượng dòng và cột
    st.caption(f"**Tổng cộng:** {df.shape[0]} dòng × {df.shape[1]} cột")

    st.markdown("---")
    st.subheader("Chọn chức năng thực hiện")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Phân tích dữ liệu (EDA)", use_container_width=True):
            st.session_state.page = "eda"
            st.rerun()

    with col2:
        if st.button("Dự đoán chi tiêu khách hàng (CLV)", use_container_width=True):
            st.session_state.page = "clv"
            st.rerun()

    with col3:
        if st.button("Dự đoán doanh thu chi nhánh", use_container_width=True):
            st.session_state.page = "branch"
            st.rerun()
