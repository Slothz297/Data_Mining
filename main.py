import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="3A Superstore AI Dashboard", layout="wide")

DATA_DIR = "data"
REQUIRED_COLUMNS = ["USERID", "ORDERID", "BRANCH_ID", "DATE_", "NAMESURNAME", "TOTALBASKET"]

# ----------------------------
# Khởi tạo trạng thái
# ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "main"
if "df" not in st.session_state:
    st.session_state.df = None

# ----------------------------
# TRANG CHÍNH
# ----------------------------
if st.session_state.page == "main":
    st.title("3A Superstore AI Dashboard")
    st.subheader("Chọn và tải dữ liệu")

    os.makedirs(DATA_DIR, exist_ok=True)
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

    # --- Chọn file ---
    if files:
        selected = st.selectbox("Chọn file dữ liệu:", files)
        selected_path = os.path.join(DATA_DIR, selected)
        st.session_state.df = pd.read_csv(selected_path)
        st.success(f"Đã tải dữ liệu: {selected}")
    else:
        st.info("⚠️ Chưa có dữ liệu trong thư mục data/. Hãy tải dữ liệu lên bên dưới.")

    # --- Upload ---
    uploaded = st.file_uploader("Tải dữ liệu mới", type=["csv"])
    if uploaded:
        save_path = os.path.join(DATA_DIR, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Đã lưu: {uploaded.name}. Hãy tải lại trang để chọn.")

    # --- Hiển thị trước dữ liệu ---
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("### 📋 Xem trước dữ liệu")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Số dòng: {df.shape[0]} | Số cột: {df.shape[1]}")

    st.markdown("---")

    # --- Nút sang Dashboard ---
    if st.button("Tiếp tục đến Dashboard"):
        if st.session_state.df is None:
            st.error("⚠️ Chưa có dữ liệu để kiểm tra.")
        else:
            df = st.session_state.df
            df_cols = list(df.columns)
            missing = [c for c in REQUIRED_COLUMNS if c not in df_cols]
            if missing:
                st.error(f" Dữ liệu không hợp lệ. Thiếu cột: {missing}")
            else:
                st.session_state.page = "dashboard"
                st.rerun()

# ----------------------------
# TRANG DASHBOARD
# ----------------------------
elif st.session_state.page == "dashboard":
    import web.Dashboard as dash
    dash.run()
elif st.session_state.page == "eda":
    import web.EDA as eda_page
    eda_page.run()

elif st.session_state.page == "clv":
    import web.Model_clv as clv_page
    clv_page.run()

elif st.session_state.page == "branch":
    import web.Model_branch as branch_page
    branch_page.run()
