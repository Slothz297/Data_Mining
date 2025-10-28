import streamlit as st
import pandas as pd
import time
from src.eda import eda_summary  # file eda.py đã định nghĩa trước đó

st.set_page_config(page_title="📊 Exploratory Data Analysis", layout="wide")

# --- HEADER ---
st.title("📊 Phân tích khám phá dữ liệu (EDA)")

# --- LẤY DỮ LIỆU ---
# Dữ liệu được lưu trong session_state khi chọn ở trang main.py
if 'df' not in st.session_state:
    st.warning("⚠️ Chưa có dữ liệu được chọn. Đang tải dữ liệu mẫu...")
    df = pd.read_csv("data/3a_superstore.csv")
    st.session_state.df = df
else:
    df = st.session_state.df

# --- HIỂN THỊ THÔNG TIN DỮ LIỆU HIỆN TẠI ---
st.subheader("📂 Dữ liệu hiện tại đang phân tích")
st.dataframe(df.head(10), use_container_width=True)

# --- NÚT PHÂN TÍCH DỮ LIỆU ---
if st.button("🚀 Bắt đầu phân tích dữ liệu"):
    with st.spinner("⏳ Đang phân tích dữ liệu, vui lòng chờ..."):
        time.sleep(1.5)
        eda = eda_summary(df)
    st.success("✅ Hoàn tất phân tích dữ liệu!")

    # ==================== HIỂN THỊ KẾT QUẢ EDA ====================
    st.markdown("---")
    st.header("📈 Thông tin phân tích dữ liệu")

    # Toggle controls (nút gạt bật/tắt)
    st.sidebar.header("🔧 Tuỳ chọn hiển thị")
    show_head = st.sidebar.toggle("Hiển thị Head", value=True)
    show_tail = st.sidebar.toggle("Hiển thị Tail", value=False)
    show_info = st.sidebar.toggle("Hiển thị Info", value=True)
    show_shape = st.sidebar.toggle("Hiển thị Shape", value=True)
    show_columns = st.sidebar.toggle("Hiển thị Columns", value=False)
    show_describe = st.sidebar.toggle("Hiển thị Describe", value=True)
    show_missing = st.sidebar.toggle("Hiển thị Missing Values", value=True)

    # --- Head ---
    if show_head:
        st.subheader("📋 Dữ liệu đầu tiên (Head)")
        st.dataframe(eda["head"], use_container_width=True)

    # --- Tail ---
    if show_tail:
        st.subheader("📋 Dữ liệu cuối cùng (Tail)")
        st.dataframe(eda["tail"], use_container_width=True)

    # --- Shape ---
    if show_shape:
        st.subheader("📐 Kích thước dữ liệu (Shape)")
        st.json(eda["shape"])

    # --- Info ---
    if show_info:
        st.subheader("📘 Thông tin các cột (Info)")
        st.dataframe(eda["info"], use_container_width=True)

    # --- Columns ---
    if show_columns:
        st.subheader("🧱 Danh sách các cột")
        st.write(eda["columns"])

    # --- Describe ---
    if show_describe:
        st.subheader("📊 Thống kê mô tả (Describe)")
        st.dataframe(eda["describe"], use_container_width=True)

    # --- Missing Values ---
    if show_missing:
        st.subheader("⚠️ Dữ liệu bị thiếu (Missing Values)")
        st.dataframe(eda["missing"], use_container_width=True)
        st.pyplot(eda["missing_plot"])

    st.markdown("---")
    st.info("💡 Mẹo: Bạn có thể bật/tắt từng phần trong sidebar bên trái để tuỳ chỉnh hiển thị.")

else:
    st.info("👈 Nhấn **'Bắt đầu phân tích dữ liệu'** để xem kết quả EDA.")

