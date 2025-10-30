import streamlit as st
import pandas as pd
from datetime import date
from src.preprocessing import preprocess_data_clv
from src.feature_engineering import feature_engineering_clv
from src.model_training import model_CLV
from src.evaluation import evaluation_clv

def run():
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.title("Dự đoán chi tiêu khách hàng (CLV)")

    if st.button("⬅️ Quay lại trang chính"):
        st.session_state.page = "dashboard"
        st.rerun()

    df = st.session_state.df
    st.dataframe(df.head(500), use_container_width=True, height=360)
    st.caption(f"Kích thước: {df.shape[0]} x {df.shape[1]}")

    col1, col2 = st.columns(2)
    today = col1.date_input(
        "Ngày theo dõi", 
        value=pd.to_datetime(df["DATE_"].max()).date() if "DATE_" in df.columns else date.today()
    )
    months_pred = col2.number_input("Số tháng dự đoán", min_value=1, max_value=24, value=3, step=1)

    if st.button("Chạy mô hình CLV"):
        with st.spinner("Đang huấn luyện mô hình..."):
            df_clean = preprocess_data_clv(df)
            rfm = feature_engineering_clv(df_clean, pd.to_datetime(today))
            res = model_CLV(rfm, pred_time=months_pred * 4)
            df_result = res["df_result"]
            clusters = res["clusters"]

            eval_result = evaluation_clv(df_result, clusters)
            metrics = eval_result["metrics"]
            customer_table = eval_result["customer_table"]
            segment_summary = eval_result["segment_summary"]

        st.success("✅ Hoàn tất dự đoán")

        st.markdown("### Dữ liệu sau khi đã tiền xử lý")
        st.dataframe(df_clean.head(500), use_container_width=True, height=360)
        st.caption(f"Kích thước sau tiền xử lý: {df_clean.shape[0]} x {df_clean.shape[1]}")

        st.subheader("Kết quả dự đoán CLV")
        st.dataframe(customer_table.head(200), use_container_width=True, height=400)

        st.caption(f"Kích thước bảng khách hàng: {customer_table.shape[0]} x {customer_table.shape[1]}")

        st.subheader("Thống kê cụm khách hàng")
        st.dataframe(segment_summary, use_container_width=True, height=180)

        st.subheader("Đánh giá mô hình CLV")
        st.dataframe(metrics, use_container_width=True, height=150)
