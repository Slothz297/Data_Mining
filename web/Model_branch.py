import streamlit as st
import pandas as pd
from src.model_training import model_branch
from src.evaluation import evaluation_prophet, evaluation_xgb
from src.preprocessing import preprocess_data_branch


def run():
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.title("Dự đoán doanh thu chi nhánh")

    if st.button("⬅️ Quay lại trang chính"):
        st.session_state.page = "dashboard"
        st.rerun()

    df = st.session_state.df
    st.dataframe(df.head(500), use_container_width=True, height=350)
    st.caption(f"Kích thước: {df.shape[0]} x {df.shape[1]}")

    st.markdown("### Tham số mô hình")
    time_pred = st.number_input("Số tháng dự đoán", min_value=1, max_value=12, value=1, step=1)

    if st.button("Chạy mô hình dự đoán doanh thu"):
        with st.spinner("Đang huấn luyện mô hình..."):
            df_clean = preprocess_data_branch(df)
            result = model_branch(df_clean, time_pred)
            all_branch_forecasts = result["branch_forecast"]
            y_test = result["y_test"]
            y_pred = result["y_pred"]
            x_test = result["x_test"]

        st.success("Đã huấn luyện mô hình xong!")

        st.markdown ("### Dữ liệu sau khi đã xử lý")
        st.dataframe(df_clean, use_container_width=True, height=350)
        st.caption(f"Kích thước sau tiền xử lý: {df_clean.shape[0]} x {df_clean.shape[1]}")

        # Prophet
        st.markdown("Kết quả dự đoán bằng Prophet")
    

        if not all_branch_forecasts.empty:
            st.dataframe(
                all_branch_forecasts.head(1000),
                use_container_width=True,
                height=400
            )
            st.caption(f"Tổng số bản ghi dự đoán: {len(all_branch_forecasts):,}")

            with st.spinner("🔍 Đang đánh giá mô hình Prophet..."):
                prophet_eval = evaluation_prophet(df_clean, all_branch_forecasts)

            st.markdown("Đánh giá mô hình Prophet")
            st.dataframe(prophet_eval, use_container_width=True, height=300)

            # Hiển thị thống kê tóm tắt
            avg_mae = prophet_eval["MAE"].mean()
            avg_rmse = prophet_eval["RMSE"].mean()
            avg_r2 = prophet_eval["R2"].mean()

            st.metric("MAE trung bình", f"{avg_mae:.2f}")
            st.metric("RMSE trung bình", f"{avg_rmse:.2f}")
            st.metric("R² trung bình", f"{avg_r2:.3f}")
        else:
            st.warning("⚠️ Không có dữ liệu hợp lệ cho mô hình Prophet.")

        # XGBoost
        st.markdown("---")
        st.subheader("Kết quả dự đoán bằng XGBoost")
        df_compare = pd.DataFrame({
            "BRANCH_ID":x_test['BRANCH_ID'],
            "Month": x_test["DATE_"],
            "y_test": y_test.values,
            "y_pred": y_pred
        })
        st.dataframe(df_compare.tail(200), use_container_width=True, height=300)

        xgb_eval = evaluation_xgb(y_test, y_pred)
        st.dataframe(xgb_eval, use_container_width=True, height=150)
