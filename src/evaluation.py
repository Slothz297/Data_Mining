import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def evaluation_clv(rfm, clusters):
    # So sánh monetary thực tế và monetary dự đoán
    mae = mean_absolute_error(rfm["monetary"], rfm["predicted_monetary"])
    rmse = root_mean_squared_error(rfm["monetary"], rfm["predicted_monetary"])
    r2 = r2_score(rfm["monetary"], rfm["predicted_monetary"])

    metrics =  pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R2"],
        "Value": [round(mae, 4), round(rmse, 4), round(r2, 4)]
    })
    # Bảng thống kê các cụm
    segment_summary = clusters[["segment", "avg_CLV", "n_customers", "percent_customers"]].copy()
    segment_summary.columns = ["Phân khúc", "CLV trung bình", "Số khách hàng", "Tỷ lệ (%)"]

    # Phân khúc khách hàng   
    customer_table = rfm[["USERID", "frequency", "recency", "monetary","predicted_frequency", "predicted_monetary",  "CLV", "customer_category"]].copy()
    customer_table.columns = ["ID", "Frequency", "Recency", "Monetary","Frequency_avg dự đoán","Monetary_avg dự đoán", "CLV dự đoán", "Hạng khách hàng"]

    result = {
        "metrics": metrics,
        "segment_summary": segment_summary,
        "customer_table": customer_table
    }

    return result


def evaluation_prophet(df, all_branch_forecasts):
    # Tạo một dataframe để lưu kết quả đánh giá
    evaluation_prophet = pd.DataFrame(columns=['Branch_ID', 'MAE', 'RMSE', 'R2'])
    df_original = df.copy()
    df_original["YearMonth"] = df_original["DATE_"].dt.to_period("M").dt.to_timestamp()
    actual_grouped = (
        df_original.groupby(["BRANCH_ID", "YearMonth"], as_index=False)["TOTALBASKET"].sum()
    ).rename(columns={"YearMonth": "ds", "TOTALBASKET": "y"})

    # Mô hình Prophet
    for branch_id in all_branch_forecasts["BRANCH_ID"].unique():
        try:
            df_branch_actual = actual_grouped[actual_grouped["BRANCH_ID"] == branch_id]
            df_branch_pred = all_branch_forecasts[all_branch_forecasts["BRANCH_ID"] == branch_id]

            merged = pd.merge(df_branch_actual, df_branch_pred, on=["BRANCH_ID", "ds"], how="inner")

            if merged.empty:
                continue

            mae = mean_absolute_error(merged["y"], merged["yhat"])
            rmse = root_mean_squared_error(merged["y"], merged["yhat"])
            r2 = r2_score(merged["y"], merged["yhat"])

            evaluation_prophet = pd.concat([
                evaluation_prophet,
                pd.DataFrame({
                    "Branch_ID": [branch_id],
                    "MAE": [round(mae, 4)],
                    "RMSE": [round(rmse, 4)],
                    "R2": [round(r2, 4)]
                })
            ], ignore_index=True)
        except Exception as e:
            print(f"Lỗi khi đánh giá chi nhánh {branch_id}: {e}")

    return evaluation_prophet


def evaluation_xgb(y_test, y_pred):
    rmse = np.sqrt(root_mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Tạo dataframe hiển thị
    result = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R2"],
        "Value": [round(mae, 4), round(rmse, 4), round(r2, 4)]
    })

    return result
