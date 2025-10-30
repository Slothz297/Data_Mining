import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.cluster import KMeans
from prophet import Prophet
from xgboost import XGBRegressor
from src.feature_engineering import feature_engineering_branch

#Mô hình dự đoán CLV của khách hàng
"""
    Hàm này thưc hiện:
    - Training mô hình BG-NBD
    - Training mô hình Gamma Gamma
    - Thực hiện dự đoán để tính CLV
    - Phân loại khách hàng dựa vào CLV

    Kết quả trả về:
    - Mô hình BG-NBD
    - Mô hình Gamma Gamma
    - Bảng dự đoán CLV của khách hàng và phân loại khách hàng
"""
def model_CLV(rfm: pd.DataFrame, pred_time: int = 4): # Mặc định pred_time sẽ là 12 tuần ( 3 tháng )
    # Mô hình BG-NBD
    bgf = BetaGeoFitter(penalizer_coef=0.1)
    bgf.fit(rfm["frequency"], rfm["recency"], rfm["T"])

    # Mô hình GammaGamma
    ggf = GammaGammaFitter(penalizer_coef = 0.01)
    # Fitting dữ liệu cho model với frequency và monetary
    ggf.fit(rfm["frequency"], rfm["monetary"])

    # Thực hiện dự đoán
    rfm["predicted_frequency"] = bgf.predict(pred_time, rfm["frequency"], rfm["recency"], rfm["T"])
    rfm["predicted_monetary"] = ggf.conditional_expected_average_profit(rfm["frequency"], rfm["monetary"])
    rfm["CLV"] = rfm["predicted_frequency"] * rfm["predicted_monetary"]


    #Phân loại khách hàng bằng K-Mean
    X = rfm[["CLV"]].values 
    km_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    km_model.fit(X)
    rfm["cluster"] = km_model.labels_

    #Tính các giá trị CLV của từng cụm và thông số về % của từng cụm
    df_clusters = rfm.groupby("cluster")["CLV"].agg(["mean", "count"]).reset_index()
    df_clusters.columns = ["cluster", "avg_CLV", "n_customers"]
    df_clusters["percent_customers"] = (df_clusters["n_customers"] / df_clusters["n_customers"].sum() * 100).round(2)

    #Gán nhãn cho từng cụm
    df_clusters = df_clusters.sort_values("avg_CLV", ascending=True).reset_index(drop=True)
    labels = ["Bronze", "Silver", "Gold", "Diamond"]
    df_clusters["segment"] = labels[:len(df_clusters)]

    #Dán lại các nhãn cho bộ dự liệu chính
    mapping = dict(zip(df_clusters["cluster"], df_clusters["segment"]))
    rfm["customer_category"] = rfm["cluster"].map(mapping)

    result = {
    "bgf":bgf,
    "ggf":ggf,
    "df_result": rfm,
    "clusters": df_clusters
    }
    return result

def model_branch(df,time_pred):
    all_branch_forecasts = {}
    data = df.copy()
    # Mô hình Prophet
    for branch_id in data['BRANCH_ID'].unique():
        df_branch = data[data['BRANCH_ID'] == branch_id].copy()
        df_branch = df_branch.rename(columns={'DATE_': 'ds', 'TOTALBASKET': 'y'})
        df_branch = df_branch.groupby('ds')['y'].sum().reset_index()

        model_prophet = Prophet()
        model_prophet.fit(df_branch)

        #predicting the total basket 1 month ahead
        future = model_prophet.make_future_dataframe(periods= time_pred, freq='M')
        forecast = model_prophet.predict(future)
        all_branch_forecasts[branch_id] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    #Gộp dữ liệu vào dataframe

    all_branch_forecasts_df = pd.concat(
        [forecast.assign(BRANCH_ID=branch_id) for branch_id, forecast in all_branch_forecasts.items()],
        ignore_index=True)[['BRANCH_ID', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    """
        - ds: Mốc thời gian
        - yhat: Dự đoán trung bình
        - yhat_lower: Giới hạn dưới (95%)
        - yhat_upper: Giới hạn trên (95%)
    """

    #Mô hình XGBoost

    #Cắt bộ dữ liệu theo thời gian dự đoán
    latest_date = df['DATE_'].max()
    split_date = latest_date - pd.DateOffset(months=time_pred)

    #Chia bộ dữ liệu
    train = df[df['DATE_'] < split_date].reset_index(drop=True)
    test = df[df['DATE_'] >= split_date].reset_index(drop=True)

    #thực hiện feature_engineering
    x_train = feature_engineering_branch(train)
    y_train = train['TOTALBASKET']
    x_test = feature_engineering_branch(test)
    y_test = test['TOTALBASKET']

    #Training model
    model_xgb = XGBRegressor(n_estimators=1000)
    model_xgb.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        verbose=False
    )

    #Predict
    y_pred = model_xgb.predict(x_test)
    result = {
        "prophet": model_prophet,
        "xbg": model_xgb,
        "branch_forecast": all_branch_forecasts_df,
        "x_test": test,
        "y_test": y_test,
        "y_pred":y_pred
    }

    return result