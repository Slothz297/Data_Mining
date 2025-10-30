import pandas as pd
import datetime as dt
from sklearn.preprocessing import LabelEncoder

#Hàm feature_engineering tạo mẫu dữ liệu RFM cho model BG-NBD và Gamma Gamma
def feature_engineering_clv(df: pd.DataFrame,today_date):
    #Bộ dữ liệu phải có ngày giao dịch nhỏ hơn ngày theo dõi
    df_feature = df.copy()
    df_feature = df_feature[df_feature['DATE_'] <= today_date]

    #Nhóm các hóa đơn của user lại và tạo dataframe theo RFTM
    rfm = df_feature.groupby("USERID").agg({"DATE_" : [lambda date: (date.max() - date.min()).days, # recency
                                                lambda date: (today_date - date.min()).days], # T
                                    "ORDERID" : lambda order: order.nunique(), # frequnecy
                                    "TOTALBASKET": lambda total_basket: total_basket.sum()}).reset_index() # monatery
    # đặt tên cho các cột 
    rfm.columns = ['USERID', 'recency', 'T', 'frequency', 'monetary']
    #Tính giá trị monetary trung bình từng đơn hàng
    rfm["monetary"] = rfm["monetary"] / rfm["frequency"]

    #Chuyển đổi recency và T từ ngày sang tuần
    rfm["recency"] = round(rfm["recency"] / 7)  #1 week = 7 day
    rfm["T"] = round(rfm["T"] / 7)

    return rfm


#Hàm tạo dữ liệu cho mô hình dự đoán doanh thu chi nhánh
def feature_engineering_branch(df: pd.DataFrame):
    """
    Hàm tạo các đặc trưng thời gian (time-based features) cho mô hình dự đoán doanh thu chi nhánh.
    - Chuyển định dạng cột DATE_ sang datetime
    - Tạo các cột đặc trưng như ngày, tháng, năm, quý, thứ trong tuần,...
    - Mã hóa nhãn cho BRANCH_ID để mô hình có thể xử lý
    - Trả về DataFrame chứa các đặc trưng đầu vào (X)
    """

    df_feature = df.copy()

    """Nhập mô hình mã hóa nhãn LabelEncoder từ thư viện sklearn.preprocessing. 
    Việc này nhằm mã hóa các dữ liệu dạng Object (String) cho việc phân tích dữ liệu sau này, tránh gây rò rỉ dữ liệu."""
    label_encoder = LabelEncoder()
    df_feature['BRANCH_ID'] = label_encoder.fit_transform(df_feature['BRANCH_ID'])

    #  Tạo đặc trưng thời gian
    df_feature['day'] = df_feature['DATE_'].dt.day
    df_feature['month'] = df_feature['DATE_'].dt.month
    df_feature['year'] = df_feature['DATE_'].dt.year
    df_feature['dayofweek'] = df_feature['DATE_'].dt.dayofweek
    df_feature['quarter'] = df_feature['DATE_'].dt.quarter
    df_feature['dayofyear'] = df_feature['DATE_'].dt.dayofyear
    df_feature['dayofmonth'] = df_feature['DATE_'].dt.day
    df_feature['weekofyear'] = df_feature['DATE_'].dt.isocalendar().week
 
    x = df_feature[[
        'BRANCH_ID',
        'day',
        'month',
        'year',
        'dayofweek',
        'quarter',
        'dayofyear',
        'dayofmonth',
        'weekofyear'
    ]].copy()

    return x