import pandas as pd
import numpy as np
#Tiền xử lý dữ liệu


def preprocess_data_clv(df):
    #Chuyển đổi ngày giao dịch sang dạng ngày
    df_clean = df.copy()
    df_clean['DATE_'] = pd.to_datetime(df_clean['DATE_'])

    
    #Xử lý TOTALBASKET chuyển dấu , sang dấu . và chuyển dạng dữ liệu sang float
    df_clean["TOTALBASKET"] = df_clean["TOTALBASKET"].str.replace(",", ".").astype(float)

    #Xử lý dữ liệu outlier
    Q1 = df_clean['TOTALBASKET'].quantile(0.01)
    Q3 = df_clean['TOTALBASKET'].quantile(0.99)
    IQR = Q3 - Q1

    upper = Q3 + 1.5 * IQR
    # Thay giá trị vượt ngưỡng trên (Không dùng lower vì không có giá trị âm)
    df_clean = df_clean[df_clean['TOTALBASKET'] <= upper]
    df_clean = df_clean.drop(columns= "BRANCH_ID")
    df_clean = df_clean.drop(columns= "NAMESURNAME")
    return df_clean

def preprocess_data_branch(df):
    data_clean = df.copy()
    #Loại bỏ các cột không cần thiết
    # drop_cols = [col for col in ['NAMESURNAME', 'USERID', 'ORDERID'] if col in data_clean.columns]
    # if drop_cols:
    #     data_clean = data_clean.drop(columns=drop_cols)

    # Chuyển định dạng ngày
    data_clean['DATE_'] = pd.to_datetime(data_clean['DATE_'], errors='coerce')

    data_clean["TOTALBASKET"] = data_clean["TOTALBASKET"].str.replace(",", ".").astype(float)

    #Xử lý ngoại lệ
    q99 = np.percentile(data_clean['TOTALBASKET'].dropna(), 99, interpolation='midpoint')
    data_clean = data_clean[data_clean['TOTALBASKET'] < q99]

    #Dọn dữ liệu bị NULL
    data_clean = data_clean.dropna(subset=['TOTALBASKET', 'DATE_'], how='any')

    #Gộp doanh thu theo trong 1 tháng của từng chi nhánh   

    data = data_clean.groupby([data_clean['BRANCH_ID'], data_clean['DATE_'].dt.to_period('M')])['TOTALBASKET'].sum().reset_index()
    data['DATE_'] = data['DATE_'].dt.to_timestamp()
    data.columns = ['BRANCH_ID', 'DATE_', 'TOTALBASKET']


    return data