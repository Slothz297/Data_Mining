import pandas as pd


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

    return df_clean
