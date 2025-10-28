import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def eda(df: pd.DataFrame):
    """
    Trả về các thông tin EDA dưới dạng dictionary có thể hiển thị lên giao diện.
    """

    # === 1. Head & Tail ===
    head_data = df.head(5)
    tail_data = df.tail(5)

    # === 2. Shape ===
    shape_info = {
        "rows": df.shape[0],
        "columns": df.shape[1]
    }

    # === 3. Info (tóm tắt theo cột) ===
    info_data = []
    for col in df.columns:
        info_data.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "nulls": int(df[col].isnull().sum()),
            "unique": int(df[col].nunique())
        })
    info_df = pd.DataFrame(info_data)

    # === 4. Columns ===
    columns = list(df.columns)

    # === 5. Quantiles (describe) ===
    describe_df = df.describe().T

    # === 6. Missing values ===
    nulls = df.isnull().sum().reset_index()
    nulls.columns = ['Column', 'MissingCount']
    nulls['MissingPercent'] = (nulls['MissingCount'] / len(df) * 100).round(2)

    # === 7. Biểu đồ trực quan hóa (nếu muốn dùng Streamlit plot) ===
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
    plt.title("Missing Values Map")

    # === 8. Kết quả trả về ===
    result = {
        "head": head_data,
        "tail": tail_data,
        "shape": shape_info,
        "info": info_df,
        "columns": columns,
        "describe": describe_df,
        "missing": nulls,
        "missing_plot": fig
    }
    return result
