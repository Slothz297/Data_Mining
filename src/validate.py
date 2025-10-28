import pandas as pd

REQUIRED_COLUMNS = ["USERID", "DATE_", "ORDERID", "TOTALBASKET"]

def validate_file(filepath):
    """Kiểm tra file có thể dùng để train model hay không."""
    try:
        df = pd.read_csv(filepath)
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            return False, f"Thiếu cột: {', '.join(missing)}, dữ liệu không hợp lệ !"
        if df.empty:
            return False, "Dữ liệu rỗng"
        return True, df
    except Exception as e:
        return False, f"Lỗi đọc file: {e}"
