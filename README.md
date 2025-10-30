## **3A Retail CLV & Branch Forecasting**

Dự án **dự đoán chi tiêu khách hàng (Customer Lifetime Value – CLV)**, **phân loại khách hàng theo nhóm chi tiêu**, và **dự đoán doanh thu cho từng chi nhánh** dựa trên bộ dữ liệu **3A Superstore** – một tập dữ liệu mô phỏng về chuỗi siêu thị bán lẻ tại Thổ Nhĩ Kỳ (dữ liệu giả lập, không phải dữ liệu thật).

Người thực hiện: **Trịnh Á Châu(Slothz)  -  Ngô Nhật Minh**

Lớp: **S25-64CNTT**

Bộ dữ liệu : **3A Superstore (Market Orders Data-CRM)**

Link: **https://www.kaggle.com/datasets/cemeraan/3a-superstore**

---

## **Tổng quan**
Dự án được xây dựng bằng **Python** và **Streamlit**, cho phép hiển thị giao diện web trực quan phục vụ phân tích dữ liệu, huấn luyện mô hình, và hiển thị kết quả dự báo.

Bộ dữ liệu mẫu nằm trong thư mục `data/`, gồm:
- `Orders_3M.csv` – dữ liệu hóa đơn bán hàng trong 3 tháng của của doanh nghiệp được trích từ dữ liệu gốc.
- `Orders.zip`, `Orders.z01`, `Orders.z02`, ... – bản dữ liệu gốc đã được chia nhỏ.
  
> **Lưu ý:** Để sử dụng dữ liệu gốc, hãy giải nén các file `.zip` và chọn bộ dữ liệu gốc tại trang đầu tiên của web.

---
## **Cấu trúc thư mục**
```bash
project/
│
├── data/
│ ├── Orders_3M.csv
│ ├── Orders.zip
│ ├── Orders.z01, Orders.z02, ...
│
├── src/ # Các hàm chức năng chính
│ ├── preprocessing.py         # Tiền xử lý dữ liệu
│ ├── eda.py                   # Phân tích khám phá dữ liệu
│ ├── feature_engineering.py   # Tạo đặc trưng đầu vào
│ ├── model_training.py        # Huấn luyện mô hình
│ └── evaluation.py            # Đánh giá mô hình
│
├── web/                       # Giao diện Streamlit
│ ├── EDA.py
│ ├── Model_clv.py
│ ├── Model_branch.py
│ └── Dashboard.py
│
├── main.py                    # File chạy chính
├── Report.pdf                 # Báo cáo tổng hợp kết quả
├── requirements.txt           # Thư viện cài đặt
└── README.md
```
---
## **Hướng dẫn cài đặt**

Streamlit là thư viện mã nguồn mở của Python giúp tạo ứng dụng web tương tác cho các dự án Machine Learning, Data Science mà không cần biết HTML/CSS/JS

Hỗ trợ đa nền tảng: **Windows, macOS, Linux** ( phiên bản Python ≥ 3.8)

Có hỗ trợ deploy trang web trên:

- Streamlit Cloud (miễn phí)
- Hugging Face Spaces
- Render, Heroku, AWS, Google Cloud, v.v.

Yêu cầu cài đặt: 

Phiên bản Python ≥ 3.8

Tải phiên bản **[Python mới nhất](https://www.python.org/downloads/)**

---
### **Windows**
Kiểm tra phiên bản python
```bash
python --version
```
Nếu chưa có python hoặc python < 3.8 thì cài đặt và nâng cấp phiên bản python

Thực hiện cài đặt dự án và thư viện
```bash
#Clone project
git clone  https://github.com/Slothz297/Data_Mining.git
cd Data_Mining

#Cập nhật pip
python -m pip install --upgrade pip

#Cài đặt thư viện
pip install -r requirements.txt
```

Sau khi đã cài đặt thư viện đầy đủ, sử dụng lệnh sau để chạy web
```bash
streamlit run main.py
```
Sau khi chạy lệnh sẽ đưa link web có dạng localhost:8501

---
### **Linux/macOS**

Kiểm tra phiên bản python
```bash
python3 --version
```
Nếu chưa có python hoặc python < 3.8 thì cài đặt và nâng cấp phiên bản python

Cài đặt Python

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install python3 python3-pip -y

# macOS (Homebrew)
brew install python3
```
Thực hiện cài đặt dự án và thư viện

```bash
#Clone project
git clone  https://github.com/Slothz297/Data_Mining.git
cd Data_Mining

#Cập nhật pip
python3 -m pip install --upgrade pip

#Cài đặt thư viện
pip3 install -r requirements.txt
```

Sau khi đã cài đặt thư viện đầy đủ, sử dụng lệnh sau để chạy web
```bash
streamlit run main.py
```
Sau khi chạy lệnh sẽ đưa link web có dạng localhost:8501

---

## **Lincse**
Dự án này được phát hành theo giấy phép [MIT License](https://opensource.org/licenses/MIT).







