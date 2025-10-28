from flask import Flask, render_template, request
import pandas as pd, os
from src.validate import validate_file
from src.preprocessing import preprocess_data
from src.model_training import model_training
from src.model_predict import model_predict

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
app.config["UPLOAD_FOLDER"] = "uploads"


@app.route('/favicon.ico')
def favicon():
    """Trả favicon cho trình duyệt"""
    return send_from_directory(
        os.path.join(app.static_folder, 'img'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return render_template("index.html", error="Chưa chọn file.")
        
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        valid, data = validate_file(path)
        if not valid:
            return render_template("index.html", error=data)

        preview = data.head(20).to_html(classes="table table-striped table-sm", index=False)
        return render_template("index.html", preview=preview, success="Tải dữ liệu thành công.")
    
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    today = pd.to_datetime(request.form["today_date"])
    future = pd.to_datetime(request.form["predict_date"])
    weeks = (future - today).days / 7

    df = pd.read_csv(os.path.join(app.config["UPLOAD_FOLDER"], os.listdir(app.config["UPLOAD_FOLDER"])[-1]))
    df_clv = preprocess_data(df, today)
    bgf, ggf = model_training(df_clv)
    result = model_predict(df_clv, bgf, ggf, weeks)

    return render_template("index.html",
                           table=result.to_html(classes="table table-bordered table-sm", index=False),
                           stats=result["CLV"].describe().to_frame().to_html(classes="table table-sm"))

if __name__ == "__main__":
    app.run(debug=True)
