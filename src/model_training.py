import dill

with open("models/clv_models.pkl", "rb") as f:
    models = dill.load(f)

bgf = models["bgf"]
ggf = models["ggf"]

with open("models/clv_models.pkl", "wb") as f:
    dill.dump(models, f)

def model_training(df_clv):
    """Huấn luyện mô hình CLV (BG/NBD + GammaGamma)."""
    bgf.fit(df_clv['frequency'], df_clv['recency'], df_clv['T'])
    ggf.fit(df_clv['frequency'], df_clv['monetary'])

    return bgf, ggf

