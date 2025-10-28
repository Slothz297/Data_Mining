import pandas as pd
import dill

with open("models/clv_models.pkl", "rb") as f:
    models = dill.load(f)

kmeans = models["kmeans"]
cluster_labels = models["cluster_labels"]


def classify_customer(df):
    df_clv = df.copy()
    kmeans.fit(df_clv[['CLV']])
    df_clv['cluster'] = kmeans.labels
    df_clusters = df_clv.groupby(['cluster'])['CLV'].agg(['mean', "count"]).reset_index()
    df_clusters.columns = ["cluster", "avg_CLV", "n_customers"]
    df_clusters['percent_customers'] = (df_clusters['n_customers']/df_clusters['n_customers'].sum())*100
    df_cluster = df_clusters.sort_values("avg_CLV")

    # Gán thứ hạng theo thứ tự
    labels = ["Bronze", "Silver", "Gold", "Diamond"]
    df_cluster["segment"] = labels[:len(df_cluster)]

    df_cluster
    mapping = dict(zip(df_cluster["cluster"], df_cluster["segment"]))
    df_clv["customer_category"] = df_clv["cluster"].map(mapping)
    return df_clv


def model_predict(df_clv, bgf, ggf, weeks):
    """Dự đoán số lần mua và CLV."""
    df_clv["pred_freq"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        weeks, df_clv["frequency"], df_clv["recency"], df_clv["T"]
    )
    df_clv["exp_monetary"] = ggf.conditional_expected_average_profit(
        df_clv["frequency"], df_clv["monetary"]
    )
    df_clv["CLV"] = df_clv["pred_freq"] * df_clv["exp_monetary"]
    df_clv = classify_customer(df_clv)
    return df_clv
