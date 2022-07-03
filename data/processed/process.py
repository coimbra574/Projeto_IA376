
import pandas as pd

def process():
    df = pd.read_csv("results.csv")
    # df = df.groupby(["amostras", "proporção"]).agg({"x_avg_pixel": ["mean", "std"], "higher_than_threshold": ["mean"]}).reset_index().pivot(index=["amostras"], columns=["proporção"])
    df = df.groupby(["amostras", "proporção"]).agg({"higher_than_threshold": ["mean"]}).reset_index().pivot(index=["amostras"], columns=["proporção"])
    df = (df * 100).round(2)
    df.to_csv("results_processed.csv")
    return df
