import pandas as pd

from pipeline import load_data

def test_dataset_not_empty():

df = load_data(
    "data/telco_churn.csv"
)

assert len(df) > 7000

def test_target_exists():

df = load_data(
    "data/telco_churn.csv"
)

assert "Churn" in df.columns

def test_target_distribution():

df = load_data(
    "data/telco_churn.csv"
)

churn_rate = (
    df["Churn"] == "Yes"
).mean()

assert 0.2 < churn_rate < 0.4