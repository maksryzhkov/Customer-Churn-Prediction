from pipeline import load_data, train_model
from sklearn.metrics import roc_auc_score


def test_dataset_not_empty():
    df = load_data(
        "data/Telco-Customer-Churn.csv"
    )

    assert len(df) > 7000


def test_target_exists():
    df = load_data(
        "data/Telco-Customer-Churn.csv"
    )

    assert "Churn" in df.columns


def test_target_distribution():
    df = load_data(
        "data/Telco-Customer-Churn.csv"
    )

    churn_rate = (
        df["Churn"] == "Yes"
    ).mean()

    assert 0.2 < churn_rate < 0.4


def test_model_auc():
    model, X_test, y_test = train_model()

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    assert auc > 0.75