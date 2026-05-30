import os
import time
import yaml
import joblib

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_data(path):
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    return df


def build_preprocessor(df, target):

    X = df.drop(columns=[target])

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


def train_model():
    config = load_config()

    start_time = time.time()

    df = load_data(
        config["data_path"]
    )

    target = config["target_column"]

    X = df.drop(columns=[target])

    y = (
        df[target]
        .map({"No": 0, "Yes": 1})
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=config["test_size"],
            random_state=config["random_state"],
            stratify=y
        )
    )

    preprocessor = build_preprocessor(
        df,
        target
    )

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["random_state"],
        class_weight="balanced"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    # mlflow.set_experiment(
    #     "rostelecom_churn_prediction"
    # )

    # with mlflow.start_run():
    #
    #     pipeline.fit(
    #         X_train,
    #         y_train
    #     )
    #
    #     y_pred = pipeline.predict(
    #         X_test
    #     )
    #
    #     y_prob = pipeline.predict_proba(
    #         X_test
    #     )[:, 1]
    #
    #     accuracy = accuracy_score(
    #         y_test,
    #         y_pred
    #     )
    #
    #     f1 = f1_score(
    #         y_test,
    #         y_pred
    #     )
    #
    #     roc_auc = roc_auc_score(
    #         y_test,
    #         y_prob
    #     )
    #
    #     mlflow.log_metric(
    #         "accuracy",
    #         accuracy
    #     )
    #
    #     mlflow.log_metric(
    #         "f1",
    #         f1
    #     )
    #
    #     mlflow.log_metric(
    #         "roc_auc",
    #         roc_auc
    #     )
    #
    #     mlflow.log_metric(
    #         "cpu_percent",
    #         psutil.cpu_percent()
    #     )
    #
    #     mlflow.log_metric(
    #         "memory_percent",
    #         psutil.virtual_memory().percent
    #     )
    #
    #     mlflow.sklearn.log_model(
    #         pipeline,
    #         "churn_model"
    #     )
    #
    #     duration = (
    #         time.time() - start_time
    #     )
    #
    #     mlflow.log_metric(
    #         "pipeline_duration_sec",
    #         duration
    #     )

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        "models/churn_model.pkl"
    )

    print(
        "Модель сохранена: models/churn_model.pkl"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print(
        f"Accuracy={accuracy:.3f}"
    )

    print(
        f"F1={f1:.3f}"
    )

    print(
        f"ROC-AUC={roc_auc:.3f}"
    )

    return (
        pipeline,
        X_test,
        y_test
    )


if __name__ == "__main__":
    train_model()