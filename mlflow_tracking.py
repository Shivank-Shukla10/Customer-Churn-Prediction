import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# --------------------------------------------------
# MLflow configuration
# --------------------------------------------------

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Customer-Churn-Prediction")



# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv("data/telco_customer_churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

df_model = df.drop(columns=["customerID"])


# --------------------------------------------------
# Split data
# --------------------------------------------------

X = df_model.drop(columns=["Churn"])
y = df_model["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_features
        )
    ]
)


# --------------------------------------------------
# Model
# --------------------------------------------------

scale_pos_weight = (
    len(y_train[y_train == 0])
    / len(y_train[y_train == 1])
)

model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# --------------------------------------------------
# Train + MLflow tracking
# --------------------------------------------------

with mlflow.start_run(run_name="Tuned-XGBoost"):

    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Final threshold obtained earlier
    threshold = 0.52529

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
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

    # --------------------------------------------------
    # Log parameters
    # --------------------------------------------------

    mlflow.log_params({
        "model": "XGBoost",
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "classification_threshold": threshold
    })

    # --------------------------------------------------
    # Log metrics
    # --------------------------------------------------

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })

    # --------------------------------------------------
    # Log model
    # --------------------------------------------------

    mlflow.sklearn.log_model(
        pipeline,
        "customer_churn_model",
        skops_trusted_types=[
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier"
        ]
    )

    print("\nMLflow Run Completed")
    print("----------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")