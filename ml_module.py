import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

def train_models(df, target, return_model=False):
    report_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target]

    label_encoders = {}

    # Encode categorical features
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Encode target if categorical
    if y.dtype == 'O' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y.astype(str))

    X = X.dropna()
    y = y[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(probability=True),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - start_time

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            }

            # Add ROC-AUC if probabilities are available and it's binary classification
            if y_proba is not None and len(np.unique(y)) == 2:
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

            results[name] = {
                "classification_report": classification_report(y_test, y_pred, output_dict=True),
                "metrics": metrics,
                "metadata": {
                    "training_time_seconds": round(elapsed, 2),
                    "features": X.shape[1],
                    "samples": len(X)
                }
            }

            trained_models[name] = model
        except Exception as e:
            results[name] = {"error": str(e)}

    return (results, trained_models) if return_model else results
