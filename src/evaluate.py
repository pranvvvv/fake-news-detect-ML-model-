"""evaluate.py
Evaluation helpers: compute metrics, assemble metrics table and save results.
"""
from typing import Dict, Any
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def _safe_proba(model, X):
    """Return probability for the positive class when possible, otherwise decision function.
    Fallbacks handled to compute ROC-AUC when feasible.
    """
    try:
        proba = model.predict_proba(X)
        # proba shape (n_samples, n_classes)
        if proba.shape[1] == 2:
            return proba[:, 1]
        # if multiclass but binary-encoded differently, try first column
        return proba[:, -1]
    except Exception:
        try:
            # decision_function might return shape (n_samples,) or (n_samples, n_classes)
            df = model.decision_function(X)
            if isinstance(df, np.ndarray) and df.ndim > 1:
                return df[:, -1]
            return df
        except Exception:
            return None


def evaluate_models(models: Dict[str, Any], X_test, y_test) -> pd.DataFrame:
    """Evaluate a dict of fitted estimators on test data.

    Args:
        models: mapping name->estimator
        X_test: features or texts depending on model
        y_test: ground truth labels

    Returns:
        DataFrame with metrics and index as model names
    """
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        proba = _safe_proba(model, X_test)
        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        if proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
            except Exception:
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("model")
    return df


def save_metrics_table(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path)
