"""visualize.py
Generate and save ROC comparison, confusion matrix and accuracy/F1 bar chart.
"""
import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix


def plot_roc_comparison(models: Dict[str, Any], X_tests, y_test, out_path: str):
    """Plot ROC curves for multiple models.

    X_tests may be a single array used for all models or a dict mapping model name -> X_for_that_model.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        try:
            X_for_model = X_tests[name] if isinstance(X_tests, dict) else X_tests
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_for_model)[:, 1]
            else:
                y_score = model.decision_function(X_for_model)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(best_model, X_test, y_test, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_accuracy_f1(metrics_df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = metrics_df.copy().reset_index()
    df_m = df.melt(id_vars=["model"], value_vars=["accuracy", "f1"], var_name="metric", value_name="score")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_m, x="model", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.title("Accuracy and F1 Comparison")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
