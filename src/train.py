"""train.py
Train classical ML models on embeddings and a TF-IDF baseline for MultinomialNB.
"""
from typing import Dict, Any, Tuple
import os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV


def _make_pipeline_estimator(estimator, use_pca: bool = False, pca_components: int = 50):
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components, random_state=42)))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def train_embedding_models(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, output_dir: str, apply_pca: bool = False, pca_components: int = 50, quick: bool = False) -> Dict[str, Any]:
    """Train Logistic Regression and SVM (RBF) on embeddings with GridSearch.

    Args:
        X_train, y_train, X_val, y_val: arrays
        output_dir: directory to save models
        apply_pca: whether to apply PCA before classifier
        pca_components: PCA components

    Returns:
        dict of trained GridSearchCV objects
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Quick mode: train simple classifiers (or DummyClassifier if only one class present)
    if quick:
        from sklearn.dummy import DummyClassifier
        unique_labels = set(y_train.tolist())
        if len(unique_labels) < 2:
            clf_lr = DummyClassifier(strategy="most_frequent")
            clf_svc = DummyClassifier(strategy="most_frequent")
            clf_lr.fit(X_train, y_train)
            clf_svc.fit(X_train, y_train)
        else:
            clf_lr = LogisticRegression(max_iter=500, random_state=42)
            clf_lr.fit(X_train, y_train)
            clf_svc = SVC(kernel="rbf", probability=True, random_state=42)
            clf_svc.fit(X_train, y_train)

        results["logistic_regression"] = clf_lr
        results["svm_rbf"] = clf_svc
        joblib.dump(results, os.path.join(output_dir, "models_quick_emb.pkl"))
        return results

    # Logistic Regression (full mode with GridSearch)
    lr = LogisticRegression(max_iter=2000, random_state=42)
    pipe_lr = _make_pipeline_estimator(lr, use_pca=apply_pca, pca_components=pca_components)
    if quick:
        param_grid_lr = {"clf__C": [0.1, 1.0]}
        cv = 2
    else:
        param_grid_lr = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
        cv = 3
    gs_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=cv, n_jobs=-1, scoring="f1", verbose=0)
    gs_lr.fit(X_train, y_train)
    joblib.dump(gs_lr, os.path.join(output_dir, "logreg_grid.pkl"))
    results["logistic_regression"] = gs_lr

    # SVM RBF
    svc = SVC(kernel="rbf", probability=True, random_state=42)
    pipe_svc = _make_pipeline_estimator(svc, use_pca=apply_pca, pca_components=pca_components)
    if quick:
        param_grid_svc = {"clf__C": [0.1, 1.0], "clf__gamma": ["scale"]}
    else:
        param_grid_svc = {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", "auto"]}
    gs_svc = GridSearchCV(pipe_svc, param_grid_svc, cv=cv, n_jobs=-1, scoring="f1", verbose=0)
    gs_svc.fit(X_train, y_train)
    joblib.dump(gs_svc, os.path.join(output_dir, "svc_grid.pkl"))
    results["svm_rbf"] = gs_svc

    return results


def train_tfidf_nb(X_train_text, y_train, X_val_text, y_val, output_dir: str, quick: bool = False) -> Dict[str, Any]:
    """Train a TF-IDF + MultinomialNB baseline.

    Returns trained GridSearchCV object.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Quick mode: train a single estimator (or DummyClassifier pipeline if only one class present)
    from sklearn.dummy import DummyClassifier
    unique_labels = set(y_train.tolist())
    if quick:
        if len(unique_labels) < 2:
            pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=20000)), ("clf", DummyClassifier(strategy="most_frequent"))])
        else:
            pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=20000)), ("clf", MultinomialNB())])
        pipe.fit(X_train_text, y_train)
        joblib.dump(pipe, os.path.join(output_dir, "nb_tfidf_quick.pkl"))
        return {"nb_tfidf": pipe}

    pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=20000)), ("clf", MultinomialNB())])
    if quick:
        param_grid = {"clf__alpha": [0.5]}
        cv = 2
    else:
        param_grid = {"clf__alpha": [0.1, 0.5, 1.0]}
        cv = 3
    gs = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, scoring="f1", verbose=0)
    gs.fit(X_train_text, y_train)
    joblib.dump(gs, os.path.join(output_dir, "nb_tfidf_grid.pkl"))
    return {"nb_tfidf": gs}
