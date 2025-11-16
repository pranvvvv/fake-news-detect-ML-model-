"""preprocessing.py
Functions to load and clean news datasets and produce stratified splits.

This module provides:
- load_dataset(path): load CSV or folder with Fake.csv and True.csv
- stratified_splits(df, test_size, val_size): returns train/val/test splits
"""
from typing import Tuple
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


def _clean_text(text: str) -> str:
    """Basic text cleaning: remove URLs, HTML tags, extra whitespace.

    Args:
        text: raw text string

    Returns:
        cleaned text
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: str, text_col_candidates=None) -> pd.DataFrame:
    """Load dataset from a single CSV or from two files (Fake/True) in a folder.

    The returned DataFrame has columns: `text` and `label` (0 fake, 1 real).

    Args:
        path: path to CSV file or folder containing `Fake.csv` and `True.csv`.
        text_col_candidates: optional list of column names to try for text.

    Returns:
        pd.DataFrame
    """
    if text_col_candidates is None:
        text_col_candidates = ["text", "content", "article", "headline", "title"]

    if os.path.isdir(path):
        fake_path = os.path.join(path, "Fake.csv")
        true_path = os.path.join(path, "True.csv")
        if os.path.exists(fake_path) and os.path.exists(true_path):
            df_fake = pd.read_csv(fake_path)
            df_true = pd.read_csv(true_path)
            df_fake["label"] = 0
            df_true["label"] = 1
            df = pd.concat([df_fake, df_true], ignore_index=True)
        else:
            raise FileNotFoundError("Expected Fake.csv and True.csv inside the folder")
    else:
        df = pd.read_csv(path)
        # If there are separate True/Fake files encoded in columns, try to detect
        if "label" not in df.columns and ("truth" in df.columns or "is_fake" in df.columns):
            # try to normalize
            if "truth" in df.columns:
                df = df.rename(columns={"truth": "label"})
            elif "is_fake" in df.columns:
                df = df.rename(columns={"is_fake": "label"})

    # Find text column
    text_col = None
    for col in text_col_candidates:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        # fallback to first object column
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not obj_cols:
            raise ValueError("No text-like column found in dataset")
        text_col = obj_cols[0]

    # Normalize label
    if "label" not in df.columns:
        # Attempt heuristics: some datasets use 'label' values as 'FAKE'/'REAL'
        if any(df[text_col].str.contains("FAKE|Fake|fake", na=False)) and any(df[text_col].str.contains("REAL|Real|real", na=False)):
            raise ValueError("Dataset format ambiguous; please provide a dataset with explicit labels")
        else:
            raise ValueError("No `label` column found; provide dataset with label column or use Fake/True files")

    out = pd.DataFrame()
    out["text"] = df[text_col].fillna("").astype(str).apply(_clean_text)
    # Map label strings to 0/1
    labels = df["label"]
    if labels.dtype == object:
        labels = labels.str.lower().map({"fake": 0, "real": 1, "true": 1})
    out["label"] = labels.astype(int)
    return out


def stratified_splits(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create stratified train/val/test splits.

    Args:
        df: DataFrame with `text` and `label` columns
        test_size: proportion for test
        val_size: proportion for validation (of total)
        random_state: seed

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("DataFrame must contain `text` and `label` columns")

    X = df["text"]
    y = df["label"]

    # First attempt stratified splits; if dataset is too small for stratify, fall back to random splits
    try:
        X_rest, X_test, y_rest, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # Now split rest into train and val
        val_relative = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_rest, y_rest, test_size=val_relative, stratify=y_rest, random_state=random_state
        )
    except ValueError:
        # fallback: non-stratified but reproducible splits
        X_rest, X_test, y_rest, y_test = train_test_split(
            X, y, test_size=test_size, stratify=None, random_state=random_state
        )
        val_relative = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_rest, y_rest, test_size=val_relative, stratify=None, random_state=random_state
        )

    return X_train.reset_index(drop=True), X_val.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_val.reset_index(drop=True), y_test.reset_index(drop=True)
