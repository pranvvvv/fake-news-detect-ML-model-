"""utils.py
Utility helpers: saving/loading, directory helpers, and seeding.
"""
import os
import joblib
import random
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_pickle(obj, path: str):
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)


def load_pickle(path: str):
    return joblib.load(path)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
