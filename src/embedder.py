"""embedder.py
Create and save sentence embeddings using sentence-transformers.
"""
from typing import Iterable, Optional
import os
import joblib
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def encode_texts(texts: Iterable[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64, device: Optional[str] = None, quick: bool = False) -> np.ndarray:
    """Encode an iterable of texts into sentence embeddings.

    Args:
        texts: iterable of strings
        model_name: sentence-transformers model name
        batch_size: encoding batch size
        device: device string like 'cpu' or 'cuda' (optional)

    Returns:
        ndarray of shape (n_samples, embedding_dim)
    """
    if quick:
        # deterministic pseudo-embeddings for fast smoke tests
        import numpy as _np
        _np.random.seed(42)
        texts_list = list(texts)
        n = len(texts_list)
        dim = 384
        return _np.random.RandomState(42).randn(n, dim).astype(_np.float32)

    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not available in the environment. Install it or run with quick=True")

    model = SentenceTransformer(model_name, device=device)
    # sentence-transformers supports batching internally
    embeddings = model.encode(list(texts), batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def save_embeddings(path: str, embeddings: np.ndarray, labels=None):
    """Save embeddings (and optional labels) with joblib.

    Args:
        path: file path to save (.pkl suggested)
        embeddings: numpy array
        labels: optional labels array/series
    """
    payload = {"embeddings": embeddings, "labels": labels}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(payload, path)


def load_embeddings(path: str):
    """Load embeddings saved by `save_embeddings`.
    Returns dict with keys `embeddings` and `labels`.
    """
    return joblib.load(path)
