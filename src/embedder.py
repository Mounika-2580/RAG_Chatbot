"""Local embeddings. Text is vectorized on-device; nothing is sent externally."""
import os

# Load the embedding model from the local HF cache only — never re-contact the
# network at query time (the model is cached during ingestion). Set before any
# huggingface import reads them.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the model once (lazy singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalized float32 vectors so inner product == cosine similarity."""
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")
