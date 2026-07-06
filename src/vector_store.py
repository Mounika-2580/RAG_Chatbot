"""FAISS index: build, persist, load, and search. Chunk metadata stored alongside."""
import json
import faiss
import numpy as np
from src.config import INDEX_DIR, INDEX_FILE, META_FILE
from src.embedder import embed


def build(chunks: list[dict]) -> None:
    """Embed chunk texts, build a cosine-similarity index, and save to disk."""
    vectors = embed([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])   # inner product on normalized vecs = cosine
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")


def load() -> tuple[faiss.Index, list[dict]]:
    """Load the persisted index and its chunk metadata."""
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("No index found. Run ingestion first (python -m src.ingest).")
    index = faiss.read_index(str(INDEX_FILE))
    chunks = json.loads(META_FILE.read_text(encoding="utf-8"))
    return index, chunks


def search(query: str, index: faiss.Index, chunks: list[dict], k: int) -> list[dict]:
    """Return the top-k most similar chunks, each with a similarity score."""
    q = embed([query])
    scores, ids = index.search(q, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:            # fewer chunks than k
            continue
        hit = dict(chunks[idx])
        hit["score"] = float(score)
        results.append(hit)
    return results
