"""Split page text into overlapping character chunks, preserving source/page."""
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Sliding-window split. Overlap keeps context across chunk boundaries."""
    if size <= overlap:
        raise ValueError("CHUNK_SIZE must be greater than CHUNK_OVERLAP")
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def chunk_records(records: list[dict]) -> list[dict]:
    """Turn page records into chunk records ready for embedding."""
    out = []
    for rec in records:
        for piece in chunk_text(rec["text"], CHUNK_SIZE, CHUNK_OVERLAP):
            out.append({"source": rec["source"], "page": rec["page"], "text": piece})
    return out
