"""Offline pipeline: PDFs -> pages -> chunks -> embeddings -> saved FAISS index.

Run once (and again whenever the PDFs change):
    python -m src.ingest
"""
from src import pdf_loader, chunker, vector_store
from src.config import PDF_DIR


def run() -> None:
    print(f"Loading PDFs from {PDF_DIR} ...")
    pages = pdf_loader.load_all(PDF_DIR)
    if not pages:
        print("No PDF text found. Drop .pdf files into data/pdfs/ and retry.")
        return
    print(f"  {len(pages)} pages with text")

    chunks = chunker.chunk_records(pages)
    print(f"  {len(chunks)} chunks")

    print("Embedding + building index (first run downloads the embed model) ...")
    vector_store.build(chunks)
    print("Done. Index saved to data/index/.")


if __name__ == "__main__":
    run()
