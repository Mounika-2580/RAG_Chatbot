"""Extract text from PDFs, one record per page (page number kept for citations)."""
from pathlib import Path
from pypdf import PdfReader


def load_pdf(path: Path) -> list[dict]:
    """Return a list of {source, page, text} for each non-empty page."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"source": path.name, "page": i, "text": text})
    return pages


def load_all(pdf_dir: Path) -> list[dict]:
    """Load every PDF in a directory."""
    records = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        records.extend(load_pdf(pdf))
    return records
