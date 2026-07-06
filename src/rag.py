"""Query layer: retrieve relevant chunks, then have the LLM answer using only them."""
from dotenv import load_dotenv
from src import vector_store, llm
from src.config import TOP_K

load_dotenv()               # reads ANTHROPIC_API_KEY from .env (Claude backend only)

SYSTEM = (
    "You answer questions using ONLY the provided context from PDF documents. "
    "Cite the source and page for each fact, like (source.pdf, p.3). "
    "If the context does not contain the answer, say you don't know — do not guess."
)


def _format_context(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        blocks.append(f"[{h['source']}, p.{h['page']}]\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def answer(question: str, index, chunks) -> dict:
    """Retrieve top-k chunks and return {answer, sources}."""
    hits = vector_store.search(question, index, chunks, TOP_K)
    if not hits:
        return {"answer": "No indexed content to answer from.", "sources": []}

    context = _format_context(hits)
    prompt = f"Context:\n\n{context}\n\nQuestion: {question}"

    text = llm.generate(SYSTEM, prompt)
    sources = [{"source": h["source"], "page": h["page"], "score": h["score"]} for h in hits]
    return {"answer": text, "sources": sources}
