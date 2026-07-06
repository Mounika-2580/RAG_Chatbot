"""Central configuration. All tunable settings live here."""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
INDEX_DIR = ROOT / "data" / "index"
INDEX_FILE = INDEX_DIR / "faiss.index"
META_FILE = INDEX_DIR / "chunks.json"

# --- Chunking ---
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # characters shared between adjacent chunks

# --- Embeddings (runs locally; document text never leaves the machine) ---
EMBED_MODEL = "all-MiniLM-L6-v2"

# --- Retrieval ---
TOP_K = 4               # chunks fed to Claude per question

# --- Answer generation ---
# "ollama"  -> fully local model, no API key (needs Ollama running)
# "claude"  -> Anthropic Claude (needs ANTHROPIC_API_KEY)
LLM_BACKEND = "ollama"
MAX_TOKENS = 1024

# Ollama (local) settings
OLLAMA_HOST = "http://127.0.0.1:11434"   # numeric IP: avoids flaky 'localhost' DNS resolution
OLLAMA_MODEL = "llama3.2"      # pulled via: ollama pull llama3.2

# Claude settings (used only when LLM_BACKEND == "claude")
CLAUDE_MODEL = "claude-opus-4-8"
