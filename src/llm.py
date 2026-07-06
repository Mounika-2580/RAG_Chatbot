"""Answer generation. Dispatches to a local Ollama model or to Claude."""
import requests
from src.config import (
    LLM_BACKEND, MAX_TOKENS,
    OLLAMA_HOST, OLLAMA_MODEL,
    CLAUDE_MODEL,
)

_claude = None

# Local session that ignores any corporate HTTP(S)_PROXY env vars — Ollama runs
# on localhost and must not be routed through a proxy.
_local = requests.Session()
_local.trust_env = False


def _ollama(system: str, prompt: str) -> str:
    """Call a local Ollama model via its REST API. No API key, fully offline."""
    try:
        resp = _local.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=120,
        )
    except requests.ConnectionError:
        return ("[Ollama not reachable. Start it and pull the model:\n"
                f"    ollama pull {OLLAMA_MODEL}\n"
                "    ollama serve  (usually already running)]")
    if resp.status_code == 404:
        return f"[Model '{OLLAMA_MODEL}' not found. Run:  ollama pull {OLLAMA_MODEL}]"
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _claude_generate(system: str, prompt: str) -> str:
    """Call Anthropic Claude (requires ANTHROPIC_API_KEY)."""
    global _claude
    if _claude is None:
        import anthropic
        _claude = anthropic.Anthropic()
    resp = _claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "")


def generate(system: str, prompt: str) -> str:
    """Generate an answer using the configured backend."""
    if LLM_BACKEND == "ollama":
        return _ollama(system, prompt)
    if LLM_BACKEND == "claude":
        return _claude_generate(system, prompt)
    raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")
