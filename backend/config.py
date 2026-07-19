import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# RAG_DATA_DIR lets the eval harness point the whole app (chroma, bm25 index, metrics
# db, settings) at an isolated directory so a run never touches the user's documents.
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR") or ROOT / "data")
CHROMA_DIR = DATA_DIR / "chroma"
CHAT_HISTORY_FILE = DATA_DIR / "chat_history.json"
SYSTEM_PROMPT_FILE = DATA_DIR / "system_prompt.txt"
SETTINGS_FILE = DATA_DIR / "settings.json"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful study and Q&A assistant. You answer the user's question using the provided context chunks first, citing the chunk IDs inline (e.g. [1], [2]) when referencing facts from them.\n\n"
    "Rules:\n"
    "- Each retrieved chunk is prefixed with its ID, like [Chunk 1], [Chunk 2], etc. Use these IDs for citations.\n"
    "- If the context contains the answer, base your response primarily on the context and cite it.\n"
    "- If the context does not contain the answer, you are encouraged to use your general knowledge to answer, connect topics, or provide prerequisite study knowledge. However, you must explicitly mention that the information is from general knowledge or outside the provided documents.\n"
    "- Keep answers concise, accurate, and educational."
)

DEFAULT_SETTINGS = {
    "OLLAMA_BASE": "http://localhost:11434",
    "EMBED_MODEL": "nomic-embed-text",
    "LLM_MODEL": "qwen2.5:0.5b",
    # auto|small|medium|large -- see backend/model_tiers.py. Empty per-role models
    # mean "use the tier's choice"; setting one explicitly overrides the tier.
    "MODEL_TIER": "auto",
    "REWRITE_MODEL": "",
    "EXTRACT_MODEL": "",
    "REDIS_URL": "redis://127.0.0.1:6379/0?protocol=2",
    "CHUNK_SIZE": 500,
    "CHUNK_OVERLAP": 50,
    "TOP_K": 20,
    "SESSION_TTL": 86400,
    "PROMPT_CACHE_TTL": 3600,
    "VECTOR_WEIGHT": 0.6,
    "BM25_WEIGHT": 0.4,
    "RERANK_ENABLED": True  ,
    "RERANK_MODEL": "BAAI/bge-reranker-base",
    "RERANK_TOP_K": 8,
    "ENABLE_QUERY_REWRITE": True,
    "QUERY_REWRITE_TIMEOUT_MS": 2000,
    "CACHE_SIMILARITY_THRESHOLD": 0.85,
    "RATE_LIMIT_PER_MINUTE": 60,
    # Keeps models resident in Ollama between requests instead of reloading them.
    "OLLAMA_KEEP_ALIVE": "10m"
}

def load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return DEFAULT_SETTINGS.copy()
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        merged = DEFAULT_SETTINGS.copy()
        for k, v in data.items():
            if k in merged:
                if isinstance(merged[k], bool):
                    merged[k] = bool(v)
                elif isinstance(merged[k], int):
                    merged[k] = int(v)
                elif isinstance(merged[k], float):
                    merged[k] = float(v)
                else:
                    merged[k] = str(v)
        return merged
    except Exception:
        return DEFAULT_SETTINGS.copy()

def save_settings(settings: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")

def _get_setting(key: str, default_val: any) -> any:
    # Check env var first, then settings.json
    env_val = os.getenv(key)
    if env_val is not None:
        if isinstance(default_val, bool):
            return env_val.lower() in ("true", "1", "yes")
        if isinstance(default_val, int):
            return int(env_val)
        if isinstance(default_val, float):
            return float(env_val)
        return env_val
        
    s = load_settings()
    return s.get(key, default_val)

def __getattr__(name: str) -> any:
    if name in DEFAULT_SETTINGS:
        return _get_setting(name, DEFAULT_SETTINGS[name])
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
