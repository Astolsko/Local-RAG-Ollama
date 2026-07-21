"""Ollama embedding calls, batched.

`/api/embed` takes a list and embeds it in one request. Older Ollama builds only
expose `/api/embeddings`, which is one text per request; we try the batch
endpoint once and remember whether it worked.

This is the single place embeddings are produced -- both query embedding and
ingestion route through it.
"""
import logging

import httpx

import config

logger = logging.getLogger(__name__)

BATCH_SIZE = 64
_BATCH_SUPPORTED: bool | None = None


def _keep_alive() -> str:
    return getattr(config, "OLLAMA_KEEP_ALIVE", "10m")


def _embed_batch(client: httpx.Client, texts: list[str]) -> list[list[float]]:
    r = client.post(
        f"{config.OLLAMA_BASE}/api/embed",
        json={"model": config.EMBED_MODEL, "input": texts, "keep_alive": _keep_alive()},
    )
    r.raise_for_status()
    out = r.json().get("embeddings")
    if not out or len(out) != len(texts):
        raise ValueError(f"/api/embed returned {len(out or [])} embeddings for {len(texts)} inputs")
    return out


def _embed_individually(client: httpx.Client, texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
        r = client.post(
            f"{config.OLLAMA_BASE}/api/embeddings",
            json={"model": config.EMBED_MODEL, "prompt": text, "keep_alive": _keep_alive()},
        )
        r.raise_for_status()
        out.append(r.json()["embedding"])
    return out


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts, preserving input order. Cached in Redis by (model, text)."""
    if not texts:
        return []

    from backend.cache.embed_cache import get_cached, set_cached, record

    model = config.EMBED_MODEL
    cached = get_cached(model, texts)
    miss_idx = [i for i, c in enumerate(cached) if c is None]
    record(hits=len(texts) - len(miss_idx), misses=len(miss_idx))

    if miss_idx:
        miss_texts = [texts[i] for i in miss_idx]
        fresh = _embed_uncached(miss_texts)
        set_cached(model, list(zip(miss_texts, fresh)))
        for j, i in enumerate(miss_idx):
            cached[i] = fresh[j]

    return cached  # type: ignore[return-value]  # all Nones filled above


def _embed_uncached(texts: list[str]) -> list[list[float]]:
    """Embed texts via Ollama, no cache. Preserves input order."""
    global _BATCH_SUPPORTED
    if not texts:
        return []

    with httpx.Client(timeout=300) as client:
        if _BATCH_SUPPORTED is not False:
            try:
                out: list[list[float]] = []
                for i in range(0, len(texts), BATCH_SIZE):
                    out.extend(_embed_batch(client, texts[i:i + BATCH_SIZE]))
                _BATCH_SUPPORTED = True
                return out
            except Exception as e:
                if _BATCH_SUPPORTED:
                    raise  # batching worked before, so this is a real failure
                logger.warning(f"/api/embed unavailable ({e}); using /api/embeddings")
                _BATCH_SUPPORTED = False

        return _embed_individually(client, texts)
