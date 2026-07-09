import time
import httpx
import config

# Heuristic: skip rewriting for very short queries (<=3 words)
def should_rewrite(query: str) -> bool:
    return len(query.split()) > 3

def _call_ollama(prompt: str) -> str:
    """Call Ollama's generate endpoint with the given prompt.
    Returns the raw response text or an empty string on failure.
    """
    try:
        timeout_sec = config.QUERY_REWRITE_TIMEOUT_MS / 1000 if hasattr(config, "QUERY_REWRITE_TIMEOUT_MS") else 2
        with httpx.Client(timeout=timeout_sec) as client:
            r = client.post(
                f"{config.OLLAMA_BASE}/api/generate",
                json={
                    "model": getattr(config, "LLM_MODEL", "qwen2.5:0.5b"),
                    "prompt": prompt,
                    "stream": False,
                },
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
    except Exception:
        return ""

# Simple metrics placeholders (can be integrated with a proper metrics system later)
query_rewrite_latency_ms = 0
num_queries_expanded = 0

def rewrite_query(raw_query: str):
    """Rewrite the raw query using the LLM and generate two alternate phrasings.
    Returns a list of queries: [original, rewritten, alt1, alt2].
    If the LLM call fails, the list will contain the original query repeated.
    """
    if not should_rewrite(raw_query):
        return [raw_query]

    system_prompt = (
        "You are a helpful assistant. Rewrite the following user query to be more search‑friendly "
        "and provide two alternative phrasings. Output exactly three lines:"
        "\n1) rewritten query"
        "\n2) first alternative"
        "\n3) second alternative."
    )
    prompt = f"{system_prompt}\nQuery: {raw_query}"
    start = time.monotonic()
    response = _call_ollama(prompt)
    latency = int((time.monotonic() - start) * 1000)

    global query_rewrite_latency_ms, num_queries_expanded
    query_rewrite_latency_ms = latency
    num_queries_expanded = 1

    # Parse response: first three non‑empty lines
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    if len(lines) >= 3:
        rewrites = lines[:3]
    else:
        rewrites = [raw_query] * 3

    return [raw_query] + rewrites
