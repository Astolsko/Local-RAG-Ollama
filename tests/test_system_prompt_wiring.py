"""The user-editable system prompt must actually reach the main chat path.

`chat_stream` hardcoded `prompt_templates.SYSTEM_PROMPT_TEMPLATE` while
`PUT /api/settings/system-prompt` wrote `data/system_prompt.txt`, which only the legacy
`POST /api/ask` ever read — so editing the prompt in the UI changed nothing for
`/api/chats/ask`. This test drives the real generator and inspects the payload sent to
Ollama.
"""
import asyncio
import json


class _FakeStream:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeAsyncClient:
    """Captures the JSON body of the chat request instead of calling Ollama."""

    def __init__(self, captured, **_kw):
        self._captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, _method, _url, json=None, **_kw):
        self._captured.append(json)
        return _FakeStream([
            '{"message": {"content": "The load is 4.2 MW [1]."}}',
            '{"prompt_eval_count": 42, "eval_count": 9, "done": true}',
        ])


def test_edited_system_prompt_reaches_the_chat_request(monkeypatch, tmp_path):
    # driven with asyncio.run rather than pulling in pytest-asyncio for one test
    import config
    import backend.chat_stream as cs
    import backend.rag as rag
    from unittest.mock import MagicMock

    custom_prompt = "SENTINEL PROMPT: answer only from the context and cite every claim."
    prompt_file = tmp_path / "system_prompt.txt"
    prompt_file.write_text(custom_prompt, encoding="utf-8")
    monkeypatch.setattr(config, "SYSTEM_PROMPT_FILE", prompt_file)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    # Redis-free: no cached prompt, no session, no cache writes.
    monkeypatch.setattr(rag, "cache_system_prompt", lambda _t: None)
    monkeypatch.setattr(rag, "get_cached_system_prompt", lambda: None)
    monkeypatch.setattr(cs, "get_session_messages", lambda _s: [])
    monkeypatch.setattr("backend.cache.semantic_cache.check_semantic_cache", lambda *a: None)
    monkeypatch.setattr("backend.cache.semantic_cache.set_semantic_cache", lambda *a: None)

    mock_collection = MagicMock()
    mock_collection.count.return_value = 3
    monkeypatch.setattr(rag, "_collection", mock_collection)

    class _FakeEmbed:
        def __call__(self, texts):
            return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(rag, "OllamaEmbed", _FakeEmbed)
    monkeypatch.setattr(
        cs, "hybrid_search_sync",
        lambda *a, **kw: (
            ["The Kestrel system draws 4.2 megawatts."],
            [{"source_name": "harbor.md", "source_id": "s1", "chunk_index": 0,
              "rerank_score": 0.91}],
            ["s1:0"],
            {"rewritten_queries": ["q"], "rewrite_skipped": True, "retrieve_latency": 0.0,
             "rerank_latency": 0.0, "rerank_scores": [0.91], "route": "vector"},
        ),
    )

    async def _no_judge(*a, **kw):
        return None

    monkeypatch.setattr(cs, "run_background_judge", _no_judge)

    captured: list = []
    monkeypatch.setattr(cs.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(captured, **kw))

    async def _drive():
        return [json.loads(chunk) async for chunk in
                cs.chat_stream("What is the connected load?", session_id=None, source_ids=["s1"])]

    events = asyncio.run(_drive())

    assert captured, "no request was sent to Ollama"
    system_messages = [m["content"] for m in captured[0]["messages"] if m["role"] == "system"]
    assert system_messages == [custom_prompt], (
        f"chat used a different system prompt than the one the user edited: {system_messages}"
    )

    # confidence must come from the rerank score, not the old flat 0.85 fallback
    final = [e for e in events if "confidence" in e][-1]
    assert final["confidence"] == 0.91
