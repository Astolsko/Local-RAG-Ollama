"""Task 3.2: extraction is robust to malformed LLM output (mocked Ollama)."""
import importlib

from backend.graph import extractor


VALID = '{"entities":[{"name":"Orbis","type":"org"},{"name":"Rotterdam","type":"location"}],' \
        '"relations":[{"source":"Orbis","target":"Rotterdam","predicate":"founded in"}]}'


def test_valid_json(monkeypatch):
    monkeypatch.setattr(extractor, "_generate_json", lambda p, m: VALID)
    out = extractor.extract_from_chunk("Orbis was founded in Rotterdam.", model="x")
    assert [e.name for e in out.entities] == ["Orbis", "Rotterdam"]
    assert out.relations[0].predicate == "founded in"


def test_malformed_then_repaired(monkeypatch):
    calls = {"n": 0}

    def fake(prompt, model):
        calls["n"] += 1
        return "not json at all" if calls["n"] == 1 else VALID

    monkeypatch.setattr(extractor, "_generate_json", fake)
    out = extractor.extract_from_chunk("chunk", model="x")
    assert calls["n"] == 2  # one repair retry
    assert len(out.entities) == 2


def test_double_failure_returns_empty(monkeypatch):
    monkeypatch.setattr(extractor, "_generate_json", lambda p, m: "garbage")
    out = extractor.extract_from_chunk("chunk", model="x")
    assert out.entities == [] and out.relations == []  # no exception raised


def test_persist_writes_graph(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    from backend.graph import store
    importlib.reload(store)
    store.init_db()

    monkeypatch.setattr(extractor, "_generate_json", lambda p, m: VALID)
    out = extractor.extract_from_chunk("chunk", model="x")
    extractor.persist(out, chunk_id="d1:0", doc_id="d1")

    c = store.counts()
    assert c["entities"] == 2 and c["relations"] == 1
