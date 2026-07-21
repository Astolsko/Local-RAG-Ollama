"""Graph indexer / router / retrieval — all with mocked Ollama and Chroma (no heavy runs)."""
import importlib


def _fresh_store(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    from backend.graph import store
    importlib.reload(store)
    store.init_db()
    return store


VALID = '{"entities":[{"name":"Orbis","type":"org"},{"name":"Kestrel","type":"concept"}],' \
        '"relations":[{"source":"Orbis","target":"Kestrel","predicate":"supplies"}]}'


def test_index_document_populates_graph(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, monkeypatch)

    from backend.graph import extractor, indexer
    monkeypatch.setattr(extractor, "_generate_json", lambda p, m: VALID)

    from backend import rag
    monkeypatch.setattr(rag, "get_source", lambda doc_id: {
        "id": doc_id, "name": "d",
        "chunks": [{"chunk_index": 0, "text": "Orbis supplies Kestrel."},
                   {"chunk_index": 1, "text": "Orbis supplies Kestrel again."}],
    })

    indexer.index_document("docA", rebuild_communities=False)
    c = store.counts()
    assert c["entities"] == 2       # deduped across both chunks
    assert c["relations"] == 2      # one per chunk


def test_router_heuristics(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, monkeypatch)
    import config
    monkeypatch.setattr(config, "ROUTER_MODE", "auto")

    from backend.retrieval import router
    assert router.route("Summarize the themes across all documents") == "global"

    # two known entities present -> relational, no LLM call
    store.upsert_entity("Orbis", "org")
    store.upsert_entity("Kestrel", "concept")
    monkeypatch.setattr(router, "_llm_classify", lambda q: (_ for _ in ()).throw(AssertionError("LLM should not be called")))
    assert router.route("How is Orbis related to Kestrel?") == "relational"

    # plain factoid
    assert router.route("What is the budget?") == "factoid"

    # pinned mode wins
    monkeypatch.setattr(config, "ROUTER_MODE", "global")
    assert router.route("anything") == "global"


def test_relational_candidates(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, monkeypatch)
    a = store.upsert_entity("Orbis", "org")
    b = store.upsert_entity("Kestrel", "concept")
    store.link_entity_chunk(a, "docA:0", "docA")
    store.link_entity_chunk(b, "docA:1", "docA")
    store.add_relation(a, b, "supplies", "docA:0")

    from backend import rag
    class FakeColl:
        def get(self, ids, include):
            return {"ids": ids,
                    "documents": [f"text {i}" for i in ids],
                    "metadatas": [{"source_name": "d", "source_id": "docA", "chunk_index": 0} for _ in ids]}
    monkeypatch.setattr(rag, "_collection", FakeColl())

    from backend.graph import retrieval
    cands = retrieval.relational_candidates("Tell me about Orbis")
    ids = [cid for cid, _ in cands]
    assert "docA:0" in ids and "docA:1" in ids  # 1-hop neighbor's chunk included
