import importlib


def _fresh_store(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    from backend.graph import store
    importlib.reload(store)
    store.init_db()
    return store


def test_upsert_dedup_relations_and_networkx(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, monkeypatch)

    a = store.upsert_entity("Orbis Robotics", "org")
    a2 = store.upsert_entity("orbis   robotics", "org")  # same after norm
    b = store.upsert_entity("Rotterdam", "location")
    assert a == a2 and a != b

    store.link_entity_chunk(a, "doc1:0", "doc1")
    store.link_entity_chunk(b, "doc1:0", "doc1")
    store.add_relation(a, b, "founded in", "doc1:0")

    c = store.counts()
    assert c["entities"] == 2 and c["relations"] == 1

    g = store.load_networkx()
    assert g.number_of_nodes() == 2
    assert g.has_edge(a, b)
    assert g[a][b]["predicate"] == "founded in"


def test_delete_doc_drops_orphans(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, monkeypatch)
    a = store.upsert_entity("A", "concept")
    b = store.upsert_entity("B", "concept")
    store.link_entity_chunk(a, "d1:0", "d1")
    store.link_entity_chunk(b, "d1:0", "d1")
    store.add_relation(a, b, "relates to", "d1:0")

    store.delete_doc("d1")
    c = store.counts()
    assert c["entities"] == 0 and c["relations"] == 0
    # cache was invalidated -> empty graph
    assert store.load_networkx().number_of_nodes() == 0
