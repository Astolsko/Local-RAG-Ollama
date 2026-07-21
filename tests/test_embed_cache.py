"""Task 2.4: second embed of identical text hits the Redis cache (no re-embed)."""
import backend.embeddings as embeddings
import backend.cache.embed_cache as embed_cache


class FakePipe:
    def __init__(self, store):
        self.store = store
        self.ops = []

    def get(self, k):
        self.ops.append(("get", k))
        return self

    def setex(self, k, ttl, v):
        self.ops.append(("setex", k, v))
        return self

    def execute(self):
        results = []
        for op in self.ops:
            if op[0] == "get":
                results.append(self.store.get(op[1]))
            elif op[0] == "setex":
                self.store[op[1]] = op[2]
                results.append(True)
        self.ops = []
        return results


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.counters = {}

    def pipeline(self):
        return FakePipe(self.store)

    def incrby(self, k, n):
        self.counters[k] = self.counters.get(k, 0) + n


def test_second_embed_hits_cache(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(embed_cache, "get_redis", lambda: fake)

    calls = {"n": 0}

    def fake_uncached(texts):
        calls["n"] += len(texts)
        return [[float(len(t)), 1.0, 2.0] for t in texts]

    monkeypatch.setattr(embeddings, "_embed_uncached", fake_uncached)

    first = embeddings.embed_texts(["hello world", "foo"])
    assert calls["n"] == 2  # both were misses
    assert fake.counters.get("metrics:embed_cache_misses") == 2

    second = embeddings.embed_texts(["hello world", "foo"])
    assert calls["n"] == 2  # no new network embeds -> cache hit
    assert second == first
    assert fake.counters.get("metrics:embed_cache_hits") == 2

    # a new text still embeds, cached ones don't
    embeddings.embed_texts(["hello world", "brand new"])
    assert calls["n"] == 3
