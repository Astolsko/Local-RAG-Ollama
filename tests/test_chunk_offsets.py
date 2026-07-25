"""Task 2.1 — structure-aware chunking, char offsets, and the document store."""
from unittest.mock import MagicMock

MARKDOWN = """# Harbor Overview

The Meridian Harbor Authority operates three deep-water berths.
Each berth handles post-panamax vessels.

## Cargo Systems

The Kestrel automated cargo system moves containers between quay and yard.
It draws a total connected load of 4.2 megawatts.

## Risk Register

Silt accumulation is the highest-rated operational risk this year.
"""


def test_locate_chunks_maps_back_to_original_offsets():
    """Offsets must index the ORIGINAL text, even though chunking collapses whitespace."""
    from backend.ingestion.chunker import locate_chunks

    chunks = [
        "The Meridian Harbor Authority operates three deep-water berths. Each berth handles post-panamax vessels.",
        "It draws a total connected load of 4.2 megawatts.",
    ]
    spans = locate_chunks(MARKDOWN, chunks)

    assert all(s != (-1, -1) for s in spans), f"chunk not located: {spans}"
    # slicing the original with the span must recover the chunk, modulo whitespace
    for chunk, (start, end) in zip(chunks, spans):
        assert MARKDOWN[start:end].split() == chunk.split()
    # spans advance through the document
    assert spans[0][0] < spans[1][0]


def test_locate_chunks_reports_unlocatable_text():
    from backend.ingestion.chunker import locate_chunks

    spans = locate_chunks(MARKDOWN, ["this sentence is not in the document", ""])
    assert spans == [(-1, -1), (-1, -1)]


def test_locate_chunks_picks_the_occurrence_matching_chunk_order():
    """A verbatim repeat must resolve forward, not snap back to the first occurrence."""
    from backend.ingestion.chunker import locate_chunks

    text = "Alpha beta. Gamma delta. Alpha beta. Epsilon zeta."
    spans = locate_chunks(text, ["Gamma delta.", "Alpha beta."])
    assert spans[1][0] > spans[0][0]
    assert text[spans[1][0]:spans[1][1]] == "Alpha beta."


def test_ingest_routes_through_structure_aware_chunker(monkeypatch):
    """`rag.add_source` used to call semantic_chunk_text directly, so headings were
    ignored on every API ingest and StructureAwareChunker was only reachable from the
    batch reingest script."""
    import config
    monkeypatch.setattr(config, "CHUNK_SIZE", 500)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 50)
    # semantic splitting embeds sentences; keep it off Ollama by failing the embed, which
    # the chunker handles by falling back to character splitting.
    monkeypatch.setattr("backend.embeddings.embed_texts",
                        lambda ts: (_ for _ in ()).throw(RuntimeError("no ollama in tests")))

    from backend.rag import chunk_text_with_metadata
    chunks = chunk_text_with_metadata(MARKDOWN, "harbor.md")

    assert chunks, "expected chunks"
    # heading-derived topics survive into metadata
    topics = {c["topic"] for c in chunks}
    assert topics & {"Harbor Overview", "Cargo Systems", "Risk Register"}, topics
    # every chunk carries offsets, and located ones round-trip through the source text
    for c in chunks:
        assert "char_start" in c and "char_end" in c
        if c["char_start"] != -1:
            assert MARKDOWN[c["char_start"]:c["char_end"]].split() == c["text"].split()


def test_document_store_round_trip(monkeypatch, tmp_path):
    from backend import document_store
    monkeypatch.setattr(document_store, "DB_PATH", tmp_path / "documents.db")

    document_store.save_document("src-1", "harbor.md", MARKDOWN)
    got = document_store.get_document("src-1")
    assert got["name"] == "harbor.md"
    assert got["text"] == MARKDOWN

    # re-saving replaces rather than duplicating
    document_store.save_document("src-1", "harbor.md", "shorter")
    assert document_store.get_document("src-1")["text"] == "shorter"

    document_store.delete_document("src-1")
    assert document_store.get_document("src-1") is None


def test_get_document_text_falls_back_to_stored_chunks(monkeypatch, tmp_path):
    """Documents ingested before the store existed have no full text; their chunks are
    re-joined so reingest still has something to work from."""
    from backend import document_store
    import backend.rag as rag
    monkeypatch.setattr(document_store, "DB_PATH", tmp_path / "documents.db")

    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": ["old:0", "old:1"],
        "documents": ["first chunk text", "second chunk text"],
        "metadatas": [
            {"chunk_index": 0, "source_name": "legacy.md", "source_id": "old"},
            {"chunk_index": 1, "source_name": "legacy.md", "source_id": "old"},
        ],
    }
    monkeypatch.setattr(rag, "_collection", mock_collection)

    assert rag.get_document_text("old") == "first chunk text\n\nsecond chunk text"
