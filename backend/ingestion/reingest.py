import os
import json
import uuid
from pathlib import Path
from datetime import datetime
import sys

# Ensure backend and root directories are in python path
backend_dir = Path(__file__).resolve().parent.parent
root_dir = backend_dir.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import time
from backend.ingestion.chunker import StructureAwareChunker
from backend.retrieval.bm25_index import build_bm25_index
from backend.rag import _collection, OllamaEmbed
from backend.ingestion.contextualizer import Contextualizer
from backend import config

# Directory containing source documents – can be overridden via env var
SOURCE_DIR = Path(os.getenv("INGEST_SRC_DIR", "data/documents"))

def _load_existing_metrics(metric_path: Path) -> list:
    if metric_path.is_file():
        try:
            return json.loads(metric_path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _save_metrics(metric_path: Path, entry: dict):
    metrics = _load_existing_metrics(metric_path)
    metrics.append(entry)
    metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

def main():
    if not SOURCE_DIR.is_dir():
        raise RuntimeError(f"Source directory does not exist: {SOURCE_DIR}")

    start_time = time.time()
    chunker = StructureAwareChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    contextualizer = Contextualizer()
    embed_fn = OllamaEmbed()

    total_docs = 0
    total_chunks = 0
    total_chunk_len = 0

    for file_path in SOURCE_DIR.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            chunks = chunker.split_file(file_path)
        except ValueError:
            # Unsupported extension – skip
            continue
        if not chunks:
            continue
        total_docs += 1
        total_chunks += len(chunks)
        total_chunk_len += sum(len(c["text"]) for c in chunks)

        # Use document_id as source_id for backward compatibility
        source_id = chunks[0]["document_id"]
        # Delete any existing chunks for this document
        existing = _collection.get(where={"source_id": source_id}, include=[])
        if existing.get("ids"):
            _collection.delete(ids=existing["ids"])

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        
        # Read the original file text for context
        doc_text = file_path.read_text(encoding="utf-8")
        
        # Generate blurbs using Contextualizer
        blurbs = contextualizer.get_blurbs_batch(doc_text, documents)
        
        # Prepend blurbs to chunks before embedding
        annotated_chunks = [
            f"{blurb}\n\n{chunk}" if blurb else chunk
            for blurb, chunk in zip(blurbs, documents)
        ]
        
        # Compute embeddings for annotated chunks
        embeddings = embed_fn(annotated_chunks)

        metadatas = []
        for i, c in enumerate(chunks):
            metadatas.append({
                "source_id": source_id,
                "source_name": file_path.name,
                "chunk_index": c["chunk_index"],
                "page_number": 1,  # placeholder – not used in current UI
                "topic": c["section_title"],
                "source_file": c["source_file"],
                "created_at": c["created_at"],
                "doc_type": c["doc_type"],
                "context_blurb": blurbs[i]
            })
        _collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        # Rebuild BM25 index after adding/updating documents
        build_bm25_index()

    # Metrics
    elapsed_time = time.time() - start_time
    ingestion_time_per_doc = elapsed_time / total_docs if total_docs else 0.0

    if total_docs == 0:
        avg_chunks = 0
        avg_len = 0
    else:
        avg_chunks = total_chunks / total_docs
        avg_len = total_chunk_len / total_chunks if total_chunks else 0
    run_id = str(uuid.uuid4())
    metric_entry = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_documents": total_docs,
        "avg_chunks_per_doc": round(avg_chunks, 2),
        "avg_chunk_token_len": round(avg_len, 2),
        "ingestion_time_per_doc": round(ingestion_time_per_doc, 4)
    }
    metric_path = Path("data/ingest_metrics.json")
    _save_metrics(metric_path, metric_entry)
    print(json.dumps(metric_entry, indent=2))

    # Bump corpus version in Redis to invalidate semantic cache entries
    try:
        from redis_store import get_redis
        redis_client = get_redis()
        if redis_client:
            redis_client.incr("corpus:version")
            print("Successfully bumped corpus:version in Redis.")
    except Exception as e:
        print(f"Warning: could not bump corpus:version in Redis: {e}")

if __name__ == "__main__":
    main()
