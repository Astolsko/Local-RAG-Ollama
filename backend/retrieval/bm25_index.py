import os
import pickle
from pathlib import Path
from typing import List, Tuple, Any

import config
import chromadb
from chromadb.api.types import Documents, Embeddings
from rank_bm25 import BM25Okapi

# Independent Chroma client for BM25 indexing (avoids circular import)
_CLIENT = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
_COLLECTION = _CLIENT.get_or_create_collection(
    name="sources",
    embedding_function=None,  # not used for BM25
    metadata={"hnsw:space": "cosine"},
)


INDEX_PATH = config.CHROMA_DIR / "bm25_index.pkl"

import re

def tokenize(text: str) -> List[str]:
    """Helper to clean, lowercase, and tokenize text."""
    if not text:
        return []
    # Lowercase and match alphanumeric word characters
    return re.findall(r'\b\w+\b', text.lower())

def build_bm25_index() -> None:
    """Build and persist a BM25 index for all documents in the Chroma collection.
    The index file is stored alongside the Chroma DB directory.
    """
    rows = _COLLECTION.get(include=["documents", "metadatas"])
    docs: List[str] = rows["documents"]
    ids: List[str] = rows["ids"]
    metas: List[Any] = rows["metadatas"]
    if not docs:
        return
    tokenized_corpus = [tokenize(doc) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "docs": docs, "metas": metas}, f)

def _load_index() -> dict | None:
    if not INDEX_PATH.is_file():
        return None
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)

def search_bm25(query: str, top_k: int = 20) -> Tuple[List[str], List[str], List[Any]]:
    """Return top_k (ids, docs, metas) for the given query using BM25.
    If the index does not exist, returns empty lists.
    """
    index = _load_index()
    if not index:
        return [], [], []
    bm25: BM25Okapi = index["bm25"]
    ids: List[str] = index["ids"]
    docs: List[str] = index["docs"]
    metas: List[Any] = index["metas"]
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    top_ids = [ids[i] for i, _ in ranked]
    top_docs = [docs[i] for i, _ in ranked]
    top_metas = [metas[i] for i, _ in ranked]
    return top_ids, top_docs, top_metas
