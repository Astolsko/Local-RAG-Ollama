import os
import uuid
import json
import re
import numpy as np
import httpx
from pathlib import Path
from datetime import datetime
from typing import List, TypedDict, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

# Define allowed extensions and their doc types
EXTENSION_MAP = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".txt": "plain",
    ".pdf": "pdf",
}

class Chunk(TypedDict):
    text: str
    source_file: str
    section_title: str
    chunk_index: int
    created_at: str
    doc_type: str
    document_id: str
    chunk_id: str

def split_into_sentences(text: str) -> List[str]:
    # ponytail: Splits text by sentence boundaries but avoids splitting on common abbreviations.
    sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
    sentences = sentence_end.split(text)
    return [s.strip() for s in sentences if s.strip()]

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    
    def _embed_one(text: str) -> List[float]:
        try:
            with httpx.Client(timeout=30) as client:
                r = client.post(
                    f"{config.OLLAMA_BASE}/api/embeddings",
                    json={"model": config.EMBED_MODEL, "prompt": text},
                )
                r.raise_for_status()
                return r.json()["embedding"]
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}") from e

    with ThreadPoolExecutor(max_workers=8) as executor:
        embeddings = list(executor.map(_embed_one, texts))
    return embeddings

def cosine_distance(v1: List[float], v2: List[float]) -> float:
    arr1 = np.array(v1)
    arr2 = np.array(v2)
    dot = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    return float(1.0 - (dot / (norm1 * norm2)))

def semantic_chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    # ponytail: semantic chunker splits into sentences, embeds them, and splits where semantic similarity drops.
    # Enforces chunk_size for oversized chunks.
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return [text] if text.strip() else []
        
    try:
        embeddings = embed_texts(sentences)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Semantic embedding failed, falling back to character chunking: {e}")
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)
        
    distances = []
    for i in range(len(embeddings) - 1):
        distances.append(cosine_distance(embeddings[i], embeddings[i+1]))
        
    if not distances:
        return [text]
        
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    # Threshold for splits
    threshold = mean_dist + 1.2 * std_dist
    
    chunks = []
    current_chunk_sentences = [sentences[0]]
    current_chunk_len = len(sentences[0])
    
    for i, dist in enumerate(distances):
        next_sentence = sentences[i+1]
        if dist > threshold or (current_chunk_len + len(next_sentence) > chunk_size):
            chunk_str = " ".join(current_chunk_sentences).strip()
            if chunk_str:
                chunks.append(chunk_str)
            current_chunk_sentences = [next_sentence]
            current_chunk_len = len(next_sentence)
        else:
            current_chunk_sentences.append(next_sentence)
            current_chunk_len += len(next_sentence) + 1
            
    if current_chunk_sentences:
        chunk_str = " ".join(current_chunk_sentences).strip()
        if chunk_str:
            chunks.append(chunk_str)
            
    # Post-process: recursively split chunks that exceed chunk_size
    final_chunks = []
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for chunk in chunks:
        if len(chunk) > chunk_size:
            final_chunks.extend(splitter.split_text(chunk))
        else:
            final_chunks.append(chunk)
            
    return final_chunks

class StructureAwareChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Generic splitter used as fallback or for final post-splitting of oversized chunks
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def _validate_extension(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext not in EXTENSION_MAP:
            raise ValueError(f"Unsupported file type: {ext}")
        return EXTENSION_MAP[ext]

    def _split_markdown_html(self, text: str) -> List[str]:
        """Split on headers first, then semantically split oversized sections."""
        import re
        header_regex = re.compile(r"^(#{1,3})\s+|<(h[1-3])[^>]*>", re.MULTILINE)
        sections = []
        last_idx = 0
        for m in header_regex.finditer(text):
            start = m.start()
            if start > last_idx:
                sections.append(text[last_idx:start])
            last_idx = start
        sections.append(text[last_idx:])
        
        chunks: List[str] = []
        for sec in sections:
            if not sec.strip():
                continue
            chunks.extend(semantic_chunk_text(sec, self.chunk_size, self.chunk_overlap))
        return chunks

    def _split_plain(self, text: str) -> List[str]:
        """Semantic splitting for plain/text/pdf content."""
        return semantic_chunk_text(text, self.chunk_size, self.chunk_overlap)

    def split_file(self, path: Path) -> List[Chunk]:
        """Read the file, detect its type, split into chunks, and attach metadata.
        Returns a list of Chunk dictionaries ready for ingestion.
        """
        doc_type = self._validate_extension(path)
        content = path.read_text(encoding="utf-8")
        if doc_type in ("markdown", "html"):
            raw_chunks = self._split_markdown_html(content)
        else:
            raw_chunks = self._split_plain(content)

        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())))
        created_at = datetime.utcnow().isoformat() + "Z"
        chunks: List[Chunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            # Derive a simple section title – first header line if present
            lines = chunk_text.splitlines()
            section_title = ""
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#"):
                    section_title = stripped.lstrip("#").strip()
                    break
                if stripped.lower().startswith("<h") and stripped.endswith(">"):
                    section_title = stripped
                    break
            chunk_id = f"{document_id}:{idx}"
            chunks.append({
                "text": chunk_text.strip(),
                "source_file": str(path),
                "section_title": section_title,
                "chunk_index": idx,
                "created_at": created_at,
                "doc_type": doc_type,
                "document_id": document_id,
                "chunk_id": chunk_id,
            })
        return chunks

