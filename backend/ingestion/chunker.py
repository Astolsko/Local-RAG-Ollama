import os
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import List, TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter

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

class StructureAwareChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Generic splitter used for plain text / pdf fallback
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
        """Split on headers first, then recursively split oversized sections.
        Supports both Markdown (e.g., ##) and HTML (<h1>-<h3>)."""
        # Simple header detection regexes
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
        # Now split each section using the recursive splitter
        chunks: List[str] = []
        for sec in sections:
            if not sec.strip():
                continue
            chunks.extend(self.recursive_splitter.split_text(sec))
        return chunks

    def _split_plain(self, text: str) -> List[str]:
        """Recursive character splitting for plain/text/pdf content."""
        return self.recursive_splitter.split_text(text)

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
                    # crude HTML header extraction
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
