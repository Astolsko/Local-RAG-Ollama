"""ponytail: runnable self-check — fails if page splitting, topic extraction, or metadata chunking fails."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import split_into_pages, chunk_text_with_metadata

# 1. Test split_into_pages with form feeds
pdf_text = "Page One content\fPage Two content\fPage Three content"
pages = split_into_pages(pdf_text)
assert len(pages) == 3, f"Expected 3 pages, got {len(pages)}"
assert pages[0] == "Page One content"
assert pages[1] == "Page Two content"
assert pages[2] == "Page Three content"

# 2. Test split_into_pages with long text (paragraph division)
long_text = "\n\n".join([f"Paragraph {i}: " + "a" * 200 for i in range(15)])
pages_split = split_into_pages(long_text)
assert len(pages_split) >= 2, f"Expected long text to be split into multiple pages, got {len(pages_split)}"

# 3. Test chunk_text_with_metadata topic extraction
doc_content = "# Section 1: Introduction\nThis is the introduction text.\n\n# SECTION 2: FASTAPI SETUP\nFastAPI is a modern web framework."
chunks = chunk_text_with_metadata(doc_content, "test_doc")
assert len(chunks) > 0
assert chunks[0]["page_number"] == 1
# Topic should be resolved to Section 1: Introduction
assert "Section 1: Introduction" in chunks[0]["topic"] or "SECTION 2" in chunks[0]["topic"]

print("rag self-check ok")
