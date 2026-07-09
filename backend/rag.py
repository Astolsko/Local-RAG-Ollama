import uuid
import json
from typing import Any, TypedDict
from datetime import datetime

import chromadb
import httpx
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

import time
import logging
import config
from backend.ingestion.contextualizer import Contextualizer
from backend.retrieval.hybrid_search import hybrid_search_sync
from redis_store import (
    append_session_turn,
    cache_system_prompt,
    get_cached_rag,
    get_cached_system_prompt,
    get_session_messages,
    set_cached_rag,
)


class OllamaEmbed(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        out: list[list[float]] = []
        with httpx.Client(timeout=120) as client:
            for text in input:
                r = client.post(
                    f"{config.OLLAMA_BASE}/api/embeddings",
                    json={"model": config.EMBED_MODEL, "prompt": text},
                )
                r.raise_for_status()
                out.append(r.json()["embedding"])
        return out


_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
_collection = _client.get_or_create_collection(
    name="sources",
    embedding_function=OllamaEmbed(),
    metadata={"hnsw:space": "cosine"},
)

def smart_chunk_page(page_content: str) -> list[str]:
    # ponytail: splits by paragraphs to respect semantic boundaries, falls back to char slice
    paragraphs = page_content.split("\n\n")
    chunks = []
    current_chunk = []
    current_len = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) > config.CHUNK_SIZE:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            if len(para) > config.CHUNK_SIZE:
                # split large paragraph
                start = 0
                while start < len(para):
                    end = min(start + config.CHUNK_SIZE, len(para))
                    piece = para[start:end].strip()
                    if piece:
                        chunks.append(piece)
                    if end >= len(para):
                        break
                    start = end - config.CHUNK_OVERLAP
                continue
        current_chunk.append(para)
        current_len += len(para)
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


def chunk_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + config.CHUNK_SIZE, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - config.CHUNK_OVERLAP
    return chunks


def split_into_pages(text: str) -> list[str]:
    if "\f" in text:
        return text.split("\f")
    
    pages = []
    paragraphs = text.split("\n\n")
    current_page = []
    current_len = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        current_page.append(para)
        current_len += len(para)
        if current_len >= 2000:
            pages.append("\n\n".join(current_page))
            current_page = []
            current_len = 0
    if current_page:
        pages.append("\n\n".join(current_page))
    
    if not pages:
        return [text]
    return pages


def chunk_text_with_metadata(text: str, source_name: str) -> list[dict[str, Any]]:
    raw_pages = split_into_pages(text)
    chunks_meta = []
    current_topic = "General Overview"
    
    for page_idx, page_content in enumerate(raw_pages):
        page_num = page_idx + 1
        page_chunks = smart_chunk_page(page_content)
        
        for chunk in page_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
                
            chunk_lines = chunk.split("\n")
            for line in chunk_lines:
                line_stripped = line.strip()
                if line_stripped.startswith("#"):
                    current_topic = line_stripped.lstrip("#").strip()
                    break
                elif line_stripped.isupper() and 0 < len(line_stripped) <= 60 and not line_stripped.isdigit():
                    current_topic = line_stripped
                    break
            
            topic_to_use = current_topic
            if topic_to_use == "General Overview" and chunk_lines:
                first_line = chunk_lines[0].strip()
                if 0 < len(first_line) <= 60 and not first_line.startswith(("[", "]", "(", ")")):
                    topic_to_use = first_line

            chunks_meta.append({
                "text": chunk,
                "page_number": page_num,
                "topic": topic_to_use
            })
            
    return chunks_meta


def add_source(name: str, text: str) -> dict[str, Any]:
    start_time = time.time()
    chunks_meta = chunk_text_with_metadata(text, name)
    if not chunks_meta:
        raise ValueError("Source text is empty")

    # Generate context blurbs using the local LLM via Contextualizer
    contextualizer = Contextualizer()
    chunks = [c["text"] for c in chunks_meta]
    blurbs = contextualizer.get_blurbs_batch(text, chunks)

    # Prepend blurbs to chunks before embedding
    annotated_chunks = [
        f"{blurb}\n\n{chunk}" if blurb else chunk
        for blurb, chunk in zip(blurbs, chunks)
    ]

    # Pre-compute embeddings for annotated chunks
    embed_fn = OllamaEmbed()
    embeddings = embed_fn(annotated_chunks)

    source_id = str(uuid.uuid4())
    ids = [f"{source_id}:{i}" for i in range(len(chunks_meta))]
    metadatas = [
        {
            "source_id": source_id,
            "source_name": name,
            "chunk_index": i,
            "page_number": c["page_number"],
            "topic": c["topic"],
            "context_blurb": blurbs[i]
        }
        for i, c in enumerate(chunks_meta)
    ]

    # Store the original chunks in the collection but with the annotated embeddings
    _collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    elapsed = time.time() - start_time
    logging.getLogger(__name__).info(
        f"Ingested source '{name}' ({len(chunks)} chunks) in {elapsed:.4f} seconds."
    )

    return {
        "id": source_id,
        "name": name,
        "chunks": len(chunks),
        "ingestion_time": round(elapsed, 4)
    }


def get_source(source_id: str) -> dict[str, Any] | None:
    rows = _collection.get(where={"source_id": source_id}, include=["documents", "metadatas"])
    ids = rows["ids"]
    if not ids:
        return None
    
    chunks_with_meta = []
    for doc, meta in zip(rows["documents"], rows["metadatas"]):
        chunks_with_meta.append({
            "chunk_index": meta["chunk_index"],
            "page_number": meta.get("page_number", 1),
            "topic": meta.get("topic", "General Context"),
            "text": doc
        })
    chunks_with_meta.sort(key=lambda x: x["chunk_index"])
    
    name = rows["metadatas"][0]["source_name"] if rows["metadatas"] else "Unknown Source"
    
    return {
        "id": source_id,
        "name": name,
        "chunks": chunks_with_meta
    }


def list_sources() -> list[dict[str, Any]]:
    all_meta = _collection.get(include=["metadatas"])["metadatas"] or []
    seen: dict[str, dict[str, Any]] = {}
    for meta in all_meta:
        sid = meta["source_id"]
        if sid not in seen:
            seen[sid] = {"id": sid, "name": meta["source_name"], "chunks": 0}
        seen[sid]["chunks"] += 1
    return sorted(seen.values(), key=lambda s: s["name"].lower())


def delete_source(source_id: str) -> bool:
    rows = _collection.get(where={"source_id": source_id}, include=[])
    ids = rows["ids"]
    if not ids:
        return False
    _collection.delete(ids=ids)
    return True


def get_system_prompt() -> str:
    if config.SYSTEM_PROMPT_FILE.exists():
        try:
            return config.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return config.DEFAULT_SYSTEM_PROMPT


def set_system_prompt(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("System prompt cannot be empty")
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.SYSTEM_PROMPT_FILE.write_text(text, encoding="utf-8")
        cache_system_prompt(text)
    except Exception as e:
        raise ValueError(f"Failed to persist system prompt: {e}") from e
    return text


def _ensure_system_prompt_cached() -> str:
    prompt = get_system_prompt()
    cached = get_cached_system_prompt()
    if cached != prompt:
        cache_system_prompt(prompt)
    return prompt


def _retrieve_context(question: str, system_prompt: str, source_ids: list[str] | None = None) -> tuple[str, list[dict[str, Any]], bool]:
    if _collection.count() == 0:
        raise ValueError("No sources indexed yet")

    if source_ids is not None and len(source_ids) == 0:
        raise ValueError("No sources selected. Please check at least one source in the right panel.")

    source_count = _collection.count()
    cached = get_cached_rag(question, source_count, system_prompt, source_ids)
    if cached:
        return cached["context"], cached["citations"], True

    where_clause = None
    if source_ids is not None:
        if len(source_ids) == 1:
            where_clause = {"source_id": source_ids[0]}
        else:
            where_clause = {"source_id": {"$in": source_ids}}

    # Hybrid search: combine vector and BM25 using Reciprocal Rank Fusion
    docs, metas, _, _ = hybrid_search_sync(question, top_k=min(config.TOP_K, source_count))
    if not docs:
        raise ValueError("No relevant context found in the selected sources")
    context = "\n\n".join(f"[{m['source_name']}] {d}" for d, m in zip(docs, metas))
    citations = [
        {
            "source_id": m.get("source_id", ""),
            "source_name": m["source_name"],
            "page_number": m.get("page_number", 1),
            "topic": m.get("topic", "General Context"),
            "chunk_index": m.get("chunk_index", 0)
        }
        for m in metas
    ]
    set_cached_rag(question, source_count, system_prompt, {"context": context, "citations": citations}, source_ids)
    return context, citations, False


def is_conversational_greeting(question: str) -> bool:
    q = question.strip().lower().rstrip("?.!")
    greetings = {"hi", "hello", "hey", "yo", "hola", "greetings", "good morning", "good afternoon", "good evening"}
    return q in greetings or q.startswith(("hi ", "hello ", "hey "))


def ask(question: str, session_id: str | None = None, source_ids: list[str] | None = None) -> dict[str, Any]:
    import uuid
    import time
    
    request_id = str(uuid.uuid4())
    t_start = time.perf_counter()
    embed_latency = 0.0
    retrieve_latency = 0.0
    rerank_latency = 0.0
    generate_latency = 0.0
    cache_hit_flag = False
    rewritten_query = None
    retrieved_ids = []
    rerank_scores = []
    prompt_tokens = 0
    response_tokens = 0
    cited_citations = []
    answer = ""

    question = question.strip()
    if not question:
        raise ValueError("Question is empty")

    try:
        from backend.rag import _collection
        is_rag = source_ids is not None and len(source_ids) > 0 and not is_conversational_greeting(question)

        query_embeddings = None
        if is_rag:
            # Perform query embedding first (reused for caching and retrieval)
            t_embed_start = time.perf_counter()
            embed_fn = OllamaEmbed()
            query_embeddings = embed_fn([question])
            embed_latency = time.perf_counter() - t_embed_start
            
            # Check semantic cache
            from backend.cache.semantic_cache import check_semantic_cache
            cache_hit = check_semantic_cache(question, query_embeddings[0])
            if cache_hit:
                cache_hit_flag = True
                cited_citations = cache_hit["citations"]
                retrieved_ids = [c["chunk_id"] for c in cited_citations]
                if session_id:
                    append_session_turn(session_id, question, cache_hit["answer_text"], cache_hit["citations"])
                return {
                    "answer_text": cache_hit["answer_text"],
                    "answer": cache_hit["answer_text"],  # legacy compatibility
                    "citations": cache_hit["citations"],
                    "confidence": cache_hit["confidence"],
                    "cached": True,
                    "session_id": session_id,
                    "prompt_tokens": 0,
                    "response_tokens": 0,
                    "request_id": request_id
                }
                
            # Perform hybrid search retrieval
            docs, metas, ids, search_metrics = hybrid_search_sync(question, top_k=min(config.TOP_K, _collection.count()))
            retrieve_latency = search_metrics["retrieve_latency"]
            rerank_latency = search_metrics["rerank_latency"]
            rerank_scores = search_metrics["rerank_scores"]
            retrieved_ids = ids
            
            if search_metrics["rewritten_queries"] and len(search_metrics["rewritten_queries"]) > 1:
                rewritten_query = ", ".join(search_metrics["rewritten_queries"][1:])
            
            # Prepare structured context and citation candidates
            context_parts = []
            initial_citations = []
            for idx, (doc, meta, chunk_id) in enumerate(zip(docs, metas, ids)):
                context_parts.append(f"[Chunk {idx + 1}] (Source: {meta['source_name']})\n{doc}")
                initial_citations.append({
                    "chunk_id": chunk_id,
                    "source_file": meta.get("source_file") or meta.get("source_name", "Unknown"),
                    "snippet": doc,
                    "source_id": meta.get("source_id", ""),
                    "source_name": meta["source_name"],
                    "page_number": meta.get("page_number", 1),
                    "topic": meta.get("topic", "General Context"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "rerank_score": meta.get("rerank_score")
                })
            context = "\n\n".join(context_parts)
            system_prompt = _ensure_system_prompt_cached()
        else:
            system_prompt = "You are a helpful, general-purpose AI assistant. Provide accurate, clear, and direct answers to the user's questions."
            context = ""
            initial_citations = []

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if session_id:
            for msg in get_session_messages(session_id):
                messages.append({"role": msg["role"], "content": msg["text"]})

        if is_rag:
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
        else:
            messages.append({"role": "user", "content": question})

        t_gen_start = time.perf_counter()
        with httpx.Client(timeout=300) as client:
            r = client.post(
                f"{config.OLLAMA_BASE}/api/chat",
                json={"model": config.LLM_MODEL, "messages": messages, "stream": False},
            )
            r.raise_for_status()
            res_data = r.json()
            answer = res_data.get("message", {}).get("content", "").strip()
            prompt_tokens = res_data.get("prompt_eval_count", 0)
            response_tokens = res_data.get("eval_count", 0)
        generate_latency = time.perf_counter() - t_gen_start

        # Post-processing inline citations and confidence score
        if is_rag:
            import re
            cited_indices = set()
            for m in re.finditer(r"\[(\d+)\]", answer):
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(initial_citations):
                    cited_indices.add(idx)
            cited_citations = []
            for idx in sorted(cited_indices):
                cited_citations.append(initial_citations[idx])
                
            if not cited_citations and "cannot find that information" not in answer.lower() and initial_citations:
                cited_citations = [initial_citations[0]]
                
            from chat_stream import calculate_confidence
            confidence = calculate_confidence(cited_citations)
        else:
            cited_citations = []
            confidence = 1.0

        # Save to semantic cache on cache MISS
        if is_rag and query_embeddings:
            from backend.cache.semantic_cache import set_semantic_cache
            set_semantic_cache(question, query_embeddings[0], answer, cited_citations, confidence)

        if session_id:
            append_session_turn(session_id, question, answer, cited_citations)

        return {
            "answer_text": answer,
            "answer": answer,  # legacy key compatibility
            "citations": cited_citations,
            "confidence": confidence,
            "cached": False,
            "session_id": session_id,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "request_id": request_id
        }
    finally:
        total_latency = time.perf_counter() - t_start
        if question:
            try:
                from backend.observability.logger import log_query
                log_query(
                    request_id=request_id,
                    query=question,
                    rewritten_query=rewritten_query,
                    retrieved_chunk_ids=retrieved_ids,
                    rerank_scores=rerank_scores,
                    cache_hit=cache_hit_flag,
                    embed_latency=embed_latency,
                    retrieve_latency=retrieve_latency,
                    rerank_latency=rerank_latency,
                    generate_latency=generate_latency,
                    total_latency=total_latency,
                    tokens_used=prompt_tokens + response_tokens
                )
            except Exception as le:
                logging.getLogger(__name__).error(f"Failed to log metrics in finally block: {le}")


def ask_stream(question: str, session_id: str | None = None, source_ids: list[str] | None = None):
    question = question.strip()
    if not question:
        raise ValueError("Question is empty")

    is_rag = source_ids is not None and len(source_ids) > 0 and not is_conversational_greeting(question)

    if is_rag:
        system_prompt = _ensure_system_prompt_cached()
        context, citations, from_cache = _retrieve_context(question, system_prompt, source_ids)
    else:
        system_prompt = "You are a helpful, general-purpose AI assistant. Provide accurate, clear, and direct answers to the user's questions."
        context = ""
        citations = []
        from_cache = False

    # Yield citations/cache status first
    yield json.dumps({"citations": citations, "cached": from_cache}) + "\n"

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if session_id:
        for msg in get_session_messages(session_id):
            messages.append({"role": msg["role"], "content": msg["text"]})

    if is_rag:
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    else:
        messages.append({"role": "user", "content": question})

    full_answer = []
    with httpx.Client(timeout=300) as client:
        with client.stream(
            "POST",
            f"{config.OLLAMA_BASE}/api/chat",
            json={"model": config.LLM_MODEL, "messages": messages, "stream": True},
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.strip():
                    continue
                chunk_data = json.loads(line)
                chunk_content = chunk_data.get("message", {}).get("content", "")
                if chunk_content:
                    full_answer.append(chunk_content)
                    yield json.dumps({"text": chunk_content}) + "\n"

    answer_str = "".join(full_answer).strip()
    if session_id:
        append_session_turn(session_id, question, answer_str, citations)
