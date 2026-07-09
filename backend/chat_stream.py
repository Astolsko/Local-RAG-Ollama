import json
import time
import re
import math
import logging
import asyncio
import httpx
import config
from backend.prompt_templates import SYSTEM_PROMPT_TEMPLATE, LLM_FAITHFULNESS_JUDGE_TEMPLATE
from backend.retrieval.hybrid_search import hybrid_search_sync
from redis_store import get_session_messages, append_session_turn

logger = logging.getLogger(__name__)

OBSERVABILITY_LOG = config.DATA_DIR / "observability.log"

def is_conversational_greeting(question: str) -> bool:
    q = question.strip().lower().rstrip("?.!")
    greetings = {"hi", "hello", "hey", "yo", "hola", "greetings", "good morning", "good afternoon", "good evening"}
    return q in greetings or q.startswith(("hi ", "hello ", "hey "))

def calculate_confidence(cited_chunks: list[dict]) -> float:
    """Calculate confidence score heuristically from vector distance or rerank score."""
    if not cited_chunks:
        return 0.0
    scores = []
    for chunk in cited_chunks:
        # Check for rerank score (higher is better)
        rerank_score = chunk.get("rerank_score")
        if rerank_score is not None:
            # Convert cross-encoder logit score to probability via sigmoid
            prob = 1 / (1 + math.exp(-rerank_score))
            scores.append(prob)
        else:
            # Fallback to vector distance (Chroma cosine distance: 0.0 is exact match, 2.0 is opposite)
            dist = chunk.get("distance")
            if dist is not None:
                sim = 1.0 - (dist / 2.0)
                scores.append(max(0.0, min(1.0, sim)))
            else:
                # Fallback default score if neither is available
                scores.append(0.85)
    return float(round(sum(scores) / len(scores), 2))

async def run_background_judge(query: str, answer: str, cited_chunks: list[dict], request_id: str):
    """Async background task that evaluates answer faithfulness using the local LLM-judge."""
    if not cited_chunks:
        return
        
    context = "\n\n".join(f"[{c['source_name']}] {c['snippet']}" for c in cited_chunks)
    prompt = LLM_FAITHFULNESS_JUDGE_TEMPLATE.format(
        context=context,
        query=query,
        answer=answer
    )
    
    try:
        # Give judge runs a low priority, run after response is complete
        await asyncio.sleep(0.5)
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{config.OLLAMA_BASE}/api/generate",
                json={
                    "model": config.LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0}
                }
            )
            r.raise_for_status()
            response_text = r.json().get("response", "").strip()
            
            # Parse faithfulness score (1-5)
            score = 3  # default fallback
            match = re.search(r"faithfulness.*?(\d)", response_text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                
            # Log to observability log file
            log_entry = {
                "timestamp": datetime_string(),
                "query": query,
                "score": score,
                "verdict": response_text
            }
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(OBSERVABILITY_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                
            logger.info(f"Observability Faithfulness Judge Score: {score}/5")
            
            # Update SQLite request logs with the faithfulness score
            try:
                from backend.observability.metrics_db import update_faithfulness
                update_faithfulness(request_id, score)
            except Exception as se:
                logger.error(f"Failed to save faithfulness score to SQLite: {se}")
    except Exception as e:
        logger.error(f"Failed to run background faithfulness judge: {e}")

def datetime_string() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat() + "Z"

async def chat_stream(question: str, session_id: str | None = None, source_ids: list[str] | None = None):
    # ponytail: we use a try-finally block to ensure metrics logging runs even on client disconnects
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
    answer_str = ""
    refusal_flag = False
    
    question = question.strip()
    if not question:
        raise ValueError("Question is empty")
        
    try:
        # Check if we should use RAG
        from backend.rag import _collection
        is_rag = source_ids is not None and len(source_ids) > 0 and _collection.count() > 0 and not is_conversational_greeting(question)
        
        query_embeddings = None
        if is_rag:
            try:
                # Compute query embedding (reused for caching and retrieval)
                t_embed_start = time.perf_counter()
                from backend.rag import OllamaEmbed
                embed_fn = OllamaEmbed()
                query_embeddings = embed_fn([question])
                embed_latency = time.perf_counter() - t_embed_start
                
                # Check semantic cache
                from backend.cache.semantic_cache import check_semantic_cache, set_semantic_cache
                cache_hit = check_semantic_cache(question, query_embeddings[0])
                if cache_hit:
                    cache_hit_flag = True
                    cited_citations = cache_hit["citations"]
                    retrieved_ids = [c["chunk_id"] for c in cited_citations]
                    # Yield cached citations
                    yield json.dumps({"citations": cache_hit["citations"], "cached": True, "request_id": request_id}) + "\n"
                    
                    # Simulate token streaming of the cached response
                    words = cache_hit["answer_text"].split(" ")
                    for w in words:
                        yield json.dumps({"text": w + " "}) + "\n"
                        await asyncio.sleep(0.01)
                        
                    # Yield final response payload
                    yield json.dumps({
                        "citations": cache_hit["citations"],
                        "cached": True,
                        "confidence": cache_hit["confidence"],
                        "answer_text": cache_hit["answer_text"],
                        "prompt_tokens": 0,
                        "response_tokens": 0,
                        "request_id": request_id
                    }) + "\n"
                    
                    # Save session history in Redis
                    if session_id:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            None,
                            lambda: append_session_turn(session_id, question, cache_hit["answer_text"], cache_hit["citations"])
                        )
                    return
                    
                # 1. Fetch relevant chunks using hybrid search
                loop = asyncio.get_running_loop()
                docs, metas, ids, search_metrics = await loop.run_in_executor(
                    None, 
                    lambda: hybrid_search_sync(question, top_k=min(config.TOP_K, _collection.count()))
                )
                
                retrieve_latency = search_metrics["retrieve_latency"]
                rerank_latency = search_metrics["rerank_latency"]
                rerank_scores = search_metrics["rerank_scores"]
                retrieved_ids = ids
                
                if search_metrics["rewritten_queries"] and len(search_metrics["rewritten_queries"]) > 1:
                    rewritten_query = ", ".join(search_metrics["rewritten_queries"][1:])
                
                if not docs:
                    # Fallback if search returns nothing
                    is_rag = False
            except Exception as e:
                logger.error(f"Error during RAG retrieval/caching: {e}")
                yield json.dumps({"citations": [], "cached": False}) + "\n"
                yield json.dumps({"text": f"\n[System Error during retrieval: {e}]"}) + "\n"
                return
                
        if is_rag:
            # Prepare structured context for Ollama (with prefixes [Chunk 1], [Chunk 2])
            context_parts = []
            initial_citations = []
            for idx, (doc, meta, chunk_id) in enumerate(zip(docs, metas, ids)):
                context_parts.append(f"[Chunk {idx + 1}] (Source: {meta['source_name']})\n{doc}")
                initial_citations.append({
                    # Structured response fields
                    "chunk_id": chunk_id,
                    "source_file": meta.get("source_file") or meta.get("source_name", "Unknown"),
                    "snippet": doc,
                    
                    # UI compatibility fields
                    "source_id": meta.get("source_id", ""),
                    "source_name": meta["source_name"],
                    "page_number": meta.get("page_number", 1),
                    "topic": meta.get("topic", "General Context"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "rerank_score": meta.get("rerank_score")
                })
                
            context = "\n\n".join(context_parts)
            system_prompt = SYSTEM_PROMPT_TEMPLATE
            
            # Yield initial citations (UI expectation)
            yield json.dumps({"citations": initial_citations, "cached": False, "request_id": request_id}) + "\n"
            
            # Prepare messages
            messages = [{"role": "system", "content": system_prompt}]
            if session_id:
                # We must load session messages inside run_in_executor to avoid blocking
                sess_msgs = await loop.run_in_executor(None, lambda: get_session_messages(session_id))
                for msg in sess_msgs:
                    messages.append({"role": msg["role"], "content": msg["text"]})
                    
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
        else:
            system_prompt = "You are a helpful, general-purpose AI assistant. Provide accurate, clear, and direct answers to the user's questions."
            messages = [{"role": "system", "content": system_prompt}]
            if session_id:
                loop = asyncio.get_running_loop()
                sess_msgs = await loop.run_in_executor(None, lambda: get_session_messages(session_id))
                for msg in sess_msgs:
                    messages.append({"role": msg["role"], "content": msg["text"]})
            messages.append({"role": "user", "content": question})
            initial_citations = []
            yield json.dumps({"citations": [], "cached": False, "request_id": request_id}) + "\n"
    
        # 2. Stream tokens from Ollama
        full_answer = []
        t_gen_start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{config.OLLAMA_BASE}/api/chat",
                    json={"model": config.LLM_MODEL, "messages": messages, "stream": True}
                ) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        chunk_data = json.loads(line)
                        if "prompt_eval_count" in chunk_data:
                            prompt_tokens = chunk_data["prompt_eval_count"]
                        if "eval_count" in chunk_data:
                            response_tokens = chunk_data["eval_count"]
                        chunk_content = chunk_data.get("message", {}).get("content", "")
                        if chunk_content:
                            full_answer.append(chunk_content)
                            yield json.dumps({"text": chunk_content}) + "\n"
        except Exception as e:
            logger.error(f"Error during Ollama token stream: {e}")
            yield json.dumps({"text": f"\n[Error streaming answer: {e}]"}) + "\n"
            
        generate_latency = time.perf_counter() - t_gen_start
        answer_str = "".join(full_answer).strip()
        refusal_flag = "cannot find that information" in answer_str.lower()
    
        # 3. Post-processing: Parse citations and compute confidence
        if is_rag:
            # Find all cited indices [1], [2], etc.
            cited_indices = set()
            for m in re.finditer(r"\[(\d+)\]", answer_str):
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(initial_citations):
                    cited_indices.add(idx)
                    
            cited_citations = []
            for idx in sorted(cited_indices):
                cited_citations.append(initial_citations[idx])
                
            # Default to top chunk if no citations were generated but it was not a refusal
            if not cited_citations and "cannot find that information" not in answer_str.lower() and initial_citations:
                cited_citations = [initial_citations[0]]
                
            # Compute confidence score
            confidence = calculate_confidence(cited_citations)
            
            # Yield the final citations, confidence, and complete structured response
            yield json.dumps({
                "citations": cited_citations,
                "cached": False,
                "confidence": confidence,
                "answer_text": answer_str,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "request_id": request_id
            }) + "\n"
            
            # Save to semantic cache on cache MISS
            if query_embeddings:
                from backend.cache.semantic_cache import set_semantic_cache
                set_semantic_cache(question, query_embeddings[0], answer_str, cited_citations, confidence)
            
            # Save session history in Redis (via run_in_executor to avoid blocking)
            if session_id:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: append_session_turn(session_id, question, answer_str, cited_citations)
                )
                
            # 4. Trigger async faithfulness judge
            asyncio.create_task(run_background_judge(question, answer_str, cited_citations, request_id))
        else:
            # Non-RAG final response
            yield json.dumps({
                "citations": [],
                "cached": False,
                "confidence": 1.0,
                "answer_text": answer_str,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "request_id": request_id
            }) + "\n"
            
            if session_id:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: append_session_turn(session_id, question, answer_str, [])
                )
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
                    tokens_used=prompt_tokens + response_tokens,
                    refusal=refusal_flag
                )
            except Exception as le:
                logger.error(f"Failed to log metrics in finally block: {le}")
