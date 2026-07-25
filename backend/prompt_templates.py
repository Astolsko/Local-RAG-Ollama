# NOTE: the chat system prompt does NOT live here. There is exactly one source of truth:
# `config.DEFAULT_SYSTEM_PROMPT`, overridden by `data/system_prompt.txt` when the user edits
# it, resolved through `rag.get_system_prompt()`. A near-identical `SYSTEM_PROMPT_TEMPLATE`
# used to sit here and was what `chat_stream` actually sent, which is why editing the prompt
# in the UI had no effect on the main chat. Do not reintroduce it.

# Template for background faithfulness judge (observability)
LLM_FAITHFULNESS_JUDGE_TEMPLATE = """You are an expert AI evaluator checking for hallucinations. Rate the generated answer against the retrieved context on a scale of 1 to 5 (integer only) for Faithfulness (no hallucination vs retrieved context).

Context:
{context}

Query:
{query}

Generated Answer:
{answer}

Rate the answer on a scale of 1 to 5 where:
- 5: Fully faithful, no hallucinations, completely supported by the context.
- 4: Mostly faithful, minor details are unsupported but no major hallucinations.
- 3: Partially faithful, some parts are supported, but contains major unsupported statements.
- 2: Mostly unfaithful, contains major hallucinations.
- 1: Completely unfaithful, hallucinated, or relies entirely on outside knowledge.

Response format:
Faithfulness: <score>
"""

# Template for checking if a cited chunk actually supports the statement citing it
LLM_CITATION_JUDGE_TEMPLATE = """You are an expert evaluator checking citation accuracy.
Analyze the Statement and the Cited Chunk below. Does the Cited Chunk actually support the claims made in the Statement?

Statement:
{statement}

Cited Chunk:
{chunk}

Reply exactly with either "Yes" or "No". Do not write any other explanation or text.
"""


# GraphRAG-lite entity/relation extraction (Ollama format="json").
GRAPH_EXTRACT_TEMPLATE = """You extract a knowledge graph from a text chunk. Return ONLY JSON.

Rules:
- Entities must appear verbatim or near-verbatim in the chunk.
- At most 10 entities and 10 relations.
- Entity type is one of: person, org, location, concept, event, other.
- Each relation's predicate is a verb phrase of at most 5 words.
- source and target of every relation must be names present in your entities list.

Return exactly this shape:
{{"entities":[{{"name":"...","type":"person|org|location|concept|event|other"}}],
 "relations":[{{"source":"...","target":"...","predicate":"<verb phrase>"}}]}}

Chunk:
{chunk}
"""
