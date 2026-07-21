"""Per-chunk entity/relation extraction via the local LLM (Ollama JSON mode).

Validates with Pydantic; on failure retries once with the validation error appended
("repair"); on a second failure logs and returns an empty graph so the caller can skip
the chunk without crashing the ingestion. The raw LLM call is isolated in
``_generate_json`` so tests can drive the whole flow without Ollama.
"""
import json
import logging
from typing import Literal

import httpx
from pydantic import BaseModel, ValidationError, field_validator

import config
from backend.prompt_templates import GRAPH_EXTRACT_TEMPLATE

logger = logging.getLogger(__name__)

EntityType = Literal["person", "org", "location", "concept", "event", "other"]


class Entity(BaseModel):
    name: str
    type: EntityType

    @field_validator("name")
    @classmethod
    def _nonempty(cls, v):
        if not v or not v.strip():
            raise ValueError("entity name is empty")
        return v.strip()


class Relation(BaseModel):
    source: str
    target: str
    predicate: str

    @field_validator("predicate")
    @classmethod
    def _short(cls, v):
        if len(v.split()) > 5:
            raise ValueError("predicate longer than 5 words")
        return v.strip()


class GraphExtraction(BaseModel):
    entities: list[Entity] = []
    relations: list[Relation] = []


def _extract_model() -> str:
    try:
        from backend.model_tiers import model_for
        return model_for("EXTRACT_MODEL")
    except Exception:
        return getattr(config, "EXTRACT_MODEL", "") or config.LLM_MODEL


def _generate_json(prompt: str, model: str) -> str:
    """One blocking Ollama generate call in JSON mode. Returns the raw response text."""
    r = httpx.post(
        f"{config.OLLAMA_BASE}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0},
            "keep_alive": getattr(config, "OLLAMA_KEEP_ALIVE", "10m"),
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _validate(raw: str) -> GraphExtraction:
    return GraphExtraction.model_validate_json(raw)


def extract_from_chunk(chunk_text: str, model: str | None = None) -> GraphExtraction:
    """Extract entities/relations from one chunk. Never raises — returns empty on failure."""
    model = model or _extract_model()
    prompt = GRAPH_EXTRACT_TEMPLATE.format(chunk=chunk_text)

    try:
        return _validate(_generate_json(prompt, model))
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        # Repair: tell the model exactly what was wrong and ask again.
        repair = f"{prompt}\n\nYour previous output was invalid: {e}\nReturn corrected JSON only."
        try:
            return _validate(_generate_json(repair, model))
        except Exception as e2:
            logger.warning(f"graph extraction failed after repair; skipping chunk: {e2}")
            return GraphExtraction()
    except Exception as e:
        logger.warning(f"graph extraction call failed; skipping chunk: {e}")
        return GraphExtraction()


def persist(extraction: GraphExtraction, chunk_id: str, doc_id: str) -> None:
    """Write an extraction into the graph store, deduping entities by norm_name."""
    from backend.graph import store

    name_to_id: dict[str, int] = {}
    for ent in extraction.entities:
        eid = store.upsert_entity(ent.name, ent.type)
        name_to_id[ent.name.strip().lower()] = eid
        store.link_entity_chunk(eid, chunk_id, doc_id)

    for rel in extraction.relations:
        sid = name_to_id.get(rel.source.strip().lower())
        tid = name_to_id.get(rel.target.strip().lower())
        if sid is None or tid is None:
            continue  # relation references an entity that didn't validate
        store.add_relation(sid, tid, rel.predicate, chunk_id, confidence=1.0)
