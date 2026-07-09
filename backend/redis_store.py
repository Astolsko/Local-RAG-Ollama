import hashlib
import json
from typing import Any

import redis

import config

_client: redis.Redis | None = None
_last_redis_url: str | None = None


def get_redis() -> redis.Redis:
    global _client, _last_redis_url
    current_url = config.REDIS_URL
    if _client is None or _last_redis_url != current_url:
        _client = redis.from_url(current_url, decode_responses=True)
        _last_redis_url = current_url
    return _client


def ping() -> bool:
    try:
        return bool(get_redis().ping())
    except redis.RedisError:
        return False


def require_redis() -> redis.Redis:
    r = get_redis()
    if not r.ping():
        raise redis.RedisError("Redis not reachable")
    return r


def _session_key(session_id: str) -> str:
    return f"chat:session:{session_id}"


def get_session_messages(session_id: str) -> list[dict[str, Any]]:
    raw = require_redis().get(_session_key(session_id))
    return json.loads(raw) if raw else []


def set_session_messages(session_id: str, messages: list[dict[str, Any]]) -> None:
    require_redis().setex(_session_key(session_id), config.SESSION_TTL, json.dumps(messages))


def append_session_turn(session_id: str, user_text: str, assistant_text: str, citations: list[dict]) -> None:
    msgs = get_session_messages(session_id)
    msgs.append({"role": "user", "text": user_text})
    msgs.append({"role": "assistant", "text": assistant_text, "citations": citations})
    set_session_messages(session_id, msgs)


def clear_session(session_id: str) -> None:
    require_redis().delete(_session_key(session_id))


def _rag_cache_key(question: str, source_count: int, system_prompt: str, source_ids: list[str] | None = None) -> str:
    ids_str = ",".join(sorted(source_ids)) if source_ids else ""
    blob = f"{question.strip().lower()}|{source_count}|{hashlib.sha256(system_prompt.encode()).hexdigest()[:16]}|{ids_str}"
    return f"prompt:rag:{hashlib.sha256(blob.encode()).hexdigest()}"


def get_cached_rag(question: str, source_count: int, system_prompt: str, source_ids: list[str] | None = None) -> dict[str, Any] | None:
    raw = require_redis().get(_rag_cache_key(question, source_count, system_prompt, source_ids))
    return json.loads(raw) if raw else None


def set_cached_rag(question: str, source_count: int, system_prompt: str, data: dict[str, Any], source_ids: list[str] | None = None) -> None:
    require_redis().setex(
        _rag_cache_key(question, source_count, system_prompt, source_ids),
        config.PROMPT_CACHE_TTL,
        json.dumps(data),
    )


SYSTEM_PROMPT_KEY = "prompt:system"


def cache_system_prompt(text: str) -> None:
    require_redis().set(SYSTEM_PROMPT_KEY, text)


def get_cached_system_prompt() -> str | None:
    return require_redis().get(SYSTEM_PROMPT_KEY)
