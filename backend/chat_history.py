import json
import uuid
from datetime import datetime, timezone
from typing import Any

try:
    from backend.config import CHAT_HISTORY_FILE, DATA_DIR
except ImportError:
    from config import CHAT_HISTORY_FILE, DATA_DIR


def _load() -> list[dict[str, Any]]:
    if not CHAT_HISTORY_FILE.exists():
        return []
    try:
        content = CHAT_HISTORY_FILE.read_text(encoding="utf-8")
        if not content.strip():
            return []
        return json.loads(content)
    except Exception:
        return []


def _save(chats: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHAT_HISTORY_FILE.write_text(json.dumps(chats, indent=2), encoding="utf-8")


def _auto_title(messages: list[dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            text = m.get("text", "")
            return text[:60] + ("…" if len(text) > 60 else "")
    return "Untitled chat"


def list_chats() -> list[dict[str, Any]]:
    return [
        {
            "id": c["id"],
            "title": c["title"],
            "created_at": c["created_at"],
            "message_count": len(c.get("messages", [])),
        }
        for c in sorted(_load(), key=lambda x: x["created_at"], reverse=True)
    ]


def get_chat(chat_id: str) -> dict[str, Any] | None:
    return next((c for c in _load() if c["id"] == chat_id), None)


def save_chat(title: str | None, messages: list[dict[str, Any]], system_prompt: str) -> dict[str, Any]:
    chat = {
        "id": str(uuid.uuid4()),
        "title": (title or "").strip() or _auto_title(messages),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system_prompt": system_prompt,
        "messages": messages,
    }
    chats = _load()
    chats.append(chat)
    _save(chats)
    return chat


def delete_chat(chat_id: str) -> bool:
    chats = _load()
    kept = [c for c in chats if c["id"] != chat_id]
    if len(kept) == len(chats):
        return False
    _save(kept)
    return True
