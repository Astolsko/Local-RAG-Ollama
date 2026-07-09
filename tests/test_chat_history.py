import pytest
import json
from pathlib import Path

def test_chat_history_save_and_load(tmp_path, monkeypatch):
    # Re-route chat history dir to a temp directory
    monkeypatch.setattr("backend.chat_history.CHAT_HISTORY_FILE", tmp_path / "chat_history.json")
    monkeypatch.setattr("backend.chat_history.DATA_DIR", tmp_path)

    from backend.chat_history import save_chat, list_chats, get_chat, delete_chat

    chats = list_chats()
    assert len(chats) == 0

    # Save chat
    saved = save_chat("My Title", [{"role": "user", "text": "hi"}], "system prompt")
    assert saved["title"] == "My Title"
    assert len(saved["messages"]) == 1

    # List chats
    chats = list_chats()
    assert len(chats) == 1
    assert chats[0]["id"] == saved["id"]

    # Get chat
    c = get_chat(saved["id"])
    assert c["system_prompt"] == "system prompt"

    # Delete chat
    deleted = delete_chat(saved["id"])
    assert deleted is True
    assert len(list_chats()) == 0
