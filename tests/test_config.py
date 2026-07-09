import os
from backend.config import DEFAULT_SYSTEM_PROMPT, OLLAMA_BASE

def test_config_defaults():
    assert OLLAMA_BASE == "http://localhost:11434"
    assert "study and Q&A assistant" in DEFAULT_SYSTEM_PROMPT
