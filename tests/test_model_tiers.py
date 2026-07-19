import config
from backend import model_tiers


def test_detect_tier_boundaries():
    assert model_tiers.detect_tier(4) == "small"
    assert model_tiers.detect_tier(9.9) == "small"
    # 10 GB is the small/medium boundary and belongs to medium
    assert model_tiers.detect_tier(10) == "medium"
    assert model_tiers.detect_tier(15.8) == "medium"
    assert model_tiers.detect_tier(20) == "medium"
    # just past 20 GB moves to large
    assert model_tiers.detect_tier(20.1) == "large"
    assert model_tiers.detect_tier(64) == "large"


def test_detect_tier_falls_back_to_small_when_ram_unknown():
    assert model_tiers.detect_tier(0.0) == "small"


def test_bare_model_name_matches_latest_tag():
    installed = ["llama3.2:latest", "qwen2.5:0.5b"]
    assert model_tiers._is_installed("llama3.2", installed)
    assert model_tiers._is_installed("qwen2.5:0.5b", installed)
    assert not model_tiers._is_installed("qwen2.5:3b", installed)
    assert not model_tiers._is_installed("", installed)


def test_resolve_falls_back_when_tier_model_not_pulled(monkeypatch):
    """The acceptance case: only the original small model is available."""
    monkeypatch.setattr(config, "MODEL_TIER", "medium")
    monkeypatch.setattr(config, "LLM_MODEL", "qwen2.5:0.5b")
    monkeypatch.setattr(config, "REWRITE_MODEL", "")
    monkeypatch.setattr(config, "EXTRACT_MODEL", "")

    resolved = model_tiers.resolve_models(installed=["qwen2.5:0.5b"])

    # medium tier wants qwen2.5:3b for generation, which is not pulled
    assert resolved["LLM_MODEL"] == "qwen2.5:0.5b"
    assert all(m == "qwen2.5:0.5b" for m in resolved.values())


def test_resolve_uses_tier_models_when_available(monkeypatch):
    monkeypatch.setattr(config, "MODEL_TIER", "medium")
    monkeypatch.setattr(config, "LLM_MODEL", "qwen2.5:0.5b")
    monkeypatch.setattr(config, "REWRITE_MODEL", "")
    monkeypatch.setattr(config, "EXTRACT_MODEL", "")

    resolved = model_tiers.resolve_models(
        installed=["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"]
    )

    assert resolved["LLM_MODEL"] == "qwen2.5:3b"
    assert resolved["REWRITE_MODEL"] == "qwen2.5:0.5b"
    assert resolved["EXTRACT_MODEL"] == "qwen2.5:1.5b"


def test_explicit_model_overrides_tier(monkeypatch):
    monkeypatch.setattr(config, "MODEL_TIER", "medium")
    monkeypatch.setattr(config, "LLM_MODEL", "llama3.2:latest")  # differs from shipped default
    monkeypatch.setattr(config, "REWRITE_MODEL", "")
    monkeypatch.setattr(config, "EXTRACT_MODEL", "")

    resolved = model_tiers.resolve_models(installed=["llama3.2:latest", "qwen2.5:3b"])

    assert resolved["LLM_MODEL"] == "llama3.2:latest"


def test_missing_models_reports_unpulled_tier_models(monkeypatch):
    monkeypatch.setattr(config, "MODEL_TIER", "medium")
    monkeypatch.setattr(config, "LLM_MODEL", "qwen2.5:0.5b")
    monkeypatch.setattr(config, "REWRITE_MODEL", "")
    monkeypatch.setattr(config, "EXTRACT_MODEL", "")

    missing = model_tiers.missing_models(installed=["qwen2.5:0.5b"])

    assert "qwen2.5:3b" in missing
    assert "qwen2.5:1.5b" in missing
    assert "qwen2.5:0.5b" not in missing


def test_missing_models_empty_when_ollama_unreachable(monkeypatch):
    """No tag list means we cannot tell -- report nothing rather than everything."""
    monkeypatch.setattr(config, "MODEL_TIER", "medium")
    assert model_tiers.missing_models(installed=[]) == []
