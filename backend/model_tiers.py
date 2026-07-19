"""Hardware-aware model tier selection.

Picks per-role models based on available RAM, and degrades gracefully: if the
tier's model is not pulled in Ollama, the configured LLM_MODEL is used instead
and a warning is logged. The app therefore keeps working on a machine that only
has the original small model.

Resolution order for each role:
    1. explicit override (env var or settings.json value)
    2. the tier's model for that role
    3. fallback to an installed model, with a logged warning
"""
import logging
import time

import httpx

import config

logger = logging.getLogger(__name__)

# Per-role models by tier. Roles: generation, query rewriting, entity extraction.
TIERS: dict[str, dict[str, str]] = {
    "small":  {"LLM_MODEL": "qwen2.5:1.5b", "REWRITE_MODEL": "qwen2.5:0.5b", "EXTRACT_MODEL": "qwen2.5:1.5b"},
    "medium": {"LLM_MODEL": "qwen2.5:3b",   "REWRITE_MODEL": "qwen2.5:0.5b", "EXTRACT_MODEL": "qwen2.5:1.5b"},
    "large":  {"LLM_MODEL": "qwen2.5:7b",   "REWRITE_MODEL": "qwen2.5:1.5b", "EXTRACT_MODEL": "qwen2.5:3b"},
}

ROLES = ("LLM_MODEL", "REWRITE_MODEL", "EXTRACT_MODEL")

_TAGS_CACHE: tuple[float, list[str]] | None = None
_TAGS_TTL = 60.0


def total_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception as e:
        logger.warning(f"Could not detect RAM ({e}); assuming small tier")
        return 0.0


def detect_tier(ram_gb: float | None = None) -> str:
    """<10 GB -> small, 10-20 GB -> medium, >20 GB -> large."""
    if ram_gb is None:
        ram_gb = total_ram_gb()
    if ram_gb < 10:
        return "small"
    if ram_gb <= 20:
        return "medium"
    return "large"


def active_tier() -> str:
    """The configured tier, resolving MODEL_TIER=auto against detected hardware."""
    tier = str(getattr(config, "MODEL_TIER", "auto") or "auto").lower()
    if tier in TIERS:
        return tier
    if tier != "auto":
        logger.warning(f"Unknown MODEL_TIER '{tier}'; falling back to auto detection")
    return detect_tier()


def installed_models(refresh: bool = False) -> list[str]:
    """Model tags currently pulled in Ollama. Cached briefly -- this sits on the
    request path via /api/health."""
    global _TAGS_CACHE
    now = time.monotonic()
    if not refresh and _TAGS_CACHE and (now - _TAGS_CACHE[0]) < _TAGS_TTL:
        return _TAGS_CACHE[1]
    try:
        r = httpx.get(f"{config.OLLAMA_BASE}/api/tags", timeout=5.0)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        logger.warning(f"Could not list Ollama models: {e}")
        names = []
    _TAGS_CACHE = (now, names)
    return names


def _is_installed(model: str, installed: list[str]) -> bool:
    """Ollama reports 'qwen2.5:0.5b' and 'llama3.2:latest'; treat a bare name as
    matching its ':latest' tag."""
    if not model:
        return False
    if model in installed:
        return True
    if ":" not in model:
        return f"{model}:latest" in installed
    return False


def _override(role: str) -> str | None:
    """An explicitly chosen model, i.e. one that differs from the shipped default.
    config.<role> already resolves env var first, then settings.json."""
    value = getattr(config, role, "") or ""
    shipped_default = config.DEFAULT_SETTINGS.get(role, "")
    return value if value and value != shipped_default else None


def resolve_models(installed: list[str] | None = None) -> dict[str, str]:
    """Resolve every role to a concrete model tag, degrading to an installed one."""
    if installed is None:
        installed = installed_models()
    tier = active_tier()
    resolved: dict[str, str] = {}

    for role in ROLES:
        wanted = _override(role) or TIERS[tier][role]
        if installed and not _is_installed(wanted, installed):
            fallback = config.LLM_MODEL
            if not _is_installed(fallback, installed):
                fallback = installed[0]
            logger.warning(
                f"{role} '{wanted}' (tier '{tier}') is not pulled in Ollama; "
                f"using '{fallback}'. Run: ollama pull {wanted}"
            )
            wanted = fallback
        resolved[role] = wanted
    return resolved


def missing_models(installed: list[str] | None = None) -> list[str]:
    """Tier models that would be used if they were pulled, but are not."""
    if installed is None:
        installed = installed_models()
    if not installed:
        return []
    tier = active_tier()
    wanted = {_override(role) or TIERS[tier][role] for role in ROLES}
    return sorted(m for m in wanted if not _is_installed(m, installed))


def model_for(role: str) -> str:
    """Concrete model tag for a role, e.g. model_for('REWRITE_MODEL')."""
    return resolve_models().get(role, config.LLM_MODEL)


def status() -> dict:
    """Tier summary for /api/health."""
    installed = installed_models()
    ram = total_ram_gb()
    return {
        "tier": active_tier(),
        "ram_gb": round(ram, 1),
        "models": resolve_models(installed),
        "missing_models": missing_models(installed),
    }
