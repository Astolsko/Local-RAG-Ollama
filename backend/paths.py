"""Single source of truth for where the app stores its data.

Change ONE value here (or set the RAG_DATA_DIR env var) to move ALL persistent data —
Chroma vectors, the SQLite DBs, settings.json, chat history, the BM25 index, and logs —
between your local machine and, e.g., Google Drive on Colab.

Everything else in the codebase reads paths from `config` (config.DATA_DIR, config.CHROMA_DIR,
...), and `config` derives them from here. You never edit anything but this file.

Precedence (highest wins):
  1. RAG_DATA_DIR environment variable   — used by the eval harness for isolation
  2. DATA_DIR_OVERRIDE below              — set this for Colab / a custom location
  3. <repo>/data                          — default for local runs
"""
import os
from pathlib import Path

# Repo root (…/RAG-LLM-Ollama), derived from this file's location — do not edit.
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── EDIT THIS to relocate all data storage ──────────────────────────────────────
# Local machine (default): leave as None  ->  <repo>/data
#
# Google Colab + Drive, for example:
#     from google.colab import drive; drive.mount('/content/drive')
#     DATA_DIR_OVERRIDE = "/content/drive/MyDrive/MyProjectOutputs/data"
#
# Accepts a str or a Path; None means "use the default".
# Colab: data persists on Drive (mount it first: drive.mount('/content/drive')).
# Set back to None when running on the local machine.
DATA_DIR_OVERRIDE = None
# ────────────────────────────────────────────────────────────────────────────────


def data_dir() -> Path:
    """Resolve the active data directory using the precedence above."""
    env = os.environ.get("RAG_DATA_DIR")
    if env:
        return Path(env)
    if DATA_DIR_OVERRIDE:
        return Path(DATA_DIR_OVERRIDE)
    return REPO_ROOT / "data"


# Location of the eval assets (golden set + history.csv). These live inside the repo, so
# they follow the code, not DATA_DIR_OVERRIDE. Anchored to the repo root so they resolve
# regardless of the current working directory.
EVAL_DIR = REPO_ROOT / "eval"
