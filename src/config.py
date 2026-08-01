""" Central environment configs & API key validation """

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# text-embedding-004 / gemini-2.5-flash blocked for many new free-tier keys.
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash-lite"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVER_K = 4
MAX_AGENT_ITERATIONS = 4
MAX_INDEXED_PDFS = 4

# Bumped when embedding model changed so old incompatible vectors are not reused.
CHROMA_COLLECTION = "finsight_ai_financial_docs_v1"
EMBED_BATCH_SIZE = 40


def get_gemini_api_key() -> str:
    """ return GEMINI_API_KEY or raise a clear configuration error """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        raise ValueError(
            "GEMINI_API_KEY is missing or unset. "
            "Copy .env.example to .env and set a valid free-tier Gemini API key."
        )

    return api_key


def ensure_directories() -> None:
    """ create data & chroma persistence directories if needed """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
